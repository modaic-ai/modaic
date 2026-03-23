from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

try:
    import httpx
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch requires `httpx` for Fireworks batch jobs. Install it with `uv add "modaic[fireworks]"`.'
    ) from exc

from ..storage import ensure_batch_storage_dirs
from ..types import BatchReponse, BatchRequest, ResultItem
from .base import CLEANUP, BatchClient, _extract_openai_compatible_message, _retry_on_network_error, logger


def _check_firectl() -> str:
    """Return the path to firectl or raise with install instructions."""
    path = shutil.which("firectl")
    if path is None:
        raise RuntimeError(
            "firectl CLI is required for Fireworks batch jobs but was not found on PATH.\n"
            "Install it from https://docs.fireworks.ai/tools-sdks/firectl/firectl"
        )
    return path


class FireworksBatchClient(BatchClient):
    BASE_URL = "https://api.fireworks.ai/v1"
    TARGET_BATCH_BYTES = 450 * 1024 * 1024  # 450 MB safety margin under 500 MB limit
    provider: str = "fireworks_ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        resolved_api_key = api_key or os.getenv("FIREWORKS_AI_API_KEY")
        if not resolved_api_key:
            raise ValueError("FIREWORKS_AI_API_KEY environment variable is not set")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )

        self.account_id = account_id or os.getenv("FIREWORKS_ACCOUNT_ID")
        if not self.account_id:
            raise ValueError("FIREWORKS_ACCOUNT_ID environment variable is not set")

        self._api_key = resolved_api_key
        self._firectl = _check_firectl()
        self._output_dataset_ids: dict[str, str] = {}

    def _get_headers(self, content_type: str = "application/json") -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def format(self, batch_request: BatchRequest) -> list[dict]:
        return [
            {
                "custom_id": request["id"],
                "body": {"messages": request["messages"], **request["lm_kwargs"]},
            }
            for request in batch_request["requests"]
        ]

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        response = raw_result.get("response", {})
        choices = response.get("choices", [])

        if not choices:
            error = raw_result.get("error") or response.get("error")
            raise ValueError(f"Batch request failed: {error}")

        choice = choices[0]
        message = choice.get("message", {})

        text, reasoning_content = _extract_openai_compatible_message(message)
        result: ResultItem = {"text": text}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        if "logprobs" in choice and choice["logprobs"] is not None:
            result["logprobs"] = choice["logprobs"]
        if "tool_calls" in message and message["tool_calls"] is not None:
            result["tool_calls"] = message["tool_calls"]

        return result

    def get_batches(self, batch_request: BatchRequest) -> list[BatchRequest]:
        formatted = self.format(batch_request)
        requests = batch_request["requests"]

        batches: list[BatchRequest] = []
        current_requests: list[dict] = []
        current_size = 0

        for req, fmt in zip(requests, formatted):
            line_size = len(json.dumps(fmt).encode("utf-8")) + 1  # +1 for newline
            if current_size + line_size > self.TARGET_BATCH_BYTES and current_requests:
                batches.append(BatchRequest(
                    requests=current_requests,
                    model=batch_request.get("model"),
                    lm_kwargs=batch_request.get("lm_kwargs"),
                ))
                current_requests = []
                current_size = 0
            current_requests.append(req)
            current_size += line_size

        if current_requests:
            batches.append(BatchRequest(
                requests=current_requests,
                model=batch_request.get("model"),
                lm_kwargs=batch_request.get("lm_kwargs"),
            ))

        if len(batches) > 1:
            logger.info(
                "Fireworks get_batches: split %d requests into %d chunks (target %d MB each)",
                len(requests),
                len(batches),
                self.TARGET_BATCH_BYTES // (1024 * 1024),
            )

        return batches

    async def _upload_dataset_firectl(self, dataset_id: str, jsonl_path: Path) -> None:
        """Upload a JSONL file to Fireworks using firectl CLI."""
        cmd = [
            self._firectl, "dataset", "create",
            "--account-id", self.account_id,
            "--quiet",
            dataset_id, str(jsonl_path),
        ]
        logger.debug("Fireworks firectl upload: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ValueError(
                f"firectl dataset create failed (exit {proc.returncode}): {stderr.decode().strip()}"
            )
        logger.debug("firectl upload complete: dataset_id=%s stdout=%s", dataset_id, stdout.decode().strip())

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        if batch_request["model"] is None:
            raise ValueError("Fireworks batch requires all requests to use the same model")

        jsonl_path = self.create_jsonl(batch_request)

        async def _do_submit() -> str:
            dataset_id = f"batch-input-{uuid.uuid4().hex[:8]}"
            output_dataset_id = f"batch-output-{uuid.uuid4().hex[:8]}"
            job_id = f"batch-job-{uuid.uuid4().hex[:8]}"

            # Upload dataset via firectl (bypasses Cloudflare worker limits)
            await self._upload_dataset_firectl(dataset_id, jsonl_path)

            # Wait for dataset to become READY, then create batch job via HTTP API
            async with httpx.AsyncClient(timeout=120.0) as client:
                dataset_status_url = f"{self.BASE_URL}/accounts/{self.account_id}/datasets/{dataset_id}"
                for _ in range(120):
                    logger.debug("Fireworks request: GET %s (dataset status)", dataset_status_url)
                    resp = await client.get(dataset_status_url, headers=self._get_headers())
                    resp.raise_for_status()
                    state = resp.json().get("state", "")
                    if state == "READY":
                        break
                    if state not in ("UPLOADING", "STATE_UNSPECIFIED"):
                        raise ValueError(f"Dataset in unexpected state: {state}")
                    await asyncio.sleep(2)
                else:
                    raise TimeoutError(f"Dataset {dataset_id} did not become READY within 240 seconds")

                model = batch_request["model"]
                if not model.startswith("accounts/"):
                    model = f"accounts/fireworks/models/{model}"

                create_job_url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs"
                logger.debug("Fireworks request: POST %s (create job)", create_job_url)
                create_job_payload = {
                    "displayName": f"Batch job {job_id}",
                    "model": model,
                    "inputDatasetId": f"accounts/{self.account_id}/datasets/{dataset_id}",
                    "outputDatasetId": f"accounts/{self.account_id}/datasets/{output_dataset_id}",
                }

                lm_kwargs = batch_request.get("lm_kwargs") or {}
                if lm_kwargs:
                    inference_params = {}
                    if "max_tokens" in lm_kwargs:
                        inference_params["maxTokens"] = lm_kwargs["max_tokens"]
                    if "temperature" in lm_kwargs:
                        inference_params["temperature"] = lm_kwargs["temperature"]
                    if "top_p" in lm_kwargs:
                        inference_params["topP"] = lm_kwargs["top_p"]
                    if inference_params:
                        create_job_payload["inferenceParameters"] = inference_params

                resp = await client.post(
                    create_job_url,
                    headers=self._get_headers(),
                    json=create_job_payload,
                    params={"batchInferenceJobId": job_id},
                )
                if resp.status_code != 200:
                    raise ValueError(f"Failed to create batch job: {resp.status_code} - {resp.text}")
                resp.raise_for_status()
                job_response = resp.json()

                batch_id = job_response.get("name", "").split("/")[-1] or job_id
                self._output_dataset_ids[batch_id] = output_dataset_id
                logger.debug("Fireworks submit: created batch_id=%s", batch_id)
                return batch_id

        try:
            return await _retry_on_network_error(_do_submit, provider_name=self.provider, max_retries=7)
        finally:
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs/{batch_id}"
            logger.debug("Fireworks request: GET %s (job status)", url)
            resp = await client.get(url, headers=self._get_headers())
            resp.raise_for_status()
            job = resp.json()
            logger.debug("Fireworks job retrieved: batch_id=%s job=%s", batch_id, job)

        status = (job.get("state", "") or "").lower()
        if status.startswith("job_state_"):
            status = status[len("job_state_") :]
        status_map = {
            "validating": "in_progress",
            "pending": "in_progress",
            "running": "in_progress",
            "creating": "in_progress",
            "writing_results": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "expired": "failed",
            "cancelled": "failed",
        }
        return status_map.get(status, status), (job.get("jobProgress", None) or {}).get("percent", None)

    async def _download_dataset_firectl(self, dataset_id: str, output_dir: Path) -> list[Path]:
        """Download a dataset from Fireworks using firectl CLI. Returns list of downloaded file paths."""
        cmd = [
            self._firectl, "dataset", "download",
            "--account-id", self.account_id,
            "--output-dir", str(output_dir),
            "--quiet",
            dataset_id,
        ]
        logger.debug("Fireworks firectl download: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ValueError(
                f"firectl dataset download failed (exit {proc.returncode}): {stderr.decode().strip()}"
            )
        logger.debug("firectl download complete: dataset_id=%s", dataset_id)

        # firectl puts files at <output_dir>/dataset/<dataset_id>/<filename>
        dataset_dir = output_dir / "dataset" / dataset_id
        if not dataset_dir.exists():
            raise ValueError(f"Expected download directory not found: {dataset_dir}")
        return list(dataset_dir.iterdir())

    async def get_results(self, batch_id: str) -> BatchReponse:
        async def _do_get() -> BatchReponse:
            # Check job status via HTTP API
            async with httpx.AsyncClient(timeout=30.0) as client:
                job_url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs/{batch_id}"
                logger.debug("Fireworks request: GET %s (job status)", job_url)
                resp = await client.get(job_url, headers=self._get_headers())
                resp.raise_for_status()
                job = resp.json()
                logger.debug("Fireworks job retrieved: batch_id=%s job=%s", batch_id, job)

                status = (job.get("state", "") or "").lower()
                if status.startswith("job_state_"):
                    status = status[len("job_state_") :]
                if status not in ("completed", "failed", "cancelled", "expired"):
                    raise ValueError(f"Batch {batch_id} is not in a terminal state. Status: {status}")

            results = []
            errors = []
            output_dataset_id = self._output_dataset_ids.get(batch_id)
            if not output_dataset_id:
                output_dataset_ref = job.get("outputDatasetId", "")
                if output_dataset_ref:
                    output_dataset_id = output_dataset_ref.split("/")[-1]

            if output_dataset_id:
                # Download results via firectl
                _, tmp_dir = ensure_batch_storage_dirs()
                download_dir = tmp_dir / f"download-{batch_id}"
                download_dir.mkdir(parents=True, exist_ok=True)
                try:
                    files = await self._download_dataset_firectl(output_dataset_id, download_dir)
                    for filepath in files:
                        filename = filepath.name
                        with open(filepath) as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    data = json.loads(line)
                                    if "error" in filename.lower():
                                        errors.append(data)
                                    else:
                                        results.append(data)
                finally:
                    if CLEANUP and download_dir.exists():
                        shutil.rmtree(download_dir)

            return BatchReponse(batch_id=batch_id, status="completed", results=results, errors=errors if errors else None)

        return await _retry_on_network_error(_do_get, provider_name=self.provider)

    async def cancel(self, batch_id: str) -> bool:
        return False
