from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    import httpx
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch.fireworks_ai requires httpx for Fireworks batch jobs. '
        'Install it with `uv add "modaic[fireworks]"`.'
    ) from exc

from .._experimental import experimental
from ..enqueued_limits import fireworks_enqueued_limits
from ..storage import ensure_batch_storage_dirs
from ..token_counting import count_tokens_hf
from ..types import BatchRequestItem, RawResults, ResultItem
from .base import RemoteBatchClient, _extract_openai_compatible_message


def _check_firectl() -> str:
    path = shutil.which("firectl")
    if path is None:
        raise RuntimeError(
            "firectl CLI is required for Fireworks batch jobs but was not found on PATH.\n"
            "Install it from https://docs.fireworks.ai/tools-sdks/firectl/firectl"
        )
    return path


@experimental
class FireworksBatchClient(RemoteBatchClient):
    BASE_URL = "https://api.fireworks.ai/v1"

    name = "fireworks_ai"
    endpoint = None
    requires_consistent_model = True
    token_counter = staticmethod(count_tokens_hf)
    enqueued_limits_fn = staticmethod(fireworks_enqueued_limits)

    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        *,
        reqs_per_file: int = 1_000_000,
        max_file_size: int = 1024 * 1024 * 1024,
        tokens_per_file: Optional[int] = None,
        default_enqueued_reqs: Optional[int] = None,
        default_enqueued_tokens: Optional[int] = None,
        default_enqueued_jobs: Optional[int] = None,
        enable_concurrent_jobs: Optional[bool] = None,
    ):
        resolved = api_key or os.getenv("FIREWORKS_AI_API_KEY")
        if not resolved:
            raise ValueError("FIREWORKS_AI_API_KEY environment variable is not set")
        super().__init__(
            api_key=resolved,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            reqs_per_file=reqs_per_file,
            max_file_size=max_file_size,
            tokens_per_file=tokens_per_file,
            default_enqueued_reqs=default_enqueued_reqs,
            default_enqueued_tokens=default_enqueued_tokens,
            default_enqueued_jobs=default_enqueued_jobs,
            enable_concurrent_jobs=enable_concurrent_jobs,
        )

        self.account_id = account_id or os.getenv("FIREWORKS_ACCOUNT_ID")
        if not self.account_id:
            raise ValueError("FIREWORKS_ACCOUNT_ID environment variable is not set")

        self._firectl = _check_firectl()
        self._output_dataset_ids: dict[str, str] = {}
        self._shard_models: dict[str, str] = {}
        self._shard_lm_kwargs: dict[str, dict] = {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def format_line(self, item: BatchRequestItem) -> dict[str, Any]:
        return {
            "custom_id": item["id"],
            "body": {"messages": item["messages"], **item.get("lm_kwargs", {})},
        }

    def parse_result(self, raw: dict[str, Any]) -> ResultItem:
        response = raw.get("response", {})
        choices = response.get("choices", [])
        if not choices:
            error = raw.get("error") or response.get("error")
            raise ValueError(f"Batch request failed: {error}")
        choice = choices[0]
        message = choice.get("message", {})
        text, reasoning = _extract_openai_compatible_message(message)
        result: ResultItem = {"text": text}
        if reasoning is not None:
            result["reasoning_content"] = reasoning
        if choice.get("logprobs") is not None:
            result["logprobs"] = choice["logprobs"]
        if message.get("tool_calls") is not None:
            result["tool_calls"] = message["tool_calls"]
        return result

    async def _peek_shard_meta(self, shard: Path) -> tuple[str, dict[str, Any]]:
        """Fireworks job create needs model + inference params; peek first line of shard."""
        with open(shard, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if not first:
            raise ValueError(f"Shard {shard} is empty")
        line = json.loads(first)
        body = line.get("body", {})
        model = body.get("model")
        if model is None:
            raise ValueError("Fireworks batch requires a model in the request body")
        lm_kwargs = {k: v for k, v in body.items() if k not in ("model", "messages")}
        return model, lm_kwargs

    async def _upload_dataset(self, dataset_id: str, shard: Path) -> None:
        cmd = [
            self._firectl, "dataset", "create",
            "--account-id", self.account_id,
            "--quiet",
            dataset_id, str(shard),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ValueError(f"firectl dataset create failed (exit {proc.returncode}): {stderr.decode().strip()}")

    async def _download_dataset(self, dataset_id: str, output_dir: Path) -> list[Path]:
        cmd = [
            self._firectl, "dataset", "download",
            "--account-id", self.account_id,
            "--output-dir", str(output_dir),
            "--quiet",
            dataset_id,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ValueError(f"firectl dataset download failed (exit {proc.returncode}): {stderr.decode().strip()}")
        dataset_dir = output_dir / "dataset" / dataset_id
        if not dataset_dir.exists():
            raise ValueError(f"Expected download directory not found: {dataset_dir}")
        return list(dataset_dir.iterdir())

    async def create_batch(self, shard: Path) -> str:
        model, lm_kwargs = await self._peek_shard_meta(shard)

        dataset_id = f"batch-input-{uuid.uuid4().hex[:8]}"
        output_dataset_id = f"batch-output-{uuid.uuid4().hex[:8]}"
        job_id = f"batch-job-{uuid.uuid4().hex[:8]}"

        await self._upload_dataset(dataset_id, shard)

        async with httpx.AsyncClient(timeout=120.0) as client:
            status_url = f"{self.BASE_URL}/accounts/{self.account_id}/datasets/{dataset_id}"
            for _ in range(120):
                resp = await client.get(status_url, headers=self._headers())
                resp.raise_for_status()
                state = resp.json().get("state", "")
                if state == "READY":
                    break
                if state not in ("UPLOADING", "STATE_UNSPECIFIED"):
                    raise ValueError(f"Dataset in unexpected state: {state}")
                await asyncio.sleep(2)
            else:
                raise TimeoutError(f"Dataset {dataset_id} did not become READY within 240s")

            model_ref = model if model.startswith("accounts/") else f"accounts/fireworks/models/{model}"
            create_url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs"
            payload: dict[str, Any] = {
                "displayName": f"Batch job {job_id}",
                "model": model_ref,
                "inputDatasetId": f"accounts/{self.account_id}/datasets/{dataset_id}",
                "outputDatasetId": f"accounts/{self.account_id}/datasets/{output_dataset_id}",
            }
            inference: dict[str, Any] = {}
            if "max_tokens" in lm_kwargs:
                inference["maxTokens"] = lm_kwargs["max_tokens"]
            if "temperature" in lm_kwargs:
                inference["temperature"] = lm_kwargs["temperature"]
            if "top_p" in lm_kwargs:
                inference["topP"] = lm_kwargs["top_p"]
            if inference:
                payload["inferenceParameters"] = inference

            resp = await client.post(
                create_url, headers=self._headers(), json=payload, params={"batchInferenceJobId": job_id}
            )
            if resp.status_code != 200:
                raise ValueError(f"Failed to create batch job: {resp.status_code} - {resp.text}")
            job = resp.json()

        batch_id = (job.get("name", "") or "").split("/")[-1] or job_id
        self._output_dataset_ids[batch_id] = output_dataset_id
        return batch_id

    async def poll_status(self, batch_id: str) -> tuple[str, Optional[int]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs/{batch_id}"
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            job = resp.json()

        status = (job.get("state", "") or "").lower()
        if status.startswith("job_state_"):
            status = status[len("job_state_"):]
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
        pct = (job.get("jobProgress") or {}).get("percent")
        return status_map.get(status, status), pct

    async def fetch_results(self, batch_id: str) -> RawResults:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.BASE_URL}/accounts/{self.account_id}/batchInferenceJobs/{batch_id}"
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            job = resp.json()

        output_dataset_id = self._output_dataset_ids.get(batch_id)
        if not output_dataset_id:
            ref = job.get("outputDatasetId", "")
            output_dataset_id = ref.split("/")[-1] if ref else None

        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        if output_dataset_id:
            _, tmp_dir = ensure_batch_storage_dirs()
            download_dir = tmp_dir / f"download-{batch_id}"
            download_dir.mkdir(parents=True, exist_ok=True)
            try:
                files = await self._download_dataset(output_dataset_id, download_dir)
                for filepath in files:
                    bucket = errors if "error" in filepath.name.lower() else results
                    with open(filepath) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                bucket.append(json.loads(line))
            finally:
                if download_dir.exists():
                    shutil.rmtree(download_dir)

        return RawResults(results=results, errors=errors)

    async def cancel(self, batch_id: str) -> bool:
        return False
