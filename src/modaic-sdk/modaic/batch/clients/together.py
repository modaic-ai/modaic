from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Callable, Optional, Tuple

try:
    from together import AsyncTogether
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch requires the Together SDK for Together batch jobs. Install it with `uv add "modaic[together]"`.'
    ) from exc

from ..types import BatchReponse, BatchRequest, ResultItem
from .base import CLEANUP, BatchClient, _extract_openai_compatible_message, _retry_on_network_error, logger


class TogetherBatchClient(BatchClient):
    provider: str = "together_ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        resolved_api_key = api_key or os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
        if not resolved_api_key:
            raise ValueError("TOGETHERAI_API_KEY or TOGETHER_API_KEY environment variable is not set")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )
        self._client = AsyncTogether(api_key=resolved_api_key, timeout=300.0)

    def format(self, batch_request: BatchRequest) -> list[dict]:
        return [
            {
                "custom_id": request["id"],
                "body": {"model": request["model"], "messages": request["messages"], **request["lm_kwargs"]},
            }
            for request in batch_request["requests"]
        ]

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        response = raw_result.get("response", {})
        body = response.get("body", response)
        choices = body.get("choices", [])

        if not choices:
            error = raw_result.get("error") or body.get("error", {})
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

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        jsonl_path = self.create_jsonl(batch_request)
        logger.debug("Together submit: uploading file %s", jsonl_path)

        async def _do_submit() -> str:
            file_obj = await self._client.files.upload(file=str(jsonl_path), purpose="batch-api", check=False)
            logger.debug("Together submit: creating batch from file_id=%s", file_obj.id)
            batch = await self._client.batches.create(input_file_id=file_obj.id, endpoint="/v1/chat/completions")

            if batch.job and batch.job.id:
                logger.debug("Together submit: created batch_id=%s", batch.job.id)
                return batch.job.id
            raise RuntimeError("Failed to get batch ID from Together API")

        try:
            return await _retry_on_network_error(_do_submit, provider_name=self.provider, max_retries=7)
        finally:
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        logger.debug("Together status request: batch_id=%s", batch_id)
        batch = await self._client.batches.retrieve(batch_id)
        logger.debug("Together status retrieved: batch_id=%s batch=%s", batch_id, batch)

        status = (batch.status or "").lower()
        status_map = {
            "validating": "in_progress",
            "queued": "in_progress",
            "running": "in_progress",
            "processing": "in_progress",
            "in_progress": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "error": "failed",
            "errored": "failed",
            "cancelled": "cancelled",
            "canceled": "cancelled",
            "cancelling": "cancelled",
            "canceling": "cancelled",
            "expired": "expired",
        }
        normalized_status = status_map.get(status, status)
        return normalized_status, batch.progress  # type: ignore

    async def get_results(self, batch_id: str) -> BatchReponse:
        async def _do_get() -> BatchReponse:
            logger.debug("Together results request: batch_id=%s", batch_id)
            batch = await self._client.batches.retrieve(batch_id)
            logger.debug("Together results retrieved: batch_id=%s batch=%s", batch_id, batch)

            status = (batch.status or "").lower()
            if status not in ("completed", "failed", "cancelled", "expired"):
                raise ValueError(f"Batch {batch_id} is not in a terminal state. Status: {status}")

            results = []
            errors = []
            batch_errors = getattr(batch, "errors", None)
            if batch_errors:
                items = batch_errors.data if hasattr(batch_errors, "data") else batch_errors
                if isinstance(items, list):
                    for err in items:
                        err_detail = {}
                        for attr in ["code", "message", "param"]:
                            if hasattr(err, attr):
                                err_detail[attr] = getattr(err, attr)

                        if err_detail:
                            errors.append({"batch_error": err_detail})
                        else:
                            errors.append({"batch_error": str(err)})

            if batch.output_file_id:
                logger.debug("Together results: downloading output_file_id=%s", batch.output_file_id)
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as tmp:
                    tmp_path = tmp.name
                    async with self._client.files.with_streaming_response.content(id=batch.output_file_id) as response:
                        async for chunk in response.iter_bytes():
                            tmp.write(chunk)

                try:
                    with open(tmp_path) as f:
                        for line in f:
                            if line.strip():
                                results.append(json.loads(line))
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            if batch.error_file_id:
                logger.debug("Together results: downloading error_file_id=%s", batch.error_file_id)
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as tmp:
                    tmp_path = tmp.name
                    async with self._client.files.with_streaming_response.content(id=batch.error_file_id) as response:
                        async for chunk in response.iter_bytes():
                            tmp.write(chunk)

                try:
                    with open(tmp_path) as f:
                        for line in f:
                            if line.strip():
                                errors.append(json.loads(line))
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            return BatchReponse(batch_id=batch_id, status="completed", results=results, errors=errors if errors else None)

        return await _retry_on_network_error(_do_get, provider_name=self.provider)

    async def cancel(self, batch_id: str) -> bool:
        try:
            logger.debug("Together cancel request: batch_id=%s", batch_id)
            await self._client.batches.cancel(batch_id)
            return True
        except Exception:
            logger.error("Together cancel failed: batch_id=%s", batch_id, exc_info=True)
            return False
