from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional, Tuple

try:
    from openai import AsyncAzureOpenAI
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch requires the OpenAI SDK for Azure batch jobs. Install it with `uv add "modaic[azure]"`.'
    ) from exc

from ..types import BatchReponse, BatchRequest, ResultItem
from .base import CLEANUP, BatchClient, _extract_openai_compatible_message, _retry_on_network_error, logger


class AzureBatchClient(BatchClient):
    provider: str = "azure"

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        resolved_api_key = api_key or os.getenv("AZURE_API_KEY")
        resolved_endpoint = azure_endpoint or os.getenv("AZURE_API_BASE")
        resolved_version = api_version or os.getenv("AZURE_API_VERSION", "2024-07-01-preview")

        if not resolved_endpoint:
            raise ValueError("AZURE_API_BASE environment variable is not set")

        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )
        self._client = AsyncAzureOpenAI(
            api_key=resolved_api_key,
            azure_endpoint=resolved_endpoint,
            api_version=resolved_version,
            timeout=300.0,
        )
        self._file_ids: dict[str, str] = {}

    def format(self, batch_request: BatchRequest) -> list[dict]:
        return [
            {
                "custom_id": request["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": request["model"], "messages": request["messages"], **request["lm_kwargs"]},
            }
            for request in batch_request["requests"]
        ]

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        response = raw_result.get("response", {})
        body = response.get("body", {})
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
        logger.debug("Azure submit: uploading file %s", jsonl_path)

        async def _do_submit() -> str:
            with open(jsonl_path, "rb") as f:
                file_obj = await self._client.files.create(file=f, purpose="batch")

            batch = await self._client.batches.create(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id=file_obj.id,
            )

            self._file_ids[batch.id] = file_obj.id
            logger.debug("Azure submit: created batch_id=%s", batch.id)
            return batch.id

        try:
            return await _retry_on_network_error(_do_submit, provider_name=self.provider, max_retries=7)
        finally:
            if CLEANUP and jsonl_path.exists():
                jsonl_path.unlink()

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        logger.debug("Azure status request: batch_id=%s", batch_id)
        batch = await self._client.batches.retrieve(batch_id)
        logger.debug("Azure status retrieved: batch_id=%s batch=%s", batch_id, batch)
        req_counts = batch.request_counts
        if req_counts is None:
            return batch.status, None  # type: ignore

        total = req_counts.total
        finished = req_counts.completed + req_counts.failed
        progress = int((finished / total) * 100) if total > 0 else 0
        return batch.status, progress  # type: ignore

    async def get_results(self, batch_id: str) -> BatchReponse:
        async def _do_get() -> BatchReponse:
            logger.debug("Azure results request: batch_id=%s", batch_id)
            batch = await self._client.batches.retrieve(batch_id)
            logger.debug("Azure results retrieved: batch_id=%s batch=%s", batch_id, batch)
            if batch.status not in ("completed", "failed", "cancelled", "expired"):
                raise ValueError(f"Batch {batch_id} is not in a terminal state. Status: {batch.status}")

            results = []
            errors = []

            if batch.output_file_id:
                logger.debug("Azure results: downloading output_file_id=%s", batch.output_file_id)
                file_response = await self._client.files.content(batch.output_file_id)
                content = file_response.content.decode("utf-8")
                for line in content.strip().split("\n"):
                    if line:
                        results.append(json.loads(line))

            if batch.error_file_id:
                logger.debug("Azure results: downloading error_file_id=%s", batch.error_file_id)
                error_response = await self._client.files.content(batch.error_file_id)
                error_content = error_response.content.decode("utf-8")
                for line in error_content.strip().split("\n"):
                    if line:
                        errors.append(json.loads(line))

            return BatchReponse(batch_id=batch_id, status=batch.status, results=results, errors=errors if errors else None)

        return await _retry_on_network_error(_do_get, provider_name=self.provider)

    async def cancel(self, batch_id: str) -> bool:
        try:
            logger.debug("Azure cancel request: batch_id=%s", batch_id)
            await self._client.batches.cancel(batch_id)
            return True
        except Exception:
            logger.error("Azure cancel failed: batch_id=%s", batch_id, exc_info=True)
            return False
