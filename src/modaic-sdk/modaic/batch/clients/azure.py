from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

try:
    from openai import AsyncAzureOpenAI
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch.azure requires the OpenAI SDK for Azure batch jobs. '
        'Install it with `uv add "modaic[azure]"`.'
    ) from exc

from .._experimental import experimental
from ..enqueued_limits import azure_enqueued_limits
from ..token_counting import count_tokens_tiktoken
from ..types import RawResults, ResultItem
from .base import RemoteBatchClient, _extract_openai_compatible_message, logger


@experimental
class AzureBatchClient(RemoteBatchClient):
    name = "azure"
    reqs_per_file = 100_000
    max_file_size = 200 * 1024 * 1024
    endpoint = "/v1/chat/completions"
    requires_consistent_model = True
    default_enqueued_jobs = 500
    token_counter = staticmethod(count_tokens_tiktoken)
    enqueued_limits_fn = staticmethod(azure_enqueued_limits)

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        *,
        reqs_per_file: Optional[int] = None,
        max_file_size: Optional[int] = None,
        tokens_per_file: Optional[int] = None,
        default_enqueued_reqs: Optional[int] = None,
        default_enqueued_tokens: Optional[int] = None,
        default_enqueued_jobs: Optional[int] = None,
        enable_concurrent_jobs: Optional[bool] = None,
    ):
        resolved_key = api_key or os.getenv("AZURE_API_KEY")
        resolved_endpoint = azure_endpoint or os.getenv("AZURE_API_BASE")
        resolved_version = api_version or os.getenv("AZURE_API_VERSION", "2024-07-01-preview")
        if not resolved_endpoint:
            raise ValueError("AZURE_API_BASE environment variable is not set")

        super().__init__(
            api_key=resolved_key,
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
        self._client = AsyncAzureOpenAI(
            api_key=resolved_key,
            azure_endpoint=resolved_endpoint,
            api_version=resolved_version,
            timeout=300.0,
        )
        self._file_ids: dict[str, str] = {}

    def parse_result(self, raw: dict[str, Any]) -> ResultItem:
        response = raw.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])
        if not choices:
            error = raw.get("error") or body.get("error", {})
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

    async def create_batch(self, shard: Path) -> str:
        with open(shard, "rb") as f:
            file_obj = await self._client.files.create(file=f, purpose="batch")
        batch = await self._client.batches.create(
            completion_window="24h",
            endpoint=self.endpoint,
            input_file_id=file_obj.id,
        )
        self._file_ids[batch.id] = file_obj.id
        logger.debug("Azure created batch_id=%s file_id=%s", batch.id, file_obj.id)
        return batch.id

    async def poll_status(self, batch_id: str) -> tuple[str, Optional[int]]:
        batch = await self._client.batches.retrieve(batch_id)
        counts = batch.request_counts
        if counts is None:
            return batch.status, None  # type: ignore[return-value]
        total = counts.total
        finished = counts.completed + counts.failed
        pct = int((finished / total) * 100) if total > 0 else 0
        return batch.status, pct  # type: ignore[return-value]

    async def fetch_results(self, batch_id: str) -> RawResults:
        batch = await self._client.batches.retrieve(batch_id)
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        if batch.output_file_id:
            resp = await self._client.files.content(batch.output_file_id)
            for line in resp.content.decode("utf-8").strip().split("\n"):
                if line:
                    results.append(json.loads(line))
        if batch.error_file_id:
            resp = await self._client.files.content(batch.error_file_id)
            for line in resp.content.decode("utf-8").strip().split("\n"):
                if line:
                    errors.append(json.loads(line))

        return RawResults(results=results, errors=errors)

    async def cancel(self, batch_id: str) -> bool:
        try:
            await self._client.batches.cancel(batch_id)
            return True
        except Exception:
            logger.error("Azure cancel failed: batch_id=%s", batch_id, exc_info=True)
            return False
