from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

try:
    from together import AsyncTogether
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch.together_ai requires the Together SDK for Together batch jobs. '
        'Install it with `uv add "modaic[together]"`.'
    ) from exc

from ..enqueued_limits import together_enqueued_limits
from ..token_counting import count_tokens_hf
from ..types import BatchRequestItem, RawResults, ResultItem
from .base import RemoteBatchClient, _extract_openai_compatible_message, logger


class TogetherBatchClient(RemoteBatchClient):
    name = "together_ai"
    reqs_per_file = 50_000
    max_file_size = 100 * 1024 * 1024
    endpoint = "/v1/chat/completions"
    token_counter = staticmethod(count_tokens_hf)
    enqueued_limits_fn = staticmethod(together_enqueued_limits)

    def __init__(
        self,
        api_key: Optional[str] = None,
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
        resolved = api_key or os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
        if not resolved:
            raise ValueError("TOGETHERAI_API_KEY or TOGETHER_API_KEY environment variable is not set")
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
        self._client = AsyncTogether(api_key=resolved, timeout=300.0)

    def format_line(self, item: BatchRequestItem) -> dict[str, Any]:
        return {
            "custom_id": item["id"],
            "body": {"model": item["model"], "messages": item["messages"], **item.get("lm_kwargs", {})},
        }

    def parse_result(self, raw: dict[str, Any]) -> ResultItem:
        response = raw.get("response", {})
        body = response.get("body", response)
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
        file_obj = await self._client.files.upload(file=str(shard), purpose="batch-api", check=False)
        batch = await self._client.batches.create(input_file_id=file_obj.id, endpoint=self.endpoint)
        if batch.job and batch.job.id:
            return batch.job.id
        raise RuntimeError("Failed to get batch ID from Together API")

    async def poll_status(self, batch_id: str) -> tuple[str, Optional[int]]:
        batch = await self._client.batches.retrieve(batch_id)
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
        return status_map.get(status, status), batch.progress  # type: ignore[return-value]

    async def fetch_results(self, batch_id: str) -> RawResults:
        batch = await self._client.batches.retrieve(batch_id)
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        batch_errors = getattr(batch, "errors", None)
        if batch_errors:
            items = batch_errors.data if hasattr(batch_errors, "data") else batch_errors
            if isinstance(items, list):
                for err in items:
                    detail = {a: getattr(err, a) for a in ("code", "message", "param") if hasattr(err, a)}
                    errors.append({"batch_error": detail or str(err)})

        if batch.output_file_id:
            results.extend(await self._stream_jsonl(batch.output_file_id))
        if batch.error_file_id:
            errors.extend(await self._stream_jsonl(batch.error_file_id))

        return RawResults(results=results, errors=errors)

    async def _stream_jsonl(self, file_id: str) -> list[dict[str, Any]]:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
            async with self._client.files.with_streaming_response.content(id=file_id) as response:
                async for chunk in response.iter_bytes():
                    tmp.write(chunk)
        try:
            out: list[dict[str, Any]] = []
            with open(tmp_path) as f:
                for line in f:
                    if line.strip():
                        out.append(json.loads(line))
            return out
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def cancel(self, batch_id: str) -> bool:
        try:
            await self._client.batches.cancel(batch_id)
            return True
        except Exception:
            logger.error("Together cancel failed: batch_id=%s", batch_id, exc_info=True)
            return False
