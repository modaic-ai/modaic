from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

try:
    import anthropic
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch.anthropic requires the Anthropic SDK for Anthropic batch jobs. '
        'Install it with `uv add "modaic[anthropic]"`.'
    ) from exc

try:
    import httpx
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch.anthropic requires httpx for Anthropic batch jobs. '
        'Install it with `uv add "modaic[anthropic]"`.'
    ) from exc

from .._experimental import experimental
from ..enqueued_limits import anthropic_enqueued_limits
from ..token_counting import count_tokens_anthropic
from ..types import BatchRequestItem, RawResults, ResultItem
from .base import RemoteBatchClient, _extract_anthropic_message_content, logger


@experimental
class AnthropicBatchClient(RemoteBatchClient):
    name = "anthropic"
    endpoint = None
    token_counter = staticmethod(count_tokens_anthropic)
    enqueued_limits_fn = staticmethod(anthropic_enqueued_limits)

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        *,
        reqs_per_file: int = 100_000,
        max_file_size: int = 256 * 1024 * 1024,
        tokens_per_file: Optional[int] = None,
        default_enqueued_reqs: int = 100_000,
        default_enqueued_tokens: Optional[int] = None,
        default_enqueued_jobs: Optional[int] = None,
        enable_concurrent_jobs: Optional[bool] = None,
    ):
        resolved = api_key or os.getenv("ANTHROPIC_API_KEY")
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
        self._client = anthropic.AsyncAnthropic(api_key=resolved, max_retries=5, timeout=300.0)

    def format_line(self, item: BatchRequestItem) -> dict[str, Any]:
        params = {"model": item["model"], "messages": item["messages"], **item.get("lm_kwargs", {})}
        if "max_tokens" not in params:
            params["max_tokens"] = 4096
        return {"custom_id": item["id"], "params": params}

    def parse_result(self, raw: dict[str, Any]) -> ResultItem:
        result = raw.get("result", {})
        rtype = result.get("type", "")
        if rtype == "errored":
            raise ValueError(f"Batch request failed: {result.get('error', {})}")
        if rtype == "canceled":
            raise ValueError("Batch request was canceled")
        if rtype == "expired":
            raise ValueError("Batch request expired")
        if rtype != "succeeded":
            raise ValueError(f"Unknown result type: {rtype}")

        message = result.get("message", {})
        text, reasoning, tool_calls = _extract_anthropic_message_content(message.get("content", []))
        item: ResultItem = {"text": text}
        if reasoning is not None:
            item["reasoning_content"] = reasoning
        if tool_calls:
            item["tool_calls"] = tool_calls
        return item

    async def create_batch(self, shard: Path) -> str:
        requests: list[dict[str, Any]] = []
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    requests.append(json.loads(line))
        batch = await self._client.messages.batches.create(requests=requests)
        return batch.id

    async def poll_status(self, batch_id: str) -> tuple[str, Optional[int]]:
        batch = await self._client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = counts.canceled + counts.errored + counts.expired + counts.processing + counts.succeeded
        pct = int((1 - counts.processing / total) * 100) if total > 0 else 0
        status = batch.processing_status
        if status == "ended":
            return "completed", pct
        return "in_progress", pct

    async def fetch_results(self, batch_id: str) -> RawResults:
        batch = await self._client.messages.batches.retrieve(batch_id)
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        if batch.results_url:
            async with httpx.AsyncClient(timeout=120.0) as client:
                headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
                resp = await client.get(batch.results_url, headers=headers)
                resp.raise_for_status()
                for line in resp.text.strip().split("\n"):
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    rtype = data.get("result", {}).get("type", "")
                    if rtype in ("errored", "canceled", "expired"):
                        errors.append(data)
                    else:
                        results.append(data)

        return RawResults(results=results, errors=errors)

    async def cancel(self, batch_id: str) -> bool:
        try:
            await self._client.messages.batches.cancel(batch_id)
            return True
        except Exception:
            logger.error("Anthropic cancel failed: batch_id=%s", batch_id, exc_info=True)
            return False
