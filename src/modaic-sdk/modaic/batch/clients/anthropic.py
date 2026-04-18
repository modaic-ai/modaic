from __future__ import annotations

import json
from typing import Any, Callable, Optional, Tuple

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

from ..types import BatchReponse, BatchRequest, ResultItem
from .base import BatchClient, _extract_anthropic_message_content, _retry_on_network_error, logger


class AnthropicBatchClient(BatchClient):
    provider: str = "anthropic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        import os

        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(
            api_key=resolved_api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )
        self._client = anthropic.AsyncAnthropic(api_key=resolved_api_key, max_retries=5, timeout=300.0)

    def format(self, batch_request: BatchRequest) -> list[dict]:
        formatted = []
        for request in batch_request["requests"]:
            params = {
                "model": request["model"],
                "messages": request["messages"],
                **request["lm_kwargs"],
            }
            if "max_tokens" not in params:
                params["max_tokens"] = 4096

            formatted.append({"custom_id": request["id"], "params": params})
        return formatted

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        result = raw_result.get("result", {})
        result_type = result.get("type", "")

        if result_type == "errored":
            error = result.get("error", {})
            raise ValueError(f"Batch request failed: {error}")
        if result_type == "canceled":
            raise ValueError("Batch request was canceled")
        if result_type == "expired":
            raise ValueError("Batch request expired")
        if result_type != "succeeded":
            raise ValueError(f"Unknown result type: {result_type}")

        message = result.get("message", {})
        text, reasoning_content, tool_calls = _extract_anthropic_message_content(message.get("content", []))

        result_item: ResultItem = {"text": text}
        if reasoning_content is not None:
            result_item["reasoning_content"] = reasoning_content
        if tool_calls:
            result_item["tool_calls"] = tool_calls

        return result_item

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        formatted_requests = self.format(batch_request)

        async def _do_submit() -> str:
            logger.debug("Anthropic submit: requests=%d", len(formatted_requests))
            batch = await self._client.messages.batches.create(requests=formatted_requests)
            return batch.id

        return await _retry_on_network_error(_do_submit, provider_name=self.provider)

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        logger.debug("Anthropic status request: batch_id=%s", batch_id)
        batch = await self._client.messages.batches.retrieve(batch_id)
        logger.debug("Anthropic status retrieved: batch_id=%s batch=%s", batch_id, batch)

        req_counts = batch.request_counts
        total = (
            req_counts.canceled + req_counts.errored + req_counts.expired + req_counts.processing + req_counts.succeeded
        )
        progress = int((1 - req_counts.processing / total) * 100 if total > 0 else 0)

        status = batch.processing_status
        if status == "ended":
            return "completed", progress
        if status == "canceling":
            return "in_progress", progress
        return "in_progress", progress

    async def get_results(self, batch_id: str) -> BatchReponse:
        async def _do_get() -> BatchReponse:
            logger.debug("Anthropic results request: batch_id=%s", batch_id)
            batch = await self._client.messages.batches.retrieve(batch_id)
            logger.debug("Anthropic results retrieved: batch_id=%s batch=%s", batch_id, batch)

            if batch.processing_status not in ("ended", "canceling"):
                raise ValueError(f"Batch {batch_id} is not in a terminal state. Status: {batch.processing_status}")

            results = []
            errors = []

            if batch.results_url:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    headers = {
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                    }
                    logger.debug("Anthropic request: GET results_url=%s", batch.results_url)
                    resp = await client.get(batch.results_url, headers=headers)
                    resp.raise_for_status()
                    for line in resp.text.strip().split("\n"):
                        if line.strip():
                            data = json.loads(line)
                            result_type = data.get("result", {}).get("type", "")
                            if result_type in ("errored", "canceled", "expired"):
                                errors.append(data)
                            else:
                                results.append(data)

            return BatchReponse(batch_id=batch_id, status="completed", results=results, errors=errors if errors else None)

        return await _retry_on_network_error(_do_get, provider_name=self.provider)

    async def cancel(self, batch_id: str) -> bool:
        try:
            logger.debug("Anthropic cancel request: batch_id=%s", batch_id)
            await self._client.messages.batches.cancel(batch_id)
            return True
        except Exception:
            logger.error("Anthropic cancel failed: batch_id=%s", batch_id, exc_info=True)
            return False
