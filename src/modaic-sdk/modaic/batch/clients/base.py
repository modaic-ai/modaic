from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import dspy

from ..storage import ensure_batch_storage_dirs
from ..types import BatchReponse, BatchRequest, ResultItem

CLEANUP = True
logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_ERRORS = (
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.PoolTimeout,
        httpx.TimeoutException,
        httpx.NetworkError,
    )
except ImportError:
    HTTPX_ERRORS = ()


@dataclass
class _BatchSubmitState:
    requests_by_id: dict[str, dict[str, Any]]
    cached_results_by_id: dict[str, dict[str, Any]]
    uncached_request_count: int


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "".join(parts)
    return str(value)


def _extract_openai_compatible_message(message: dict[str, Any]) -> tuple[str, Optional[str]]:
    text = _stringify_content(message.get("content"))
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None:
        reasoning_content = _stringify_content(reasoning_content)
    else:
        reasoning_parts = []
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"reasoning", "reasoning_text", "thinking"} and isinstance(item.get("text"), str):
                    reasoning_parts.append(item["text"])
        if reasoning_parts:
            reasoning_content = "".join(reasoning_parts)
    return text, reasoning_content


def _extract_anthropic_message_content(content: Any) -> tuple[str, Optional[str], list[dict]]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []

    if not isinstance(content, list):
        return "", None, tool_calls

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block_type == "thinking" and isinstance(block.get("thinking"), str):
            reasoning_parts.append(block["thinking"])
        elif block_type == "redacted_thinking" and isinstance(block.get("data"), str):
            reasoning_parts.append(block["data"])
        elif block_type == "tool_use":
            tool_calls.append(block)

    reasoning_content = "".join(reasoning_parts) if reasoning_parts else None
    return "".join(text_parts), reasoning_content, tool_calls


async def _retry_on_network_error(
    coro_func: Callable[..., Any], *args: Any, max_retries: int = 5, provider_name: str = "provider", **kwargs: Any
) -> Any:
    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.debug("%s network attempt %d/%d", provider_name, attempt + 1, max_retries)
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_name = type(e).__name__
            error_msg = str(e).lower()

            is_network_error = (
                any(indicator in error_name for indicator in ["Connection", "Timeout", "ReadError", "WriteError"])
                or any(indicator in error_msg for indicator in ["connection", "timeout", "read error", "write error"])
                or "RemoteDisconnected" in error_name
                or (HTTPX_ERRORS and isinstance(e, HTTPX_ERRORS))
            )

            if not is_network_error:
                if "openai" in provider_name.lower():
                    try:
                        import openai

                        if isinstance(e, openai.APIConnectionError):
                            is_network_error = True
                    except ImportError:
                        pass
                elif "anthropic" in provider_name.lower():
                    try:
                        import anthropic

                        if isinstance(e, anthropic.APIConnectionError):
                            is_network_error = True
                    except ImportError:
                        pass

            if not is_network_error or attempt == max_retries - 1:
                logger.error(
                    "%s operation failed without retry (attempt %d/%d): %s",
                    provider_name,
                    attempt + 1,
                    max_retries,
                    e,
                    exc_info=True,
                )
                raise

            delay = (2 ** (attempt + 1)) + (random.random() * 2)
            logger.warning(
                "%s network error (attempt %d/%d): %s. Retrying in %.2fs",
                provider_name,
                attempt + 1,
                max_retries,
                e,
                delay,
            )
            await asyncio.sleep(delay)

    if last_exception:
        logger.error(
            "%s operation failed after %d attempts: %s",
            provider_name,
            max_retries,
            last_exception,
            exc_info=True,
        )
        raise last_exception


def _parse_time_string(time_str: str) -> float:
    import re

    match = re.match(r"^(\d+(?:\.\d+)?)\s*([smh])$", time_str.strip().lower())
    if not match:
        raise ValueError(f"Invalid time format: '{time_str}'. Expected format like '30s', '5m', or '24h'.")

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    raise ValueError(f"Unknown time unit: '{unit}'")


class BatchClient:
    provider: str
    status_callback: Optional[Callable[[str, str, Optional[int], dict], None]]

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    ):
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.max_poll_time_s = _parse_time_string(max_poll_time)
        self.status_callback = status_callback
        self.start_time: Optional[float] = None
        self.num_requests: Optional[int] = None
        self._consecutive_failures = 0
        self._submit_state_by_batch_id: dict[str, _BatchSubmitState] = {}

    def _build_cache_request(self, request: dict[str, Any]) -> dict[str, Any]:
        cache_request = {
            "model": request["model"],
            "messages": request["messages"],
        }
        cache_request.update(request.get("lm_kwargs", {}))
        return cache_request

    def _normalize_cached_result(self, request_id: str, cached_value: Any) -> Optional[dict[str, Any]]:
        if not isinstance(cached_value, dict):
            return None
        if "response" not in cached_value and "error" not in cached_value:
            return None
        normalized = dict(cached_value)
        normalized["custom_id"] = request_id
        return normalized

    def format(self, batch_request: BatchRequest) -> list[dict]:
        raise NotImplementedError("format is not implemented")

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        raise NotImplementedError("parse is not implemented")

    def create_jsonl(self, batch_request: BatchRequest, path: Optional[Path] = None) -> Path:
        if path is None:
            _, tmp_dir = ensure_batch_storage_dirs()
            path = tmp_dir / f"batch_{id(batch_request)}.jsonl"
        logger.debug("Creating JSONL batch file: path=%s requests=%d", path, len(batch_request["requests"]))
        formatted = self.format(batch_request)
        with open(path, "w") as f:
            for item in formatted:
                f.write(json.dumps(item) + "\n")
        return path

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        raise NotImplementedError("_submit_batch_request is not implemented")

    async def submit(self, batch_request: BatchRequest) -> str:
        requests_by_id: dict[str, dict[str, Any]] = {request["id"]: request for request in batch_request["requests"]}
        cached_results_by_id: dict[str, dict[str, Any]] = {}
        uncached_requests: list[dict[str, Any]] = []

        for request in batch_request["requests"]:
            cache_request = self._build_cache_request(request)
            cached_value = dspy.cache.get(cache_request)
            cached_result = self._normalize_cached_result(request["id"], cached_value)
            if cached_result is not None:
                cached_results_by_id[request["id"]] = cached_result
            else:
                uncached_requests.append(request)

        if self.status_callback is not None:
            self.status_callback(None, "submitting", 0, {"num_cached": len(cached_results_by_id)})

        logger.debug(
            "Batch submit cache check: provider=%s total=%d cached=%d uncached=%d",
            self.provider,
            len(batch_request["requests"]),
            len(cached_results_by_id),
            len(uncached_requests),
        )

        if not uncached_requests:
            batch_id = f"cached-{self.provider}-{uuid.uuid4()}"
            self._submit_state_by_batch_id[batch_id] = _BatchSubmitState(
                requests_by_id=requests_by_id,
                cached_results_by_id=cached_results_by_id,
                uncached_request_count=0,
            )
            return batch_id

        uncached_batch_request: BatchRequest = {
            "requests": uncached_requests,
            "model": batch_request.get("model"),
            "lm_kwargs": batch_request.get("lm_kwargs"),
        }
        batch_id = await self._submit_batch_request(uncached_batch_request)
        self._submit_state_by_batch_id[batch_id] = _BatchSubmitState(
            requests_by_id=requests_by_id,
            cached_results_by_id=cached_results_by_id,
            uncached_request_count=len(uncached_requests),
        )
        return batch_id

    async def get_status(self, batch_id: str) -> Tuple[str, Optional[int]]:
        submit_state = self._submit_state_by_batch_id.get(batch_id)
        if submit_state and submit_state.uncached_request_count == 0:
            status = "completed"
            progress = 100
            if self.status_callback is not None:
                import time

                elapsed = time.time() - self.start_time if self.start_time else 0.0
                metadata = {
                    "provider": self.provider,
                    "num_requests": self.num_requests,
                    "elapsed_time": elapsed,
                }
                self.status_callback(batch_id, status, progress, metadata)
            logger.debug(
                "Batch status from cache-only state: provider=%s batch_id=%s status=%s progress=%s",
                self.provider,
                batch_id,
                status,
                progress,
            )
            return status, progress

        try:
            status, progress = await self._get_status_impl(batch_id)
            self._consecutive_failures = 0
        except Exception as e:
            logger.warning(
                "Batch status check failed: provider=%s batch_id=%s error=%s",
                self.provider,
                batch_id,
                e,
                exc_info=True,
            )
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                import warnings

                warnings.warn(
                    f"Batch status check failed {self._consecutive_failures} times in a row for batch {batch_id} ({self.provider}): {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            status = "in_progress"
            progress = None

        if self.status_callback is not None:
            import time

            elapsed = time.time() - self.start_time if self.start_time else 0.0
            metadata = {
                "provider": self.provider,
                "num_requests": self.num_requests,
                "elapsed_time": elapsed,
            }
            self.status_callback(batch_id, status, progress, metadata)
        logger.debug(
            "Batch status retrieved: provider=%s batch_id=%s status=%s progress=%s",
            self.provider,
            batch_id,
            status,
            progress,
        )
        return status, progress

    async def _get_status_impl(self, batch_id: str) -> Tuple[str, Optional[int]]:
        raise NotImplementedError("get_status is not implemented")

    async def get_results(self, batch_id: str) -> BatchReponse:
        raise NotImplementedError("get_results is not implemented")

    async def cancel(self, batch_id: str) -> bool:
        raise NotImplementedError("cancel is not implemented")

    async def submit_and_wait(
        self,
        batch_request: BatchRequest,
        show_progress: bool = True,
    ) -> BatchReponse:
        import time

        self.num_requests = len(batch_request["requests"])
        self.start_time = time.time()

        logger.debug(
            "submit_and_wait start: provider=%s requests=%d show_progress=%s",
            self.provider,
            self.num_requests,
            show_progress,
        )
        batch_id = await self.submit(batch_request)
        logger.debug("submit_and_wait submitted: provider=%s batch_id=%s", self.provider, batch_id)
        submit_state = self._submit_state_by_batch_id.get(batch_id)

        if submit_state and submit_state.uncached_request_count == 0:
            ordered_cached_results = [
                submit_state.cached_results_by_id[request["id"]]
                for request in batch_request["requests"]
                if request["id"] in submit_state.cached_results_by_id
            ]
            self._submit_state_by_batch_id.pop(batch_id, None)
            return BatchReponse(
                batch_id=batch_id,
                status="completed",
                results=ordered_cached_results,
                errors=None,
                raw_response={"source": "cache"},
            )

        waited = 0.0

        while waited < self.max_poll_time_s:
            status, progress = await self.get_status(batch_id)
            logger.debug(
                "submit_and_wait poll: provider=%s batch_id=%s status=%s progress=%s waited=%.1fs",
                self.provider,
                batch_id,
                status,
                progress,
                waited,
            )

            if status.lower() == "completed":
                logger.debug("submit_and_wait completed: provider=%s batch_id=%s", self.provider, batch_id)
                api_results = await self.get_results(batch_id)
                submit_state = self._submit_state_by_batch_id.pop(batch_id, None)

                if not submit_state:
                    return api_results

                for raw_result in list(api_results.results) + list(api_results.errors or []):
                    request_id = raw_result.get("custom_id")
                    if not request_id or request_id not in submit_state.requests_by_id:
                        continue
                    request_item = submit_state.requests_by_id[request_id]
                    cache_request = self._build_cache_request(request_item)
                    dspy.cache.put(cache_request, raw_result)

                uncached_results_by_id = {
                    raw_result["custom_id"]: raw_result
                    for raw_result in list(api_results.results) + list(api_results.errors or [])
                    if isinstance(raw_result, dict) and raw_result.get("custom_id")
                }

                merged_results: list[dict[str, Any]] = []
                merged_errors: list[dict[str, Any]] = []
                for request in batch_request["requests"]:
                    request_id = request["id"]
                    raw_result = submit_state.cached_results_by_id.get(request_id) or uncached_results_by_id.get(
                        request_id
                    )
                    if raw_result is None:
                        continue
                    if "error" in raw_result and "response" not in raw_result:
                        merged_errors.append(raw_result)
                    else:
                        merged_results.append(raw_result)

                return BatchReponse(
                    batch_id=api_results.batch_id,
                    status=api_results.status,
                    results=merged_results,
                    errors=merged_errors if merged_errors else None,
                    raw_response=api_results.raw_response,
                )
            if status.lower() in ("failed", "cancelled", "expired"):
                try:
                    failure_results = await self.get_results(batch_id)
                    error_msg = f"Batch job {batch_id} failed with status: {status}"
                    all_errors = failure_results.errors or []

                    if all_errors:
                        display_errors = []
                        for e in all_errors[:5]:
                            if "batch_error" in e:
                                display_errors.append(e["batch_error"])
                            elif "error" in e:
                                display_errors.append(e["error"])
                            else:
                                display_errors.append(e)

                        error_details = json.dumps(display_errors, indent=2)
                        error_msg += f"\nErrors found:\n{error_details}"
                    else:
                        error_msg += (
                            f"\nNo specific error details found in batch. Check {self.provider} dashboard for more info."
                            f"\nresponse: {failure_results.raw_response}"
                        )

                    logger.error(
                        "submit_and_wait failure: provider=%s batch_id=%s failure_results=%s",
                        self.provider,
                        batch_id,
                        failure_results.raw_response,
                    )
                    raise RuntimeError(error_msg)
                except Exception as e:
                    if isinstance(e, RuntimeError) and "Batch job" in str(e):
                        raise
                    logger.error(
                        "submit_and_wait failure details fetch failed: provider=%s batch_id=%s error=%s",
                        self.provider,
                        batch_id,
                        e,
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Batch job {batch_id} failed with status: {status}. Also failed to fetch error details: {e}"
                    ) from None
                finally:
                    self._submit_state_by_batch_id.pop(batch_id, None)

            logger.debug("submit_and_wait sleeping: provider=%s seconds=%.1f", self.provider, self.poll_interval)
            await asyncio.sleep(self.poll_interval)
            waited += self.poll_interval

        logger.error(
            "submit_and_wait timeout: provider=%s batch_id=%s max_poll_time=%s",
            self.provider,
            batch_id,
            self.max_poll_time,
        )
        self._submit_state_by_batch_id.pop(batch_id, None)
        raise TimeoutError(f"Batch job {batch_id} did not complete within {self.max_poll_time}")
