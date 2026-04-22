from __future__ import annotations

import asyncio
import logging
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal, Optional

from ..enqueued_limits import EnqueuedLimitsFn
from ..token_counting import TokenCounter
from ..types import BatchRequestItem, RawResults, ResultItem, ShardOutcome

Concurrency = Literal["parallel", "sequential"]

CLEANUP = True
logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_ERRORS: tuple = (
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.PoolTimeout,
        httpx.TimeoutException,
        httpx.NetworkError,
    )
except ImportError:
    HTTPX_ERRORS = ()


class ShardReporter:
    """Per-shard progress sink passed to BatchClient.execute_shard."""

    def started(self, batch_id: str) -> None:
        pass

    def percent(self, pct: int) -> None:
        pass


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
    if reasoning_content is None:
        reasoning_content = message.get("reasoning")
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
    last_exception: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not _looks_like_network_error(e) or attempt == max_retries - 1:
                raise
            delay = (2 ** (attempt + 1)) + random.random() * 2
            logger.warning(
                "%s network error (attempt %d/%d): %s. Retrying in %.2fs",
                provider_name, attempt + 1, max_retries, e, delay,
            )
            await asyncio.sleep(delay)
    if last_exception is not None:
        raise last_exception


def _looks_like_network_error(e: BaseException) -> bool:
    if HTTPX_ERRORS and isinstance(e, HTTPX_ERRORS):
        return True
    try:
        import openai
        if isinstance(e, openai.APIConnectionError):
            return True
    except ImportError:
        pass
    try:
        import anthropic
        if isinstance(e, anthropic.APIConnectionError):
            return True
    except ImportError:
        pass
    return False


def parse_time_string(time_str: str) -> float:
    import re
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([smh])$", time_str.strip().lower())
    if not match:
        raise ValueError(f"Invalid time format: '{time_str}'. Expected '30s', '5m', or '24h'.")
    value = float(match.group(1))
    unit = match.group(2)
    return value * {"s": 1, "m": 60, "h": 3600}[unit]


class BatchShardFailed(RuntimeError):
    def __init__(self, batch_id: str, status: str, errors: list[dict]):
        self.batch_id = batch_id
        self.status = status
        self.errors = errors
        super().__init__(f"Shard {batch_id} ended with status={status}")


class BatchClient:
    """Minimal client surface. Runner drives lifecycle via session() + execute_shard().

    Subclasses shadow the class-level config defaults (``reqs_per_file`` etc.)
    to declare provider-specific limits. Constructor kwargs that override those
    defaults should set instance attributes with the same name so lookup
    naturally resolves to the override.
    """

    # --- Config (subclasses shadow with provider defaults) ---
    name: str = "unknown"
    reqs_per_file: int = 50_000
    max_file_size: int = sys.maxsize
    tokens_per_file: Optional[int] = None
    default_enqueued_reqs: Optional[int] = None
    default_enqueued_tokens: Optional[int] = None
    default_enqueued_jobs: Optional[int] = None
    enable_concurrent_jobs: bool = True
    concurrency: Concurrency = "parallel"
    safety_margin: float = 0.95
    endpoint: Optional[str] = None
    requires_consistent_model: bool = False
    resumable: bool = True
    token_counter: Optional[TokenCounter] = None
    enqueued_limits_fn: Optional[EnqueuedLimitsFn] = None

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
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.max_poll_time_s = parse_time_string(max_poll_time)
        if reqs_per_file is not None:
            self.reqs_per_file = reqs_per_file
        if max_file_size is not None:
            self.max_file_size = max_file_size
        if tokens_per_file is not None:
            self.tokens_per_file = tokens_per_file
        if default_enqueued_reqs is not None:
            self.default_enqueued_reqs = default_enqueued_reqs
        if default_enqueued_tokens is not None:
            self.default_enqueued_tokens = default_enqueued_tokens
        if default_enqueued_jobs is not None:
            self.default_enqueued_jobs = default_enqueued_jobs
        if enable_concurrent_jobs is not None:
            self.enable_concurrent_jobs = enable_concurrent_jobs

    # --- Derived caps ---

    @property
    def request_cap(self) -> int:
        return max(1, int(self.reqs_per_file * self.safety_margin))

    @property
    def byte_cap(self) -> int:
        if self.max_file_size >= sys.maxsize // 2:
            return self.max_file_size
        return int(self.max_file_size * self.safety_margin)

    @property
    def token_cap(self) -> Optional[int]:
        if self.tokens_per_file is None:
            return None
        return int(self.tokens_per_file * self.safety_margin)

    # --- Per-shard behavior ---

    def format_line(self, item: BatchRequestItem) -> dict[str, Any]:
        body = {"model": item["model"], "messages": item["messages"], **item.get("lm_kwargs", {})}
        line = {"custom_id": item["id"], "body": body}
        if self.endpoint is not None:
            line["method"] = "POST"
            line["url"] = self.endpoint
        return line

    def parse_result(self, raw: dict[str, Any]) -> ResultItem:
        raise NotImplementedError

    def to_litellm_response(self, raw: dict[str, Any]) -> Any:
        """Convert a successful provider output dict to a litellm ``ModelResponse``.

        Used when writing batch results into ``dspy.cache`` so a future non-batch
        ``dspy.LM`` call with the same prompt can hit the cached completion.
        Raises whatever ``parse_result`` raises if ``raw`` represents a failure.
        """
        from litellm import Choices, Message, ModelResponse

        item = self.parse_result(raw)
        message = Message(role="assistant", content=item.get("text") or "")
        if item.get("reasoning_content") is not None:
            message.reasoning_content = item["reasoning_content"]
        if item.get("tool_calls") is not None:
            message.tool_calls = item["tool_calls"]
        choice = Choices(index=0, message=message, finish_reason="stop")
        if item.get("logprobs") is not None:
            choice.logprobs = item["logprobs"]
        return ModelResponse(choices=[choice], model=raw.get("model", ""))

    @asynccontextmanager
    async def session(self) -> AsyncIterator[None]:
        yield

    async def execute_shard(self, shard: Path, reporter: ShardReporter) -> ShardOutcome:
        raise NotImplementedError


class RemoteBatchClient(BatchClient):
    """Upload/poll/fetch client. Subclasses implement the three RPCs."""

    async def create_batch(self, shard: Path) -> str:
        raise NotImplementedError

    async def poll_status(self, batch_id: str) -> tuple[str, Optional[int]]:
        raise NotImplementedError

    async def fetch_results(self, batch_id: str) -> RawResults:
        raise NotImplementedError

    async def cancel(self, batch_id: str) -> bool:
        raise NotImplementedError

    async def execute_shard(self, shard: Path, reporter: ShardReporter) -> ShardOutcome:
        batch_id = await _retry_on_network_error(
            self.create_batch, shard, provider_name=self.name, max_retries=7
        )
        reporter.started(batch_id)

        waited = 0.0
        consecutive_failures = 0
        while waited < self.max_poll_time_s:
            try:
                status, pct = await self.poll_status(batch_id)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                logger.warning("status poll failed (%d): %s", consecutive_failures, e)
                if consecutive_failures >= 5:
                    raise
                await asyncio.sleep(self.poll_interval)
                waited += self.poll_interval
                continue

            if pct is not None:
                reporter.percent(pct)

            lower = status.lower()
            if lower == "completed":
                raw = await _retry_on_network_error(
                    self.fetch_results, batch_id, provider_name=self.name
                )
                return ShardOutcome(
                    batch_id=batch_id, results=raw.results, errors=raw.errors, raw_response=raw.raw_response
                )
            if lower in ("failed", "cancelled", "canceled", "expired"):
                try:
                    raw = await self.fetch_results(batch_id)
                    errors = raw.errors or []
                except Exception:
                    errors = []
                raise BatchShardFailed(batch_id, lower, errors)

            await asyncio.sleep(self.poll_interval)
            waited += self.poll_interval

        raise TimeoutError(f"Batch {batch_id} did not complete within {self.max_poll_time}")
