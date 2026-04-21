from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from .clients.base import BatchClient, BatchShardFailed, ShardReporter
from .enqueued_limits import EnqueuedLimits
from .progress import BatchProgressDisplay, ShardEvent
from .storage import ensure_batch_storage_dirs
from .types import BatchRequestItem, BatchResponse, ShardOutcome
from .writer import JSONLShardWriter

logger = logging.getLogger(__name__)

Mode = Literal["parallel", "sequential"]

CACHE_HIT_MARKER = "__modaic_cached_result_item__"

# dspy.LM.forward adds these to its cache request; we must mirror them or
# keys won't match and non-batch calls won't hit batch-written entries.
_DSPY_SYNC_FN_IDENTIFIER = "dspy.clients.lm.litellm_completion"
_DSPY_ASYNC_FN_IDENTIFIER = "dspy.clients.lm.alitellm_completion"
_DSPY_IGNORED_CACHE_KEYS = ["api_key", "api_base", "base_url"]


class _ReporterToDisplay(ShardReporter):
    def __init__(self, display: BatchProgressDisplay, index: int, total: int):
        self.display = display
        self.index = index
        self.total = total
        self.batch_id: Optional[str] = None

    def started(self, batch_id: str) -> None:
        self.batch_id = batch_id
        self.display(
            ShardEvent(
                kind="shard_started",
                shard_index=self.index,
                total_shards=self.total,
                batch_id=batch_id,
            )
        )

    def percent(self, pct: int) -> None:
        self.display(
            ShardEvent(
                kind="shard_progress",
                shard_index=self.index,
                total_shards=self.total,
                batch_id=self.batch_id,
                percent=pct,
            )
        )


class _NullReporter(ShardReporter):
    pass


@dataclass
class _ShardRun:
    """Per-shard execution bookkeeping used by ``_execute_waves`` and ``_merge``."""

    items: list[BatchRequestItem]
    outcome: Optional[ShardOutcome] = None
    error: Optional[str] = None
    batch_id: Optional[str] = None


class BatchJobRunner:
    """Owns cache partitioning, shard writing, concurrency policy, and result merging.

    Cache defaults to ``dspy.cache`` when ``cache`` is ``None``. Pass ``cache=False``
    to disable caching entirely. Successful shard results are written to the cache
    as each shard completes, so partial progress survives a failed shard: a rerun
    with the same inputs will resubmit only the items whose shard failed.
    """

    def __init__(
        self,
        client: BatchClient,
        cache: Any = None,
        mode: Mode = "parallel",
        max_concurrent: Optional[int] = None,
        display: Optional[BatchProgressDisplay] = None,
    ):
        self.client = client
        self.cache = _resolve_cache(cache)
        self.max_concurrent = max_concurrent
        self.display = display

        desired = mode
        actual = client.concurrency
        if desired == "parallel" and actual == "sequential":
            logger.info(
                "Downgrading mode parallel→sequential because client=%s is sequential-only",
                client.name,
            )
            self.mode = "sequential"
        else:
            self.mode = desired

    async def run(self, items: list[BatchRequestItem]) -> BatchResponse:
        async with self.client.session():
            cached_by_id, uncached = self._partition_cache(items)

            if self.display is not None:
                self.display(ShardEvent(kind="submitting", extra={"num_cached": len(cached_by_id)}))

            if not uncached:
                return self._cached_only_response(items, cached_by_id)

            writer = self._write_shards(uncached)
            try:
                shard_runs = await self._execute_waves(uncached, writer)
            finally:
                writer.cleanup()

            return self._merge(items, cached_by_id, shard_runs)

    def _partition_cache(
        self, items: list[BatchRequestItem]
    ) -> tuple[dict[str, dict[str, Any]], list[BatchRequestItem]]:
        if self.cache is None:
            return {}, list(items)
        cached: dict[str, dict[str, Any]] = {}
        uncached: list[BatchRequestItem] = []
        for item in items:
            hit = _cache_get(self.cache, item)
            if hit is None:
                uncached.append(item)
                continue
            try:
                result_item = _litellm_response_to_result_item(hit)
            except Exception:
                logger.debug("failed to decode cached response for %s; treating as miss", item["id"], exc_info=True)
                uncached.append(item)
                continue
            cached[item["id"]] = {
                "custom_id": item["id"],
                CACHE_HIT_MARKER: result_item,
            }
        return cached, uncached

    def _cache_shard_results(self, shard_items: list[BatchRequestItem], outcome: ShardOutcome) -> None:
        """Write successful results from one completed shard to the cache.

        Errors (per-item failures inside the shard) are intentionally NOT cached —
        a rerun should retry them.
        """
        if self.cache is None:
            return
        by_id = {item["id"]: item for item in shard_items}
        for raw in outcome.results:
            cid = raw.get("custom_id")
            item = by_id.get(cid)
            if item is None:
                continue
            try:
                response = self.client.to_litellm_response(raw)
            except Exception:
                logger.warning("skipping cache write for %s: could not build litellm response", cid, exc_info=True)
                continue
            _cache_put(self.cache, item, response)

    def _count_tokens(self, items: list[BatchRequestItem]) -> list[Optional[int]]:
        counter = self.client.token_counter
        if counter is None:
            return [None] * len(items)
        counts: list[Optional[int]] = []
        for item in items:
            try:
                counts.append(counter(item["model"], item["messages"]))
            except Exception:
                logger.debug("token_counter failed for one item; skipping its token count", exc_info=True)
                counts.append(None)
        return counts

    def _write_shards(self, items: list[BatchRequestItem]) -> JSONLShardWriter:
        _, tmp_dir = ensure_batch_storage_dirs()
        base = tmp_dir / f"batch-{uuid.uuid4().hex[:8]}"
        writer = JSONLShardWriter(self.client, base)
        token_counts = self._count_tokens(items)
        for item, n_tokens in zip(items, token_counts, strict=True):
            writer.add(self.client.format_line(item), n_tokens=n_tokens)
        writer.finalize()
        logger.info(
            "BatchJobRunner shards: client=%s n_items=%d n_shards=%d",
            self.client.name, len(items), len(writer.shards),
        )
        return writer

    def _effective_enqueued_limits(self, model: str) -> EnqueuedLimits:
        fn = self.client.enqueued_limits_fn
        base = fn(model) if fn is not None else EnqueuedLimits()
        return EnqueuedLimits(
            max_enqueued_reqs=self.client.default_enqueued_reqs or base.max_enqueued_reqs,
            max_enqueued_tokens=self.client.default_enqueued_tokens or base.max_enqueued_tokens,
            max_enqueued_jobs=self.client.default_enqueued_jobs or base.max_enqueued_jobs,
        )

    def _shard_models(self, items: list[BatchRequestItem], shard_req_counts: list[int]) -> list[list[str]]:
        """Return the set of distinct models touched by each shard."""
        per_shard: list[list[str]] = []
        cursor = 0
        for count in shard_req_counts:
            seen: list[str] = []
            seen_set: set[str] = set()
            for item in items[cursor:cursor + count]:
                model = item["model"]
                if model not in seen_set:
                    seen.append(model)
                    seen_set.add(model)
            per_shard.append(seen)
            cursor += count
        return per_shard

    def _items_per_shard(
        self, items: list[BatchRequestItem], shard_req_counts: list[int]
    ) -> list[list[BatchRequestItem]]:
        out: list[list[BatchRequestItem]] = []
        cursor = 0
        for count in shard_req_counts:
            out.append(list(items[cursor:cursor + count]))
            cursor += count
        return out

    def _tightest_limits(self, models: list[str]) -> EnqueuedLimits:
        """Across models in one shard, take the most restrictive limit per field."""
        if not models:
            return self._effective_enqueued_limits("")

        def tighten(a: Optional[int], b: Optional[int]) -> Optional[int]:
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)

        limits = self._effective_enqueued_limits(models[0])
        for model in models[1:]:
            other = self._effective_enqueued_limits(model)
            limits = EnqueuedLimits(
                max_enqueued_reqs=tighten(limits.max_enqueued_reqs, other.max_enqueued_reqs),
                max_enqueued_tokens=tighten(limits.max_enqueued_tokens, other.max_enqueued_tokens),
                max_enqueued_jobs=tighten(limits.max_enqueued_jobs, other.max_enqueued_jobs),
            )
        return limits

    def _select_wave(
        self,
        pending: list[int],
        shard_req_counts: list[int],
        shard_token_counts: list[Optional[int]],
        shard_models: list[list[str]],
    ) -> list[int]:
        if not self.client.enable_concurrent_jobs:
            return [pending[0]]

        wave: list[int] = []
        used_reqs = 0
        used_tokens = 0
        used_jobs = 0
        for idx in pending:
            limits = self._tightest_limits(shard_models[idx])
            if limits.max_enqueued_jobs is not None and used_jobs + 1 > limits.max_enqueued_jobs:
                break
            if limits.max_enqueued_reqs is not None:
                if used_reqs + shard_req_counts[idx] > limits.max_enqueued_reqs and wave:
                    break
            if limits.max_enqueued_tokens is not None:
                tokens = shard_token_counts[idx]
                if tokens is not None and used_tokens + tokens > limits.max_enqueued_tokens and wave:
                    break
            wave.append(idx)
            used_reqs += shard_req_counts[idx]
            used_jobs += 1
            tokens = shard_token_counts[idx]
            if tokens is not None:
                used_tokens += tokens
        if not wave:
            # Always make progress — a single shard can exceed per-wave caps.
            wave = [pending[0]]
        return wave

    async def _execute_waves(
        self,
        items: list[BatchRequestItem],
        writer: JSONLShardWriter,
    ) -> list[_ShardRun]:
        shards = writer.shards
        total = len(shards)
        shard_req_counts = writer.shard_req_counts
        shard_token_counts = writer.shard_token_counts
        shard_models = self._shard_models(items, shard_req_counts)
        per_shard_items = self._items_per_shard(items, shard_req_counts)

        runs: list[_ShardRun] = [_ShardRun(items=per_shard_items[i]) for i in range(total)]

        async def _one(idx: int, shard: Path) -> None:
            reporter: ShardReporter = (
                _ReporterToDisplay(self.display, idx, total) if self.display else _NullReporter()
            )
            try:
                outcome = await self.client.execute_shard(shard, reporter)
            except BatchShardFailed as e:
                runs[idx].error = str(e)
                runs[idx].batch_id = e.batch_id
                logger.warning(
                    "shard %d/%d failed (batch_id=%s status=%s); %d items will be retried on rerun",
                    idx, total, e.batch_id, e.status, len(runs[idx].items),
                )
                if self.display:
                    self.display(ShardEvent(kind="shard_failed", shard_index=idx, total_shards=total,
                                            batch_id=e.batch_id))
                return
            except Exception as e:
                runs[idx].error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "shard %d/%d raised %s; %d items will be retried on rerun",
                    idx, total, type(e).__name__, len(runs[idx].items), exc_info=True,
                )
                if self.display:
                    self.display(ShardEvent(kind="shard_failed", shard_index=idx, total_shards=total))
                return

            runs[idx].outcome = outcome
            runs[idx].batch_id = outcome.batch_id
            try:
                self._cache_shard_results(runs[idx].items, outcome)
            except Exception:
                logger.warning("cache write failed for shard %d/%d", idx, total, exc_info=True)

            if self.display:
                self.display(ShardEvent(kind="shard_completed", shard_index=idx, total_shards=total,
                                        batch_id=outcome.batch_id))

        if self.mode == "sequential":
            for i, shard in enumerate(shards):
                await _one(i, shard)
            return runs

        pending: list[int] = list(range(total))
        while pending:
            wave = self._select_wave(pending, shard_req_counts, shard_token_counts, shard_models)
            if self.max_concurrent is not None and len(wave) > self.max_concurrent:
                wave = wave[: self.max_concurrent]
            await asyncio.gather(*[_one(i, shards[i]) for i in wave])
            submitted = set(wave)
            pending = [i for i in pending if i not in submitted]

        return runs

    def _cached_only_response(
        self, items: list[BatchRequestItem], cached: dict[str, dict[str, Any]]
    ) -> BatchResponse:
        batch_id = f"cached-{self.client.name}-{uuid.uuid4()}"
        ordered = [cached[item["id"]] for item in items if item["id"] in cached]
        if self.display:
            self.display(ShardEvent(kind="completed", batch_id=batch_id))
        return BatchResponse(
            batch_id=batch_id, status="completed", results=ordered, errors=None,
            raw_response={"source": "cache"},
        )

    def _merge(
        self,
        items: list[BatchRequestItem],
        cached: dict[str, dict[str, Any]],
        shard_runs: list[_ShardRun],
    ) -> BatchResponse:
        fresh_by_id: dict[str, dict[str, Any]] = {}
        errors: list[dict[str, Any]] = []
        batch_ids: list[str] = []
        for run in shard_runs:
            if run.batch_id:
                batch_ids.append(run.batch_id)
            if run.outcome is None:
                for item in run.items:
                    err_entry = {
                        "custom_id": item["id"],
                        "error": {
                            "message": run.error or "shard failed",
                            "shard_batch_id": run.batch_id,
                        },
                    }
                    errors.append(err_entry)
                continue
            for raw in run.outcome.results:
                cid = raw.get("custom_id")
                if cid:
                    fresh_by_id[cid] = raw
            for raw in run.outcome.errors or []:
                cid = raw.get("custom_id")
                if cid:
                    fresh_by_id[cid] = raw
                errors.append(raw)

        merged: list[dict[str, Any]] = []
        for item in items:
            cid = item["id"]
            raw = cached.get(cid) or fresh_by_id.get(cid)
            if raw is not None:
                merged.append(raw)

        if self.display:
            self.display(ShardEvent(kind="completed", batch_id=",".join(batch_ids)))

        return BatchResponse(
            batch_id=",".join(batch_ids),
            status="completed",
            results=merged,
            errors=errors if errors else None,
        )


def _resolve_cache(cache: Any) -> Any:
    """Resolve the cache argument.

    - ``None`` (default) → ``dspy.cache``
    - ``False`` → no cache
    - anything else → used as-is (must implement ``.get(request)`` / ``.put(request, value)``)
    """
    if cache is False:
        return None
    if cache is not None:
        return cache
    try:
        import dspy

        return getattr(dspy, "cache", None)
    except ImportError:
        return None


def _cache_request(item: BatchRequestItem, *, fn_identifier: str = _DSPY_SYNC_FN_IDENTIFIER) -> dict[str, Any]:
    """Build the request dict used as the cache key.

    Mirrors what ``dspy.clients.lm.LM.forward`` builds so a batch-written entry
    can be hit by a later plain ``dspy.LM(...)`` call with the same prompt.
    """
    request: dict[str, Any] = {"model": item["model"], "messages": item["messages"]}
    for k, v in item.get("lm_kwargs", {}).items():
        if k in _DSPY_IGNORED_CACHE_KEYS or k == "cache":
            continue
        request[k] = v
    request["_fn_identifier"] = fn_identifier
    return request


def _cache_get(cache: Any, item: BatchRequestItem) -> Any:
    request = _cache_request(item)
    try:
        return cache.get(request, _DSPY_IGNORED_CACHE_KEYS)
    except TypeError:
        return cache.get(request)


def _cache_put(cache: Any, item: BatchRequestItem, value: Any) -> None:
    for fn_identifier in (_DSPY_SYNC_FN_IDENTIFIER, _DSPY_ASYNC_FN_IDENTIFIER):
        request = _cache_request(item, fn_identifier=fn_identifier)
        try:
            cache.put(request, value, _DSPY_IGNORED_CACHE_KEYS)
        except TypeError:
            cache.put(request, value)


def _litellm_response_to_result_item(resp: Any) -> dict[str, Any]:
    """Decode a cached litellm ``ModelResponse`` into a ``ResultItem``-shaped dict."""
    if not hasattr(resp, "choices") or not resp.choices:
        raise ValueError("cached response has no choices")
    choice = resp.choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise ValueError("cached response choice has no message")

    content = getattr(message, "content", None) or ""
    if not isinstance(content, str):
        content = str(content)
    result: dict[str, Any] = {"text": content}

    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        reasoning = getattr(message, "reasoning", None)
    if reasoning is not None:
        result["reasoning_content"] = reasoning

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = tool_calls

    logprobs = getattr(choice, "logprobs", None)
    if logprobs is not None:
        result["logprobs"] = logprobs
    return result
