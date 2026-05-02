from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Optional

import dspy
from litellm import get_llm_provider

from .adapters import (
    BATCH_ADAPTERS,
    BatchAdapter,
    BatchChatAdapter,
    BatchJSONAdapter,
    BatchRequestContext,
    BatchXMLAdapter,
    get_batch_adapter,
)
from .clients.base import BatchClient, RemoteBatchClient
from .progress import BatchProgressDisplay, ProgressCallback, make_progress
from .runner import CACHE_HIT_MARKER, BatchJobRunner, Mode
from .types import (
    ABatchResult,
    ABatchRow,
    BatchRequestItem,
    BatchResponse,
    FailedPrediction,
    ResultItem,
)

logger = logging.getLogger(__name__)

GroupedBatchInputs = list[tuple[dspy.Predict, list[dict[str, Any]]]]

CLIENTS: dict[str, tuple[str, str]] = {
    "openai": ("modaic.batch.clients.openai", "OpenAIBatchClient"),
    "anthropic": ("modaic.batch.clients.anthropic", "AnthropicBatchClient"),
    "together_ai": ("modaic.batch.clients.together", "TogetherBatchClient"),
    "azure": ("modaic.batch.clients.azure", "AzureBatchClient"),
    "fireworks_ai": ("modaic.batch.clients.fireworks", "FireworksBatchClient"),
}


def get_batch_client(
    provider: str,
    api_key: Optional[str] = None,
    poll_interval: float = 30.0,
    max_poll_time: str = "24h",
) -> BatchClient:
    """Get a batch client for the given provider."""
    if provider not in CLIENTS:
        raise ValueError(
            f"Provider '{provider}' does not support batching. Supported providers: {list(CLIENTS.keys())}"
        )
    module_name, class_name = CLIENTS[provider]
    client_cls = getattr(import_module(module_name), class_name)
    return client_cls(api_key=api_key, poll_interval=poll_interval, max_poll_time=max_poll_time)


def _get_predictor_lm(predictor: dspy.Predict) -> dspy.LM:
    lm = getattr(predictor, "lm", None) or dspy.settings.lm
    if lm is None:
        raise ValueError(
            "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. "
            "e.g, `dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
        )
    return lm


def _get_batch_context(predictor: dspy.Predict) -> tuple[str, Optional[str]]:
    lm = _get_predictor_lm(predictor)
    _, provider, _, _ = get_llm_provider(lm.model)
    api_key = getattr(lm, "kwargs", {}).get("api_key")
    return provider, api_key


def _flatten_grouped_inputs(
    inputs: GroupedBatchInputs,
    custom_ids: Optional[list[str]] = None,
) -> list[BatchRequestContext]:
    contexts: list[BatchRequestContext] = []
    for group_index, (predictor, predictor_inputs) in enumerate(inputs):
        for example_index, input_example in enumerate(predictor_inputs):
            request_index = len(contexts)
            request_id = (
                custom_ids[request_index]
                if custom_ids is not None
                else f"request-{request_index}"
            )
            contexts.append(
                BatchRequestContext(
                    predictor=predictor,
                    inputs=dict(input_example),
                    group_index=group_index,
                    example_index=example_index,
                    request_index=request_index,
                    request_id=request_id,
                )
            )
    return contexts


def _request_index_from_custom_id(custom_id: Any) -> int:
    if isinstance(custom_id, str) and custom_id.startswith("request-"):
        try:
            return int(custom_id.split("-")[-1])
        except (ValueError, IndexError):
            pass
    return 0


def _resolve_grouped_batch_context(inputs: GroupedBatchInputs) -> tuple[str, Optional[str]]:
    providers: list[str] = []
    api_keys: set[str] = set()
    for predictor, _ in inputs:
        provider, api_key = _get_batch_context(predictor)
        providers.append(provider)
        if api_key is not None:
            api_keys.add(api_key)
    if len(set(providers)) != 1:
        raise ValueError("All predictors passed to modaic.batch.abatch must use the same provider")
    if len(api_keys) > 1:
        raise ValueError("All predictors passed to modaic.batch.abatch must use the same API key or pass `client=`")
    return providers[0], next(iter(api_keys), None)


def _validate_explicit_client(inputs: GroupedBatchInputs, client: BatchClient) -> None:
    provider_name = client.name
    if provider_name == "vllm":
        client_lm = getattr(client, "lm", None)
        client_model = getattr(client_lm, "model", None)
        for predictor, _ in inputs:
            predictor_model = getattr(_get_predictor_lm(predictor), "model", None)
            if not isinstance(predictor_model, str) or not predictor_model.startswith("huggingface/"):
                raise ValueError("vllm batch client requires all predictors to use `huggingface/...` models")
            if isinstance(client_model, str) and predictor_model != client_model:
                raise ValueError("vllm batch client requires all predictors to use the same model as the client LM")
        return

    for predictor, _ in inputs:
        predictor_provider, _ = _get_batch_context(predictor)
        if predictor_provider != provider_name:
            raise ValueError(
                f"Explicit batch client provider '{provider_name}' does not match predictor provider '{predictor_provider}'"
            )


def _parse_raw(client: BatchClient, raw: dict[str, Any]) -> ResultItem | FailedPrediction:
    cached = raw.get(CACHE_HIT_MARKER)
    if cached is not None:
        return cached
    try:
        return client.parse_result(raw)
    except Exception as exc:
        return FailedPrediction(error=str(exc), index=-1)


def _sort_results(
    contexts: list[BatchRequestContext],
    response: BatchResponse,
    client: BatchClient,
) -> list[ResultItem | FailedPrediction | None]:
    by_id: dict[int, ResultItem | FailedPrediction] = {}
    for raw in list(response.results) + list(response.errors or []):
        idx = _request_index_from_custom_id(raw.get("custom_id"))
        by_id[idx] = _parse_raw(client, raw)
    return [by_id.get(ctx.request_index) for ctx in contexts]


def _build_rows(
    contexts: list[BatchRequestContext],
    items: list[BatchRequestItem],
    predictions: list[dspy.Prediction | FailedPrediction],
    result_items: list[ResultItem | FailedPrediction | None],
    return_messages: bool,
) -> list[ABatchRow]:
    rows: list[ABatchRow] = []
    for ctx, item, pred, result_item in zip(contexts, items, predictions, result_items, strict=True):
        messages = list(item["messages"])
        text = None
        reasoning = None
        if isinstance(result_item, dict):
            text = result_item.get("text")
            reasoning = result_item.get("reasoning_content")
        outputs = {"text": text, "reasoning_content": reasoning}

        if return_messages:
            pred._messages = list(messages)
            pred._outputs = outputs

        rows.append(ABatchRow(prediction=pred, messages=messages, outputs=outputs, example=dict(ctx.inputs)))
    return rows


def _build_grouped_results(
    batch_id: str,
    inputs: GroupedBatchInputs,
    contexts: list[BatchRequestContext],
    rows: list[ABatchRow],
) -> list[tuple[dspy.Predict, ABatchResult]]:
    rows_by_group: list[list[tuple[int, ABatchRow]]] = [[] for _ in inputs]
    for ctx, row in zip(contexts, rows, strict=True):
        rows_by_group[ctx.group_index].append((ctx.example_index, row))

    grouped: list[tuple[dspy.Predict, ABatchResult]] = []
    for group_index, (predictor, _) in enumerate(inputs):
        group_rows = [r for _, r in sorted(rows_by_group[group_index], key=lambda x: x[0])]
        grouped.append((predictor, ABatchResult.from_rows(batch_id, group_rows, predict_index=group_index)))
    return grouped


async def abatch(
    inputs: GroupedBatchInputs,
    show_progress: bool = True,
    poll_interval: float = 30.0,
    max_poll_time: str = "24h",
    return_messages: bool = False,
    mode: Mode = "parallel",
    max_concurrent: Optional[int] = None,
    cache: Any = None,
    client: Optional[BatchClient] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[tuple[dspy.Predict, ABatchResult]]:
    """Execute a grouped batch request and return one ABatchResult per predictor group.

    Args:
        inputs: List of (predictor, input_examples) pairs.
        show_progress: Whether to render the live rich panel.
        poll_interval: Seconds between status polls for remote providers.
        max_poll_time: Maximum time to wait for remote shard completion.
        return_messages: Attach request messages / raw outputs to each prediction.
        mode: "parallel" runs shards concurrently; "sequential" runs them one-at-a-time.
              Automatically downgraded to sequential for providers that require it (e.g. vllm).
        max_concurrent: Cap on concurrent shards when mode="parallel". None = unbounded.
        cache: Cache backend. ``None`` (default) uses ``dspy.cache``; pass ``False`` to disable
               caching; otherwise a ``dspy.clients.cache.Cache``-compatible object. Successful
               shard results are written per-shard, so a rerun with the same inputs resumes
               only the items whose shard failed.
        client: Pre-built BatchClient. Required for non-API providers (vllm).
        progress_callback: Optional callback receiving ShardEvent updates.
    """
    if not inputs:
        return []

    contexts = _flatten_grouped_inputs(inputs)

    if client is None:
        provider_name, api_key = _resolve_grouped_batch_context(inputs)
        client = get_batch_client(provider_name, api_key=api_key, poll_interval=poll_interval, max_poll_time=max_poll_time)
    else:
        _validate_explicit_client(inputs, client)

    display = make_progress(show_progress, len(contexts), client.name, progress_callback)
    adapter = get_batch_adapter()

    async def _run(batch_adapter: BatchAdapter, sub_contexts: list[BatchRequestContext]):
        items = batch_adapter.format(sub_contexts)
        runner = BatchJobRunner(client, cache=cache, mode=mode, max_concurrent=max_concurrent, display=display)
        response = await runner.run(items)
        result_items = _sort_results(sub_contexts, response, client)
        predictions = batch_adapter.parse(sub_contexts, result_items)
        return predictions, items, result_items, response

    async def _run_with_fallback():
        predictions, items, result_items, response = await _run(adapter, contexts)

        fallback = adapter.fallback_for_failures()
        if fallback is not None and any(isinstance(p, FailedPrediction) for p in predictions):
            failed_idx: list[int] = []
            failed_ctx: list[BatchRequestContext] = []
            for i, pred in enumerate(predictions):
                if isinstance(pred, FailedPrediction):
                    failed_idx.append(i)
                    failed_ctx.append(contexts[i])
            logger.info("retrying %d failures with fallback %s", len(failed_ctx), type(fallback).__name__)

            r_preds, r_items, r_result_items, _ = await _run(fallback, failed_ctx)
            for original_i, pred, item, result_item in zip(failed_idx, r_preds, r_items, r_result_items, strict=True):
                predictions[original_i] = pred
                items[original_i] = item
                result_items[original_i] = result_item

        rows = _build_rows(contexts, items, predictions, result_items, return_messages)
        return response.batch_id, rows

    if display is not None and getattr(display, "_enabled", False):
        task = asyncio.create_task(_run_with_fallback())
        batch_id, rows = await display.run(task)
    else:
        batch_id, rows = await _run_with_fallback()

    return _build_grouped_results(batch_id, inputs, contexts, rows)


# ---------------------------------------------------------------------------
# Remote-only submit / poll / fetch
# ---------------------------------------------------------------------------


@dataclass
class ShardInfo:
    """Per-shard metadata returned at submit time.

    Lets a caller persist (e.g. to a database) the link between a provider
    batch_id and the input rows / custom_ids that landed in it, so results
    can be mapped back after a process restart.
    """

    batch_id: str
    n_rows: int
    file_size_bytes: int
    custom_ids: list[str]
    file_id: Optional[str] = None


@dataclass
class BatchJobHandle:
    """Handle for a submitted remote batch job spanning one or more provider shards."""

    shard_batch_ids: list[str]
    provider: str
    adapter_type: str
    shards: list[ShardInfo] = field(default_factory=list)

    @property
    def batch_id(self) -> str:
        return ",".join(self.shard_batch_ids)


def _count_tokens(client: RemoteBatchClient, item: BatchRequestItem) -> Optional[int]:
    counter = client.token_counter
    if counter is None:
        return None
    try:
        return counter(item["model"], item["messages"])
    except Exception:
        return None


async def plan_shards(
    predictor: dspy.Predict,
    inputs: list[dict],
    *,
    reqs_per_file: Optional[int] = None,
    max_file_size: Optional[int] = None,
    tokens_per_file: Optional[int] = None,
    dspy_adapter: Optional[dspy.Adapter] = None,
) -> list[list[int]]:
    """Plan how `inputs` would be partitioned into provider shards without uploading.

    Returns a list of shards; each shard is a list of input indices (into ``inputs``)
    that would land in that shard. Useful for callers that want to persist a
    per-shard plan ahead of time and submit shards in groups (see ``submit_shard``).

    ``dspy_adapter`` selects the formatter used to size the prompts (XML / JSON /
    Chat). Pass the same adapter you intend to use at submit time so the shard
    plan reflects the real upload bytes/tokens; otherwise this falls back to
    ``dspy.settings.adapter or ChatAdapter()`` which can mis-size prompts.
    """
    if not inputs:
        return []

    provider_name, api_key = _get_batch_context(predictor)
    client = get_batch_client(
        provider_name,
        api_key=api_key,
        # poll_interval/max_poll_time are unused here but the constructor wants them
    )
    if not isinstance(client, RemoteBatchClient):
        raise ValueError(f"plan_shards requires a remote/resumable provider; got {provider_name}")

    if reqs_per_file is not None:
        client.reqs_per_file = reqs_per_file
    if max_file_size is not None:
        client.max_file_size = max_file_size
    if tokens_per_file is not None:
        client.tokens_per_file = tokens_per_file

    adapter = get_batch_adapter(dspy_adapter)
    contexts = _flatten_grouped_inputs([(predictor, inputs)])
    items = adapter.format(contexts)

    import json as _json

    request_cap = client.request_cap
    byte_cap = client.byte_cap
    token_cap = client.token_cap

    shards: list[list[int]] = [[]]
    cur_n = 0
    cur_bytes = 0
    cur_tokens: Optional[int] = None

    for idx, item in enumerate(items):
        line = _json.dumps(client.format_line(item)) + "\n"
        size = len(line.encode("utf-8"))
        if size > client.max_file_size:
            raise ValueError(
                f"single request is {size}B which exceeds client {client.name} "
                f"max_file_size={client.max_file_size}"
            )
        n_tokens = _count_tokens(client, item)

        rolls = False
        if cur_n > 0:
            if cur_bytes + size > byte_cap:
                rolls = True
            elif cur_n + 1 > request_cap:
                rolls = True
            elif n_tokens is not None and token_cap is not None:
                cur = cur_tokens or 0
                if cur + n_tokens > token_cap:
                    rolls = True
        if rolls:
            shards.append([])
            cur_n = 0
            cur_bytes = 0
            cur_tokens = None

        shards[-1].append(idx)
        cur_n += 1
        cur_bytes += size
        if n_tokens is not None:
            cur_tokens = (cur_tokens or 0) + n_tokens

    return [s for s in shards if s]


async def submit_shard(
    predictor: dspy.Predict,
    shard_inputs: list[dict],
    *,
    custom_ids: Optional[list[str]] = None,
    dspy_adapter: Optional[dspy.Adapter] = None,
) -> ShardInfo:
    """Submit a single pre-planned shard and return its ``ShardInfo``.

    The caller is expected to have used ``plan_shards`` (or otherwise sized
    ``shard_inputs`` to fit) so that this set of inputs lands in exactly one
    provider shard. If the inputs would overflow the provider's per-file caps,
    this raises ``ValueError`` rather than silently splitting.

    ``dspy_adapter`` selects the formatter (XML / JSON / Chat). When omitted,
    falls back to ``dspy.settings.adapter or ChatAdapter()`` — pass it
    explicitly to avoid the global-settings trap.
    """
    if not shard_inputs:
        raise ValueError("submit_shard requires at least one input")
    if custom_ids is not None and len(custom_ids) != len(shard_inputs):
        raise ValueError("custom_ids must align 1:1 with shard_inputs")

    provider_name, api_key = _get_batch_context(predictor)
    client = get_batch_client(provider_name, api_key=api_key)
    if not isinstance(client, RemoteBatchClient):
        raise ValueError(f"submit_shard requires a remote/resumable provider; got {provider_name}")

    adapter = get_batch_adapter(dspy_adapter)
    contexts = _flatten_grouped_inputs(
        [(predictor, shard_inputs)],
        custom_ids=custom_ids,
    )
    items = adapter.format(contexts)
    resolved_custom_ids = [item["id"] for item in items]

    import uuid
    from pathlib import Path

    from .storage import ensure_batch_storage_dirs
    from .writer import JSONLShardWriter

    _, tmp_dir = ensure_batch_storage_dirs()
    base = Path(tmp_dir) / f"shard-{uuid.uuid4().hex[:8]}"
    writer = JSONLShardWriter(client, base)
    for item in items:
        writer.add(client.format_line(item), n_tokens=_count_tokens(client, item))
    writer.finalize()

    if len(writer.shards) != 1:
        writer.cleanup()
        raise ValueError(
            f"submit_shard inputs do not fit in a single provider shard "
            f"(got {len(writer.shards)} shards). Slice via plan_shards first."
        )

    shard_path = writer.shards[0]
    file_size_bytes = shard_path.stat().st_size

    async with client.session():
        batch_id = await client.create_batch(shard_path)

    return ShardInfo(
        batch_id=batch_id,
        n_rows=len(shard_inputs),
        file_size_bytes=file_size_bytes,
        custom_ids=resolved_custom_ids,
        file_id=None,
    )


async def submit_batch_job(
    predictor: dspy.Predict,
    inputs: list[dict],
    *,
    custom_ids: Optional[list[str]] = None,
    dspy_adapter: Optional[dspy.Adapter] = None,
) -> BatchJobHandle:
    """Submit a batch job without waiting for completion. Returns a resumable handle.

    Args:
        predictor: dspy Predict configured with a batch-capable LM.
        inputs: list of input dicts, one per request.
        custom_ids: optional list of caller-supplied stable IDs aligned 1:1 with
            ``inputs``. If provided, these IDs are used as the per-row
            ``custom_id`` in the provider's batch JSONL and round-trip onto
            each result row, so callers can map results back to source rows
            after a restart. If omitted, synthetic ``request-{i}`` IDs are
            used (back-compat).
        dspy_adapter: format selector (XML / JSON / Chat). When omitted, falls
            back to ``dspy.settings.adapter or ChatAdapter()`` — pass it
            explicitly to avoid the global-settings trap.

    Raises:
        ValueError: if the predictor's provider is not resumable (e.g. vllm),
            or if ``custom_ids`` is provided but doesn't align with ``inputs``.
    """
    if custom_ids is not None and len(custom_ids) != len(inputs):
        raise ValueError("custom_ids must align 1:1 with inputs")

    provider_name, api_key = _get_batch_context(predictor)
    client = get_batch_client(provider_name, api_key=api_key)
    if not isinstance(client, RemoteBatchClient):
        raise ValueError(f"submit_batch_job requires a remote/resumable provider; got {provider_name}")

    adapter = get_batch_adapter(dspy_adapter)
    contexts = _flatten_grouped_inputs([(predictor, inputs)], custom_ids=custom_ids)
    items = adapter.format(contexts)
    resolved_custom_ids = [item["id"] for item in items]

    import uuid
    from pathlib import Path

    from .storage import ensure_batch_storage_dirs
    from .writer import JSONLShardWriter

    _, tmp_dir = ensure_batch_storage_dirs()
    base = Path(tmp_dir) / f"submit-{uuid.uuid4().hex[:8]}"
    writer = JSONLShardWriter(client, base)
    for item in items:
        writer.add(client.format_line(item), n_tokens=_count_tokens(client, item))
    writer.finalize()

    # Walk shard files in order so we can attribute custom_ids back to each shard.
    shard_offset = 0
    shard_metadata: list[tuple[Path, int, int, list[str]]] = []
    for shard_path, n_rows in zip(writer.shards, writer.shard_req_counts, strict=True):
        ids_for_shard = resolved_custom_ids[shard_offset : shard_offset + n_rows]
        size_bytes = shard_path.stat().st_size
        shard_metadata.append((shard_path, n_rows, size_bytes, ids_for_shard))
        shard_offset += n_rows

    async with client.session():
        shard_ids: list[str] = []
        shards_info: list[ShardInfo] = []
        for shard_path, n_rows, size_bytes, ids_for_shard in shard_metadata:
            batch_id = await client.create_batch(shard_path)
            shard_ids.append(batch_id)
            shards_info.append(
                ShardInfo(
                    batch_id=batch_id,
                    n_rows=n_rows,
                    file_size_bytes=size_bytes,
                    custom_ids=ids_for_shard,
                    file_id=None,
                )
            )

    return BatchJobHandle(
        shard_batch_ids=shard_ids,
        provider=provider_name,
        adapter_type=type(adapter).__name__,
        shards=shards_info,
    )


async def aget_batch_status(
    batch_id: str, provider: str, api_key: Optional[str] = None
) -> tuple[str, Optional[int]]:
    client = get_batch_client(provider, api_key=api_key)
    if not isinstance(client, RemoteBatchClient):
        raise ValueError(f"aget_batch_status requires a remote provider; got {provider}")
    return await client.poll_status(batch_id)


async def aget_batch_results(batch_id: str, provider: str, api_key: Optional[str] = None) -> BatchResponse:
    client = get_batch_client(provider, api_key=api_key)
    if not isinstance(client, RemoteBatchClient):
        raise ValueError(f"aget_batch_results requires a remote provider; got {provider}")
    raw = await client.fetch_results(batch_id)
    return BatchResponse(
        batch_id=batch_id,
        status="completed",
        results=raw.results,
        errors=raw.errors if raw.errors else None,
        raw_response=raw.raw_response,
    )


async def acancel_batch(batch_id: str, provider: str, api_key: Optional[str] = None) -> bool:
    client = get_batch_client(provider, api_key=api_key)
    return await client.cancel(batch_id)


def supports_abatch(lm_or_model: dspy.LM | str) -> bool:
    model = lm_or_model.model if isinstance(lm_or_model, dspy.LM) else lm_or_model
    _, provider, _, _ = get_llm_provider(model)
    return provider in CLIENTS


__all__ = [
    "BATCH_ADAPTERS",
    "BatchAdapter",
    "BatchChatAdapter",
    "BatchJSONAdapter",
    "BatchJobHandle",
    "BatchProgressDisplay",
    "BatchRequestContext",
    "BatchXMLAdapter",
    "GroupedBatchInputs",
    "ShardInfo",
    "abatch",
    "acancel_batch",
    "aget_batch_results",
    "aget_batch_status",
    "get_batch_adapter",
    "get_batch_client",
    "plan_shards",
    "submit_batch_job",
    "submit_shard",
    "supports_abatch",
]
