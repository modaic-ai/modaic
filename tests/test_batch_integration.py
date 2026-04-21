"""Batch integration tests — real API round-trips.

Mark tiers:
  slow          — seconds-to-minutes; concurrency correctness at small scale (20–40 req)
  extra_slow    — minutes scale; hybrid concurrency at moderate load (2 k req, 4×500 shards)
  batch_soak — 24 h+; full provider shard-limit tests (100 k–3 M req per provider)

Run examples:
    uv run python -m pytest -m slow tests/test_batch_integration.py -s
    uv run python -m pytest -m extra_slow tests/test_batch_integration.py -s
    uv run python -m pytest -m batch_soak -k test_openai_extra_large_batch -s
    uv run python -m pytest -m "batch_soak and modal" -k test_vllm_extra_large_batch -s
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
from typing import Any

import dspy
import pytest
from datasets import load_dataset

from modaic import Predict
from modaic.batch import abatch, aget_batch_results, aget_batch_status, submit_batch_job
from modaic.batch.clients.base import BatchClient
from modaic.batch.clients.openai import OpenAIBatchClient
from modaic.batch.clients.together import TogetherBatchClient
from modaic.batch.types import ABatchResult, BatchRequestItem, FailedPrediction

pytestmark = [pytest.mark.asyncio]

# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

YELP_PREDICTOR = Predict(
    "review_text -> star_rating: Literal['1 star', '2 stars', '3 stars', '4 stars', '5 stars']"
)
BATCH_PREDICTOR = dspy.Predict("question -> answer")
YELP_VALID_LABELS = {"1 star", "2 stars", "3 stars", "4 stars", "5 stars"}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def yelp_base() -> list[dict[str, Any]]:
    """Load the full Yelp review train split once per session (~650k rows)."""
    ds = load_dataset("Yelp/yelp_review_full", split="train")
    return [{"review_text": row["text"]} for row in ds]


def _make_inputs(n: int) -> list[dict[str, Any]]:
    """Cycle through 5 reasoning-friendly questions to produce n inputs."""
    base = [
        {"question": "What is 17 × 23?"},
        {"question": "If x² − 5x + 6 = 0, what are the values of x?"},
        {"question": "What is the sum of interior angles in a hexagon?"},
        {"question": "How many prime numbers are there between 1 and 50?"},
        {"question": "If a train travels at 60 mph for 2.5 hours, how far does it go?"},
    ]
    return list(itertools.islice(itertools.cycle(base), n))


def _assert_messages_and_outputs(
    result: ABatchResult, n: int, *, assert_reasoning: bool = False
) -> None:
    """Assert at least one fully-successful row with valid content.

    Some rows legitimately fail parsing (e.g. model returns unstructured text);
    tests only require that at least one row passes every content check.
    """
    assert len(result) == n
    fully_successful = 0
    counts = {
        "failed_prediction": 0,
        "no_answer": 0,
        "no_messages": 0,
        "no_text": 0,
        "no_reasoning": 0,
    }
    sample_bad = None
    for row in result:
        if isinstance(row.prediction, FailedPrediction):
            counts["failed_prediction"] += 1
            if sample_bad is None:
                sample_bad = {"stage": "failed_prediction", "row": repr(row)[:800]}
            continue
        if not (hasattr(row.prediction, "answer") and row.prediction.answer):
            counts["no_answer"] += 1
            if sample_bad is None:
                sample_bad = {"stage": "no_answer", "row": repr(row)[:800]}
            continue
        if not (isinstance(row.messages, list) and len(row.messages) > 0):
            counts["no_messages"] += 1
            if sample_bad is None:
                sample_bad = {"stage": "no_messages", "row": repr(row)[:800]}
            continue
        outputs = row.outputs or {}
        if not (outputs.get("text") and len(outputs["text"]) > 0):
            counts["no_text"] += 1
            if sample_bad is None:
                sample_bad = {"stage": "no_text", "row": repr(row)[:800]}
            continue
        if assert_reasoning and not (
            outputs.get("reasoning_content") and len(outputs["reasoning_content"]) > 0
        ):
            counts["no_reasoning"] += 1
            if sample_bad is None:
                sample_bad = {"stage": "no_reasoning", "row": repr(row)[:800]}
            continue
        fully_successful += 1
    assert fully_successful >= 1, (
        f"No fully-successful rows out of {len(result)}. "
        f"Breakdown: {counts}. Sample bad row: {sample_bad}"
    )
    assert result.path.exists()


def _assert_yelp_messages_and_outputs(
    result: ABatchResult, n: int, *, assert_reasoning: bool = False
) -> None:
    """Yelp-predictor variant of _assert_messages_and_outputs.

    Requires at least one row with valid star_rating, non-empty messages,
    and non-empty outputs["text"] (plus reasoning_content if requested).
    """
    assert len(result) == n
    fully_successful = 0
    for row in result:
        if isinstance(row.prediction, FailedPrediction):
            continue
        if getattr(row.prediction, "star_rating", None) not in YELP_VALID_LABELS:
            continue
        if not (isinstance(row.messages, list) and len(row.messages) > 0):
            continue
        outputs = row.outputs or {}
        if not (outputs.get("text") and len(outputs["text"]) > 0):
            continue
        if assert_reasoning and not (
            outputs.get("reasoning_content") and len(outputs["reasoning_content"]) > 0
        ):
            continue
        fully_successful += 1
    assert fully_successful >= 1, f"No fully-successful rows out of {len(result)}"
    assert result.path.exists()


def _assert_raw_results_have_content(
    results: list[dict[str, Any]], *, assert_reasoning: bool = False
) -> None:
    """Check at least one raw OpenAI-compatible batch response has non-empty content.

    Used for tests that drive submit_batch_job + aget_batch_results (raw RawResults.results)
    rather than abatch. Tolerates parse failures: only requires one fully-successful row.
    """
    fully_successful = 0
    for record in results:
        if record.get("error"):
            continue
        body = (record.get("response") or {}).get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            continue
        message = (choices[0] or {}).get("message") or {}
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not text:
            continue
        if assert_reasoning and not reasoning:
            continue
        fully_successful += 1
    assert fully_successful >= 1, f"No fully-successful rows out of {len(results)}"


def _inputs_for_n_shards(
    client: BatchClient,
    model_str: str,
    base_inputs: list[dict[str, Any]],
    n_shards: int = 3,
) -> list[dict[str, Any]]:
    """Return enough inputs to force exactly n_shards for the given client.

    Samples one formatted JSONL line to measure the real byte size, then:
        rows_per_shard = min(request_cap, byte_cap // row_bytes)
        total = rows_per_shard * n_shards + 1
    Cycles base_inputs as needed so any provider limit can be reached.
    """
    sample: BatchRequestItem = {
        "id": "sample",
        "model": model_str,
        "messages": [{"role": "user", "content": base_inputs[0]["review_text"]}],
        "lm_kwargs": {},
    }
    row_bytes = len((json.dumps(client.format_line(sample)) + "\n").encode("utf-8"))

    if client.max_file_size < sys.maxsize // 2:
        byte_based = client.byte_cap // row_bytes
        rows_per_shard = min(client.request_cap, byte_based)
    else:
        rows_per_shard = client.request_cap

    total = rows_per_shard * n_shards + 1
    return list(itertools.islice(itertools.cycle(base_inputs), total))


# ---------------------------------------------------------------------------
# slow — concurrency correctness at small scale (20–40 req)
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_batch_2x10_sync() -> None:
    """2 shards of 10 run one-at-a-time (enable_concurrent_jobs=False). OpenAI / gpt-5-mini."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    client = OpenAIBatchClient(reqs_per_file=10, enable_concurrent_jobs=False)
    inputs = _make_inputs(20)

    with dspy.context(lm=dspy.LM("gpt-5-mini"), adapter=dspy.ChatAdapter()):
        grouped = await abatch([(BATCH_PREDICTOR, inputs)], client=client, return_messages=True)

    _, result = grouped[0]
    _assert_messages_and_outputs(result, 20)


@pytest.mark.slow
async def test_batch_2x10_async() -> None:
    """2 shards of 10 run concurrently (enable_concurrent_jobs=True). Together / gpt-oss-20b."""
    if not (os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY")):
        pytest.skip("TOGETHERAI_API_KEY or TOGETHER_API_KEY not set")

    client = TogetherBatchClient(reqs_per_file=10, enable_concurrent_jobs=True)
    inputs = _make_inputs(20)

    lm = dspy.LM("together_ai/openai/gpt-oss-20b", reasoning_effort="low")
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        grouped = await abatch([(BATCH_PREDICTOR, inputs)], client=client, return_messages=True)

    _, result = grouped[0]
    _assert_messages_and_outputs(result, 20, assert_reasoning=True)


@pytest.mark.slow
async def test_batch_4x10_hybrid() -> None:
    """4 shards of 10: wave 1=[shard0,shard1] concurrent, wave 2=[shard2,shard3] concurrent.
    max_concurrent=2 caps each wave. OpenAI / gpt-5-mini."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    client = OpenAIBatchClient(reqs_per_file=10)
    inputs = _make_inputs(40)

    with dspy.context(lm=dspy.LM("gpt-5-mini"), adapter=dspy.ChatAdapter()):
        grouped = await abatch(
            [(BATCH_PREDICTOR, inputs)],
            client=client,
            return_messages=True,
            max_concurrent=2,
        )

    _, result = grouped[0]
    _assert_messages_and_outputs(result, 40)


# ---------------------------------------------------------------------------
# extra_slow — hybrid concurrency at moderate load (2 k req)
# ---------------------------------------------------------------------------


@pytest.mark.extra_slow
async def test_batch_4x500_hybrid() -> None:
    """4 shards of 500 at max_concurrent=2; tests sync+async hybrid at moderate load.
    OpenAI / gpt-5-mini."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    client = OpenAIBatchClient(reqs_per_file=500)
    inputs = _make_inputs(2_000)

    with dspy.context(lm=dspy.LM("gpt-5-mini"), adapter=dspy.ChatAdapter()):
        grouped = await abatch(
            [(BATCH_PREDICTOR, inputs)],
            client=client,
            return_messages=True,
            max_concurrent=2,
        )

    _, result = grouped[0]
    _assert_messages_and_outputs(result, 2_000)


# ---------------------------------------------------------------------------
# batch_soak — full provider shard-limit tests (24 h+)
# ---------------------------------------------------------------------------


@pytest.mark.batch_soak
async def test_openai_extra_large_batch(yelp_base: list[dict[str, Any]]) -> None:
    """3 OpenAI shards (~47.5k requests each); uses blocking abatch()."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = "openai/gpt-4o-mini"
    client = OpenAIBatchClient()
    inputs = _inputs_for_n_shards(client, model, yelp_base)

    with dspy.context(lm=dspy.LM(model), adapter=dspy.ChatAdapter()):
        grouped = await abatch(
            [(YELP_PREDICTOR, inputs)],
            client=client,
            max_poll_time="48h",
            return_messages=True,
        )

    _, result = grouped[0]
    _assert_yelp_messages_and_outputs(result, len(inputs))


@pytest.mark.batch_soak
async def test_together_extra_large_batch(yelp_base: list[dict[str, Any]]) -> None:
    """3 Together shards (~47.5k requests each); uses async submit_batch_job + poll."""
    if not (os.getenv("TOGETHERAI_API_KEY") or os.getenv("TOGETHER_API_KEY")):
        pytest.skip("TOGETHERAI_API_KEY or TOGETHER_API_KEY not set")

    model = "together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo"
    client = TogetherBatchClient()
    inputs = _inputs_for_n_shards(client, model, yelp_base)

    with dspy.context(lm=dspy.LM(model), adapter=dspy.ChatAdapter()):
        handle = await submit_batch_job(YELP_PREDICTOR, inputs)

    while True:
        status, _ = await aget_batch_status(handle.batch_id, handle.provider)
        if status in ("completed", "failed", "cancelled", "expired"):
            break
        await asyncio.sleep(120)

    assert status == "completed", f"Batch ended with status: {status}"
    response = await aget_batch_results(handle.batch_id, handle.provider)
    assert len(response.results) > 0
    _assert_raw_results_have_content(response.results)


@pytest.mark.slow
@pytest.mark.modal
async def test_vllm_small_batch() -> None:
    """1 vLLM shard (batch_size=20 → 20 rows); smoke test for VLLMBatchClient inside a Modal H100."""
    import modal
    from tests.modal_app import app, run_vllm_batch

    modal.enable_output()
    async with app.run():
        result = await run_vllm_batch.remote.aio(
            model_id="openai/gpt-oss-20b",
            batch_size=20,
            n_rows=20,
            enforce_eager=True,
        )

    assert result["shards"] == 1
    assert result["total"] == 20
    assert result["fully_successful"] >= 1, f"No fully-successful rows: {result}"


@pytest.mark.slow
@pytest.mark.modal
async def test_vllm_2x10_batch() -> None:
    """2 vLLM shards (batch_size=10 → 20 rows); tests multi-shard sequential execution on Modal H100."""
    import modal
    from tests.modal_app import app, run_vllm_batch

    modal.enable_output()
    async with app.run():
        result = await run_vllm_batch.remote.aio(
            model_id="openai/gpt-oss-20b",
            batch_size=10,
            n_rows=20,
            enforce_eager=True,
        )

    assert result["shards"] == 2
    assert result["total"] == 20
    assert result["fully_successful"] >= 1, f"No fully-successful rows: {result}"


@pytest.mark.extra_slow
@pytest.mark.modal
async def test_vllm_extra_large_batch() -> None:
    """3 vLLM shards (batch_size=50k → 150k rows); runs VLLMBatchClient inside a Modal H100."""
    import modal
    from tests.modal_app import app, run_vllm_batch

    modal.enable_output()
    async with app.run():
        result = await run_vllm_batch.remote.aio(
            model_id="openai/gpt-oss-20b",
            batch_size=50_000,
            n_rows=150_000,
        )

    assert result["shards"] == 3
    assert result["total"] >= 135_000, f"Too many failures: total={result['total']}"
    assert result["fully_successful"] >= 1, f"No fully-successful rows: {result}"
