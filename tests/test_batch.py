"""Tests for batch processing with different LM clients."""

import importlib.util
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import dspy
import pytest
from modaic.batch import BatchJobRunner
from modaic.batch.batch import BatchAdapter, abatch
from modaic.batch.clients import AnthropicBatchClient, OpenAIBatchClient, VLLMBatchClient
from modaic.batch.clients.base import BatchClient, RemoteBatchClient, ShardReporter
from modaic.batch.types import ABatchResult, ABatchRow, BatchResponse, FailedPrediction, RawResults, ShardOutcome
from modaic.programs.predict import Predict

PREDICTOR = dspy.Predict("question -> answer")
INPUTS = [
    {"question": "What is the capital of France?"},
    {"question": "What is 2 + 2?"},
    {"question": "Who wrote Romeo and Juliet?"},
]

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(importlib.util.find_spec("duckdb") is None, reason="duckdb not installed"),
]


def _make_stub(parser_cls: type, raw_results: list[dict[str, Any]], name: str = "openai") -> BatchClient:
    """Create a BatchClient that skips real RPC and replays fixed raw_results."""
    stub_name = name

    class _StubClient(parser_cls):
        name = stub_name
        reqs_per_file = 50_000
        max_file_size = 100 * 1024 * 1024
        endpoint = "/v1/chat/completions"

        def __init__(self):
            self.api_key = None
            self.poll_interval = 1.0
            self.max_poll_time_s = 60.0
            self.execute_calls = 0
            self.items_formatted: list[dict[str, Any]] = []

        @asynccontextmanager
        async def session(self):
            yield

        async def execute_shard(self, shard: Path, reporter: ShardReporter) -> ShardOutcome:
            self.execute_calls += 1
            self.items_formatted.extend(
                __import__("json").loads(line) for line in shard.read_text().splitlines() if line
            )
            reporter.started("batch-test")
            reporter.percent(100)
            return ShardOutcome(batch_id="batch-test", results=list(raw_results), errors=[])

    return _StubClient()


@pytest.fixture(autouse=True)
def _batch_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("modaic_client.config.settings.modaic_cache", str(tmp_path / "modaic-cache"))
    # Isolate dspy.cache: BatchJobRunner now defaults to dspy.cache, which is a
    # global singleton. Replace it with a fresh memory-only Cache so tests don't
    # cross-contaminate.
    from dspy.clients.cache import Cache
    monkeypatch.setattr(
        "dspy.cache",
        Cache(enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir=str(tmp_path / "dspy-cache")),
    )


def _assert_predictions(results: ABatchResult) -> None:
    assert len(results) == len(INPUTS)
    for res in results:
        pred = res.prediction
        assert not isinstance(pred, FailedPrediction), f"Prediction failed: {pred.error}"
        assert hasattr(pred, "answer")
        assert pred.answer is not None
        assert isinstance(res, ABatchRow)
        assert isinstance(res.messages, list)
        assert set(res.outputs) == {"text", "reasoning_content"}
        assert isinstance(res.example, dict)
    assert results.path.name.endswith(".duckdb")
    assert results.path.exists()


async def test_batch_client_session_default_is_noop():
    class _MinimalBatchClient(BatchClient):
        name = "test"
        reqs_per_file = 10

        def parse_result(self, raw):
            del raw
            return {"text": ""}

        async def execute_shard(self, shard, reporter):
            reporter.started("x")
            return ShardOutcome(batch_id="x", results=[], errors=[])

    client = _MinimalBatchClient()
    entered = False
    async with client.session():
        entered = True
    assert entered is True


async def _run_batch(model: str) -> ABatchResult:
    with dspy.context(lm=dspy.LM(model), adapter=dspy.ChatAdapter()):
        grouped_results = await abatch([(PREDICTOR, INPUTS)], show_progress=False)
    return grouped_results[0][1]


async def test_abatch_groups_predict_results_into_separate_duckdb_files(monkeypatch):
    first_predictor = Predict("question -> answer")
    second_predictor = Predict("question -> answer")
    stub_client = _make_stub(
        OpenAIBatchClient,
        raw_results=[
            {"custom_id": "request-0", "response": {"body": {"choices": [{"message": {"content": '{"answer":"Paris"}'}}]}}},
            {"custom_id": "request-1", "response": {"body": {"choices": [{"message": {"content": '{"answer":"4"}'}}]}}},
            {"custom_id": "request-2", "response": {"body": {"choices": [{"message": {"content": '{"answer":"Shakespeare"}'}}]}}},
        ],
    )
    monkeypatch.setattr("modaic.batch.batch.get_batch_client", lambda *args, **kwargs: stub_client)

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.JSONAdapter()):
        grouped_results = await abatch(
            [
                (first_predictor, [{"question": "What is the capital of France?"}, {"question": "What is 2 + 2?"}]),
                (second_predictor, [{"question": "Who wrote Romeo and Juliet?"}]),
            ],
            show_progress=False,
        )

    assert stub_client.execute_calls == 1
    assert [item["custom_id"] for item in stub_client.items_formatted] == ["request-0", "request-1", "request-2"]

    returned_predictor, first_results = grouped_results[0]
    assert returned_predictor is first_predictor
    assert len(first_results) == 2
    assert first_results[0].prediction.answer == "Paris"
    assert first_results[1].prediction.answer == "4"
    assert first_results.path.name == "0.duckdb"
    assert first_results.path.parent.name == "batch-test"

    returned_predictor, second_results = grouped_results[1]
    assert returned_predictor is second_predictor
    assert len(second_results) == 1
    assert second_results[0].prediction.answer == "Shakespeare"
    assert second_results.path.name == "1.duckdb"


async def test_abatch_rejects_mixed_predict_providers():
    with dspy.context(adapter=dspy.JSONAdapter()):
        with pytest.raises(ValueError, match="same provider"):
            await abatch(
                [
                    (Predict("question -> answer", lm=dspy.LM("openai/gpt-4o-mini")), [{"question": "a"}]),
                    (Predict("question -> answer", lm=dspy.LM("anthropic/claude-3-5-haiku-latest")), [{"question": "b"}]),
                ],
                show_progress=False,
            )


async def test_abatch_chat_fallback_retries_failed_predictions_via_json_adapter(monkeypatch):
    """ChatAdapter failures should be re-run through JSONAdapter."""

    call_log: list[str] = []

    # First call uses ChatAdapter and returns mal-formed content for request-0 and request-2;
    # the fallback BatchJSONAdapter should retry those two.
    first_raw = [
        {"custom_id": "request-0", "response": {"body": {"choices": [{"message": {"content": "not chat-adapter formatted"}}]}}},
        {"custom_id": "request-1", "response": {"body": {"choices": [{"message": {"content": "[[ ## answer ## ]]\nParis"}}]}}},
        {"custom_id": "request-2", "response": {"body": {"choices": [{"message": {"content": "also broken"}}]}}},
    ]
    retry_raw = [
        {"custom_id": "request-0", "response": {"body": {"choices": [{"message": {"content": '{"answer":"Four"}'}}]}}},
        {"custom_id": "request-2", "response": {"body": {"choices": [{"message": {"content": '{"answer":"Shakespeare"}'}}]}}},
    ]

    class _SwitchingStub(OpenAIBatchClient):
        name = "openai"
        reqs_per_file = 50_000
        max_file_size = 100 * 1024 * 1024
        endpoint = "/v1/chat/completions"

        def __init__(self):
            self.api_key = None
            self.poll_interval = 1.0
            self.max_poll_time_s = 60.0
            self.n_calls = 0

        @asynccontextmanager
        async def session(self):
            call_log.append("enter")
            try:
                yield
            finally:
                call_log.append("exit")

        async def execute_shard(self, shard, reporter):
            self.n_calls += 1
            reporter.started(f"batch-{self.n_calls}")
            raw = first_raw if self.n_calls == 1 else retry_raw
            return ShardOutcome(batch_id=f"batch-{self.n_calls}", results=list(raw), errors=[])

    stub = _SwitchingStub()
    monkeypatch.setattr("modaic.batch.batch.get_batch_client", lambda *args, **kwargs: stub)

    first_predictor = Predict("question -> answer")
    second_predictor = Predict("question -> answer")
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.ChatAdapter()):
        grouped_results = await abatch(
            [
                (first_predictor, [{"question": "first"}, {"question": "second"}]),
                (second_predictor, [{"question": "third"}]),
            ],
            show_progress=False,
        )

    assert stub.n_calls == 2
    assert call_log.count("enter") == 2

    _, first_results = grouped_results[0]
    assert [row.prediction.answer for row in first_results] == ["Four", "Paris"]
    _, second_results = grouped_results[1]
    assert [row.prediction.answer for row in second_results] == ["Shakespeare"]


async def test_abatch_return_messages_outputs_openai_compatible_reasoning(monkeypatch):
    predictor = Predict("question -> answer")
    stub = _make_stub(
        OpenAIBatchClient,
        raw_results=[
            {
                "custom_id": "request-0",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": [{"type": "text", "text": '{"answer":"Paris"}'}],
                                    "reasoning_content": "step by step",
                                }
                            }
                        ]
                    }
                },
            }
        ],
    )
    monkeypatch.setattr("modaic.batch.batch.get_batch_client", lambda *args, **kwargs: stub)

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.JSONAdapter()):
        results = await predictor.abatch(
            [{"question": "What is the capital of France?"}],
            show_progress=False,
            return_messages=True,
        )

    pred = results[0].prediction
    assert not isinstance(pred, FailedPrediction)
    assert pred.answer == "Paris"
    assert pred._messages == results[0].messages
    assert pred._outputs["text"] == '{"answer":"Paris"}'
    assert pred._outputs["reasoning_content"] == "step by step"
    assert results[0].outputs == {"text": '{"answer":"Paris"}', "reasoning_content": "step by step"}
    assert results[0].example == {"question": "What is the capital of France?"}
    assert results.path.name == "0.duckdb"


async def test_abatch_return_messages_outputs_anthropic_thinking(monkeypatch):
    predictor = Predict("question -> answer")
    stub = _make_stub(
        AnthropicBatchClient,
        name="anthropic",
        raw_results=[
            {
                "custom_id": "request-0",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "trace"},
                            {"type": "text", "text": '{"answer":"Paris"}'},
                        ]
                    },
                },
            }
        ],
    )
    monkeypatch.setattr("modaic.batch.batch.get_batch_client", lambda *args, **kwargs: stub)

    with dspy.context(lm=dspy.LM("anthropic/claude-3-5-haiku-latest"), adapter=dspy.JSONAdapter()):
        results = await predictor.abatch(
            [{"question": "What is the capital of France?"}],
            show_progress=False,
            return_messages=True,
        )

    pred = results[0].prediction
    assert not isinstance(pred, FailedPrediction)
    assert pred.answer == "Paris"
    assert pred._outputs["reasoning_content"] == "trace"


async def test_abatch_return_messages_outputs_omits_reasoning_when_absent(monkeypatch):
    predictor = Predict("question -> answer")
    stub = _make_stub(
        OpenAIBatchClient,
        raw_results=[
            {
                "custom_id": "request-0",
                "response": {"body": {"choices": [{"message": {"content": '{"answer":"Paris"}'}}]}},
            }
        ],
    )
    monkeypatch.setattr("modaic.batch.batch.get_batch_client", lambda *args, **kwargs: stub)

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.JSONAdapter()):
        results = await predictor.abatch(
            [{"question": "What is the capital of France?"}],
            show_progress=False,
            return_messages=True,
        )

    pred = results[0].prediction
    assert pred._outputs["text"] == '{"answer":"Paris"}'
    assert pred._outputs["reasoning_content"] is None


async def test_vllm_batch_client_requires_huggingface_model():
    with pytest.raises(ValueError, match="huggingface/<repo_path>"):
        VLLMBatchClient(dspy.LM("openai/gpt-4o-mini"))


async def test_vllm_batch_client_parse_raises_row_error():
    client = object.__new__(VLLMBatchClient)
    with pytest.raises(ValueError, match="prompt_too_long"):
        client.parse_result({"custom_id": "request-0", "response": None, "error": "prompt_too_long"})


async def test_abatch_uses_explicit_vllm_client_without_provider_resolution(monkeypatch):
    predictor = Predict("question -> answer")
    client = object.__new__(VLLMBatchClient)
    client.api_key = None
    client.poll_interval = 1.0
    client.max_poll_time_s = 60.0
    client.lm = dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct")
    client.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    client.reqs_per_file = 10
    client.enable_thinking = False
    client.thinking_budget = None
    client._engine_client = object()

    @asynccontextmanager
    async def _session():
        yield

    async def _execute_shard(shard, reporter):
        reporter.started("00000000-0000-0000-0000-000000000777")
        return ShardOutcome(
            batch_id="00000000-0000-0000-0000-000000000777",
            results=[{"custom_id": "request-0", "response": {"text": '{"answer":"Paris"}'}}],
            errors=[],
        )

    client.session = _session
    client.execute_shard = _execute_shard

    monkeypatch.setattr(
        "modaic.batch.batch._get_batch_context", lambda predictor: (_ for _ in ()).throw(AssertionError())
    )

    with dspy.context(lm=dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct"), adapter=dspy.JSONAdapter()):
        results = await predictor.abatch(
            [{"question": "What is the capital of France?"}],
            show_progress=False,
            client=client,
        )

    pred = results[0].prediction
    assert not isinstance(pred, FailedPrediction)
    assert pred.answer == "Paris"
    assert results.batch_id == "00000000-0000-0000-0000-000000000777"


async def test_abatch_preserves_vllm_row_error_text(monkeypatch):
    predictor = Predict("question -> answer")
    client = object.__new__(VLLMBatchClient)
    client.api_key = None
    client.poll_interval = 1.0
    client.max_poll_time_s = 60.0
    client.lm = dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct")
    client.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    client.reqs_per_file = 10
    client.enable_thinking = False
    client.thinking_budget = None
    client._engine_client = object()

    @asynccontextmanager
    async def _session():
        yield

    async def _execute_shard(shard, reporter):
        reporter.started("00000000-0000-0000-0000-000000000888")
        return ShardOutcome(
            batch_id="00000000-0000-0000-0000-000000000888",
            results=[],
            errors=[{"custom_id": "request-0", "response": None, "error": "prompt_too_long"}],
        )

    client.session = _session
    client.execute_shard = _execute_shard

    monkeypatch.setattr(
        "modaic.batch.batch._get_batch_context", lambda predictor: (_ for _ in ()).throw(AssertionError())
    )

    with dspy.context(lm=dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct"), adapter=dspy.JSONAdapter()):
        results = await predictor.abatch(
            [{"question": "What is the capital of France?"}],
            show_progress=False,
            return_messages=True,
            client=client,
        )

    pred = results[0].prediction
    assert isinstance(pred, FailedPrediction)
    assert pred.error == "prompt_too_long"
    assert results[0].outputs == {"text": None, "reasoning_content": None}


async def test_runner_partitions_shards_by_request_cap(tmp_path, monkeypatch):
    """BatchJobRunner should produce multiple shards when request count exceeds the client cap."""

    seen_shards: list[int] = []

    class _RecorderClient(BatchClient):
        name = "tiny"
        reqs_per_file = 2
        max_file_size = 10_000_000
        endpoint = "/v1/chat/completions"
        safety_margin = 1.0

        @asynccontextmanager
        async def session(self):
            yield

        async def execute_shard(self, shard, reporter):
            line_count = sum(1 for line in shard.read_text().splitlines() if line.strip())
            seen_shards.append(line_count)
            reporter.started(f"batch-{len(seen_shards)}")
            return ShardOutcome(batch_id=f"batch-{len(seen_shards)}", results=[], errors=[])

        def parse_result(self, raw):
            return {"text": raw.get("text", "")}

    client = _RecorderClient()
    runner = BatchJobRunner(client, mode="sequential")

    items = [
        {"id": f"request-{i}", "model": "m", "messages": [{"role": "user", "content": str(i)}], "lm_kwargs": {}}
        for i in range(5)
    ]

    response = await runner.run(items)
    assert seen_shards == [2, 2, 1]
    assert response.status == "completed"


async def test_runner_caches_successful_shards_and_resumes_failed(tmp_path):
    """A rerun with the same items should resubmit only the failed shard's items.

    Shard 0 succeeds and its 2 items are written to cache. Shard 1 fails; those 2
    items stay uncached. On rerun, _partition_cache finds the 2 hits and only
    resubmits the 2 uncached items — so execute_shard is called with 2 items total.
    """
    from dspy.clients.cache import Cache
    from modaic.batch.clients.base import BatchShardFailed

    cache = Cache(
        enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir=str(tmp_path / "cache")
    )

    class _FlakyClient(BatchClient):
        name = "flaky"
        reqs_per_file = 2
        max_file_size = 10_000_000
        endpoint = "/v1/chat/completions"
        safety_margin = 1.0

        def __init__(self):
            self.calls: list[list[str]] = []
            self.fail_first_shard_on_first_call = True

        @asynccontextmanager
        async def session(self):
            yield

        async def execute_shard(self, shard, reporter):
            import json as _json
            lines = [_json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
            custom_ids = [line["custom_id"] for line in lines]
            self.calls.append(custom_ids)
            bid = f"batch-{len(self.calls)}"
            reporter.started(bid)
            # First shard of the first run must fail to simulate partial failure.
            if self.fail_first_shard_on_first_call and len(self.calls) == 2:
                # The second shard submitted — fail it.
                raise BatchShardFailed(bid, "failed", [])
            results = [
                {
                    "custom_id": cid,
                    "response": {
                        "body": {"choices": [{"message": {"content": f"answer-{cid}"}}]}
                    },
                }
                for cid in custom_ids
            ]
            return ShardOutcome(batch_id=bid, results=results, errors=[])

        def parse_result(self, raw):
            content = raw["response"]["body"]["choices"][0]["message"]["content"]
            return {"text": content}

    client = _FlakyClient()
    items = [
        {"id": f"request-{i}", "model": "m", "messages": [{"role": "user", "content": str(i)}], "lm_kwargs": {}}
        for i in range(4)
    ]

    runner1 = BatchJobRunner(client, cache=cache, mode="sequential")
    resp1 = await runner1.run(items)

    # First run submitted two shards (2+2 items).
    assert [sorted(c) for c in client.calls] == [["request-0", "request-1"], ["request-2", "request-3"]]
    assert len(resp1.results) == 2  # only shard-0 items returned
    assert resp1.errors is not None and len(resp1.errors) == 2
    assert {e["custom_id"] for e in resp1.errors} == {"request-2", "request-3"}

    # Second run: shard-0 items are cached; only shard-1 items should be resubmitted.
    client.fail_first_shard_on_first_call = False
    client.calls.clear()

    runner2 = BatchJobRunner(client, cache=cache, mode="sequential")
    resp2 = await runner2.run(items)

    assert client.calls == [["request-2", "request-3"]]
    assert len(resp2.results) == 4
    returned = {r.get("custom_id") for r in resp2.results}
    assert returned == {f"request-{i}" for i in range(4)}


async def test_runner_does_not_cache_errors(tmp_path):
    """Per-item errors inside a surviving shard's outcome.errors must not be cached."""
    from dspy.clients.cache import Cache
    from modaic.batch.runner import _cache_get

    cache = Cache(
        enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir=str(tmp_path / "cache")
    )

    class _ErrorEmittingClient(BatchClient):
        name = "err"
        reqs_per_file = 10
        max_file_size = 10_000_000
        endpoint = "/v1/chat/completions"
        safety_margin = 1.0

        @asynccontextmanager
        async def session(self):
            yield

        async def execute_shard(self, shard, reporter):
            reporter.started("b1")
            return ShardOutcome(
                batch_id="b1",
                results=[
                    {
                        "custom_id": "request-0",
                        "response": {"body": {"choices": [{"message": {"content": "ok"}}]}},
                    }
                ],
                errors=[{"custom_id": "request-1", "error": "boom"}],
            )

        def parse_result(self, raw):
            content = raw["response"]["body"]["choices"][0]["message"]["content"]
            return {"text": content}

    client = _ErrorEmittingClient()
    items = [
        {"id": "request-0", "model": "m", "messages": [{"role": "user", "content": "a"}], "lm_kwargs": {}},
        {"id": "request-1", "model": "m", "messages": [{"role": "user", "content": "b"}], "lm_kwargs": {}},
    ]

    await BatchJobRunner(client, cache=cache, mode="sequential").run(items)

    assert _cache_get(cache, items[0]) is not None
    assert _cache_get(cache, items[1]) is None


async def test_runner_cache_key_matches_dspy_lm_forward(tmp_path):
    """Batch cache writes must use the same key that dspy.LM.forward uses on lookup."""
    from dspy.clients.cache import Cache
    from modaic.batch.runner import _DSPY_IGNORED_CACHE_KEYS, _cache_request

    cache = Cache(
        enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir=str(tmp_path / "cache")
    )

    item = {
        "id": "x",
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "lm_kwargs": {"temperature": 0.0, "max_tokens": 1000, "api_key": "secret"},
    }

    # Simulate what dspy.LM.forward would compute: same messages+model+kwargs,
    # with the sync litellm_completion identifier added by the request_cache decorator.
    dspy_request = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 1000,
        "api_key": "secret",  # ignored by the cache key
        "_fn_identifier": "dspy.clients.lm.litellm_completion",
    }

    batch_request = _cache_request(item)
    # Both requests must produce identical keys (api_key is in the ignored list).
    assert cache.cache_key(batch_request, _DSPY_IGNORED_CACHE_KEYS) == cache.cache_key(
        dspy_request, _DSPY_IGNORED_CACHE_KEYS
    )


@pytest.mark.slow
async def test_openai_batch():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    predictions = await _run_batch("openai/gpt-4o-mini")
    _assert_predictions(predictions)


@pytest.mark.slow
async def test_together_batch():
    if not os.getenv("TOGETHERAI_API_KEY") and not os.getenv("TOGETHER_API_KEY"):
        pytest.skip("TOGETHERAI_API_KEY or TOGETHER_API_KEY not set")
    predictions = await _run_batch("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo")
    _assert_predictions(predictions)


@pytest.mark.slow
async def test_fireworks_batch():
    if not os.getenv("FIREWORKS_AI_API_KEY"):
        pytest.skip("FIREWORKS_AI_API_KEY not set")
    if not os.getenv("FIREWORKS_ACCOUNT_ID"):
        pytest.skip("FIREWORKS_ACCOUNT_ID not set")
    predictions = await _run_batch("fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct")
    _assert_predictions(predictions)
