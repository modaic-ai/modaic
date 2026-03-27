"""Tests for batch processing with different LM clients."""

import importlib.util
import gzip
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

import dspy
import pytest
from modaic.batch.batch import BatchAdapter, abatch
from modaic.batch.clients import AnthropicBatchClient, BatchClient, OpenAIBatchClient, VLLMBatchClient, VLLMBatchClient2
from modaic.batch.modal_client import ModalBatchClient
from modaic.batch.modal_job import ResponseGenerator
from modaic.batch.types import ABatchResult, ABatchRow, BatchReponse, FailedPrediction
from modaic.programs.predict import Predict

# Shared predictor and inputs for all tests
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


class _StubBatchClient:
    def __init__(self, raw_results: list[dict[str, Any]], parser_cls: type):
        self._raw_results = raw_results
        self._parser = object.__new__(parser_cls)
        self.submit_calls = 0
        self.batch_requests: list[dict[str, Any]] = []

    async def submit_and_wait(self, batch_request, show_progress: bool = True):
        self.submit_calls += 1
        self.batch_requests.append(batch_request)
        return BatchReponse(
            batch_id="batch-test",
            status="completed",
            results=self._raw_results,
            errors=None,
            raw_response={"source": "stub"},
        )

    def parse(self, raw_result: dict[str, Any]):
        return self._parser.parse(raw_result)

    @asynccontextmanager
    async def start(self):
        yield


@pytest.fixture(autouse=True)
def _batch_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("modaic_client.config.settings.modaic_cache", str(tmp_path / "modaic-cache"))


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


async def test_batch_client_start_default_is_noop():
    class _MinimalBatchClient(BatchClient):
        provider = "test"

        def format(self, batch_request):
            del batch_request
            return []

        def parse(self, raw_result: dict[str, Any]):
            del raw_result
            return {"text": ""}

        async def _submit_batch_request(self, batch_request):
            del batch_request
            return "batch-test"

        async def _get_status_impl(self, batch_id: str):
            del batch_id
            return "completed", 100

        async def get_results(self, batch_id: str) -> BatchReponse:
            del batch_id
            return BatchReponse(
                batch_id="batch-test",
                status="completed",
                results=[],
                errors=None,
                raw_response={"source": "test"},
            )

        async def cancel(self, batch_id: str) -> bool:
            del batch_id
            return False

    client = _MinimalBatchClient()
    entered = False

    async with client.start():
        entered = True

    assert entered is True


async def _run_batch(model: str) -> ABatchResult:
    with dspy.context(lm=dspy.LM(model), adapter=dspy.ChatAdapter()):
        grouped_results = await abatch([(PREDICTOR, INPUTS)], show_progress=False)
    return grouped_results[0][1]


async def test_abatch_groups_predict_results_into_separate_duckdb_files(monkeypatch):
    first_predictor = Predict("question -> answer")
    second_predictor = Predict("question -> answer")
    stub_client = _StubBatchClient(
        raw_results=[
            {
                "custom_id": "request-0",
                "response": {"body": {"choices": [{"message": {"content": '{"answer":"Paris"}'}}]}},
            },
            {
                "custom_id": "request-1",
                "response": {"body": {"choices": [{"message": {"content": '{"answer":"4"}'}}]}},
            },
            {
                "custom_id": "request-2",
                "response": {"body": {"choices": [{"message": {"content": '{"answer":"Shakespeare"}'}}]}},
            },
        ],
        parser_cls=OpenAIBatchClient,
    )
    monkeypatch.setattr("modaic.batch.batch.get_batch_client", lambda *args, **kwargs: stub_client)

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.JSONAdapter()):
        grouped_results = await abatch(
            [
                (
                    first_predictor,
                    [
                        {"question": "What is the capital of France?"},
                        {"question": "What is 2 + 2?"},
                    ],
                ),
                (
                    second_predictor,
                    [{"question": "Who wrote Romeo and Juliet?"}],
                ),
            ],
            show_progress=False,
        )

    assert stub_client.submit_calls == 1
    assert [request["id"] for request in stub_client.batch_requests[0]["requests"]] == [
        "request-0",
        "request-1",
        "request-2",
    ]

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
    assert second_results.path.parent.name == "batch-test"


async def test_abatch_rejects_mixed_predict_providers():
    with dspy.context(adapter=dspy.JSONAdapter()):
        with pytest.raises(ValueError, match="same provider"):
            await abatch(
                [
                    (Predict("question -> answer", lm=dspy.LM("openai/gpt-4o-mini")), [{"question": "a"}]),
                    (
                        Predict("question -> answer", lm=dspy.LM("anthropic/claude-3-5-haiku-latest")),
                        [{"question": "b"}],
                    ),
                ],
                show_progress=False,
            )


async def test_abatch_chat_fallback_retries_all_failed_examples_in_one_second_batch(monkeypatch):
    first_predictor = Predict("question -> answer")
    second_predictor = Predict("question -> answer")
    execute_calls: list[list[tuple[int, int, str]]] = []
    start_events: list[str] = []

    class _StartTrackingClient:
        @asynccontextmanager
        async def start(self):
            start_events.append("enter")
            try:
                yield
            finally:
                start_events.append("exit")

    async def _fake_execute_rows(self, request_contexts, batch_client, show_progress=True, return_messages=False):
        del self, show_progress, return_messages
        assert start_events == ["enter"]
        execute_calls.append(
            [(request.group_index, request.example_index, request.request_id) for request in request_contexts]
        )
        if len(execute_calls) == 1:
            return "batch-test", [
                ABatchRow(
                    prediction=FailedPrediction(error="parse", index=0),
                    messages=[],
                    outputs={"text": None, "reasoning_content": None},
                    example={"question": "first"},
                ),
                ABatchRow(
                    prediction=dspy.Prediction(answer="Paris"),
                    messages=[],
                    outputs={"text": '{"answer":"Paris"}', "reasoning_content": None},
                    example={"question": "second"},
                ),
                ABatchRow(
                    prediction=FailedPrediction(error="parse", index=0),
                    messages=[],
                    outputs={"text": None, "reasoning_content": None},
                    example={"question": "third"},
                ),
            ]

        assert [(group_index, example_index) for group_index, example_index, _ in execute_calls[-1]] == [(0, 0), (1, 0)]
        return "retry-batch", [
            ABatchRow(
                prediction=dspy.Prediction(answer="Four"),
                messages=[],
                outputs={"text": '{"answer":"Four"}', "reasoning_content": None},
                example={"question": "first"},
            ),
            ABatchRow(
                prediction=dspy.Prediction(answer="Shakespeare"),
                messages=[],
                outputs={"text": '{"answer":"Shakespeare"}', "reasoning_content": None},
                example={"question": "third"},
            ),
        ]

    monkeypatch.setattr(BatchAdapter, "_execute_rows", _fake_execute_rows)
    batch_client = _StartTrackingClient()
    monkeypatch.setattr("modaic.batch.batch.get_batch_client", lambda *args, **kwargs: batch_client)

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.ChatAdapter()):
        grouped_results = await abatch(
            [
                (first_predictor, [{"question": "first"}, {"question": "second"}]),
                (second_predictor, [{"question": "third"}]),
            ],
            show_progress=False,
        )

    assert len(execute_calls) == 2
    assert start_events == ["enter", "exit"]
    assert execute_calls[0] == [
        (0, 0, "request-0"),
        (0, 1, "request-1"),
        (1, 0, "request-2"),
    ]
    assert execute_calls[1] == [
        (0, 0, "request-0"),
        (1, 0, "request-2"),
    ]

    _, first_results = grouped_results[0]
    assert [row.prediction.answer for row in first_results] == ["Four", "Paris"]
    _, second_results = grouped_results[1]
    assert [row.prediction.answer for row in second_results] == ["Shakespeare"]


async def test_abatch_return_messages_outputs_openai_compatible_reasoning(monkeypatch):
    predictor = Predict("question -> answer")
    stub_client = _StubBatchClient(
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
        parser_cls=OpenAIBatchClient,
    )
    monkeypatch.setattr("modaic.batch.batch.get_batch_client", lambda *args, **kwargs: stub_client)

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
    assert results.path.parent.name == "batch-test"


async def test_abatch_return_messages_outputs_anthropic_thinking(monkeypatch):
    predictor = Predict("question -> answer")
    stub_client = _StubBatchClient(
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
        parser_cls=AnthropicBatchClient,
    )
    monkeypatch.setattr("modaic.batch.batch.get_batch_client", lambda *args, **kwargs: stub_client)

    with dspy.context(lm=dspy.LM("anthropic/claude-3-5-haiku-latest"), adapter=dspy.JSONAdapter()):
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
    assert pred._outputs["reasoning_content"] == "trace"
    assert results[0].outputs == {"text": '{"answer":"Paris"}', "reasoning_content": "trace"}


async def test_abatch_return_messages_outputs_omits_reasoning_when_absent(monkeypatch):
    predictor = Predict("question -> answer")
    stub_client = _StubBatchClient(
        raw_results=[
            {
                "custom_id": "request-0",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"answer":"Paris"}',
                                }
                            }
                        ]
                    }
                },
            }
        ],
        parser_cls=OpenAIBatchClient,
    )
    monkeypatch.setattr("modaic.batch.batch.get_batch_client", lambda *args, **kwargs: stub_client)

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.JSONAdapter()):
        results = await predictor.abatch(
            [{"question": "What is the capital of France?"}],
            show_progress=False,
            return_messages=True,
        )

    pred = results[0].prediction
    assert not isinstance(pred, FailedPrediction)
    assert pred._outputs["text"] == '{"answer":"Paris"}'
    assert pred._outputs["reasoning_content"] is None
    assert results[0].outputs == {"text": '{"answer":"Paris"}', "reasoning_content": None}


async def test_modal_batch_client_requires_huggingface_model():
    with pytest.raises(ValueError, match="huggingface/<repo_path>"):
        ModalBatchClient(dspy.LM("openai/gpt-4o-mini"))


async def test_vllm_batch_client_requires_huggingface_model():
    with pytest.raises(ValueError, match="huggingface/<repo_path>"):
        VLLMBatchClient(dspy.LM("openai/gpt-4o-mini"))


async def test_modal_batch_client_parse_raises_serialized_row_error():
    client = object.__new__(ModalBatchClient)

    with pytest.raises(ValueError, match="prompt_too_long"):
        client.parse({"custom_id": "request-0", "response": None, "error": "prompt_too_long"})


async def test_modal_batch_client_parse_reads_run_batch_chat_completion_shape():
    client = object.__new__(ModalBatchClient)

    parsed = client.parse(
        {
            "custom_id": "request-0",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": '{"answer":"Paris"}',
                                "reasoning": "trace",
                            }
                        }
                    ]
                }
            },
            "error": None,
        }
    )

    assert parsed == {"text": '{"answer":"Paris"}', "reasoning_content": "trace"}


async def test_vllm_batch_client_parse_raises_row_error():
    client = object.__new__(VLLMBatchClient)

    with pytest.raises(ValueError, match="prompt_too_long"):
        client.parse({"custom_id": "request-0", "response": None, "error": "prompt_too_long"})


async def test_abatch_uses_explicit_modal_client_without_provider_resolution(monkeypatch):
    predictor = Predict("question -> answer")
    client = object.__new__(ModalBatchClient)
    client.provider = "modal"
    client.parse = lambda raw_result: {"text": raw_result["response"]["text"]}

    @asynccontextmanager
    async def _start():
        yield

    async def _submit_and_wait(batch_request, show_progress=True):
        del batch_request, show_progress
        return BatchReponse(
            batch_id="00000000-0000-0000-0000-000000000321",
            status="completed",
            results=[{"custom_id": "request-0", "response": {"text": '{"answer":"Paris"}'}}],
            errors=None,
            raw_response={"source": "modal"},
        )

    client.start = _start
    client.submit_and_wait = _submit_and_wait

    monkeypatch.setattr(
        "modaic.batch.batch._get_batch_context", lambda predictor: (_ for _ in ()).throw(AssertionError())
    )

    with dspy.context(lm=dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct"), adapter=dspy.JSONAdapter()):
        results = await predictor.abatch(
            [{"question": "What is the capital of France?"}],
            show_progress=True,
            client=client,
        )

    pred = results[0].prediction
    assert not isinstance(pred, FailedPrediction)
    assert pred.answer == "Paris"
    assert results.batch_id == "00000000-0000-0000-0000-000000000321"


async def test_abatch_uses_explicit_vllm_client_without_provider_resolution(monkeypatch):
    predictor = Predict("question -> answer")
    client = object.__new__(VLLMBatchClient)
    client.provider = "vllm"
    client.parse = lambda raw_result: {"text": raw_result["response"]["text"]}
    client.lm = dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct")

    @asynccontextmanager
    async def _start():
        yield

    async def _submit_and_wait(batch_request, show_progress=True):
        del batch_request, show_progress
        return BatchReponse(
            batch_id="00000000-0000-0000-0000-000000000777",
            status="completed",
            results=[{"custom_id": "request-0", "response": {"text": '{"answer":"Paris"}'}}],
            errors=None,
            raw_response={"source": "vllm"},
        )

    client.start = _start
    client.submit_and_wait = _submit_and_wait

    monkeypatch.setattr(
        "modaic.batch.batch._get_batch_context", lambda predictor: (_ for _ in ()).throw(AssertionError())
    )

    with dspy.context(lm=dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct"), adapter=dspy.JSONAdapter()):
        results = await predictor.abatch(
            [{"question": "What is the capital of France?"}],
            show_progress=True,
            client=client,
        )

    pred = results[0].prediction
    assert not isinstance(pred, FailedPrediction)
    assert pred.answer == "Paris"
    assert results.batch_id == "00000000-0000-0000-0000-000000000777"


async def test_abatch_preserves_modal_row_error_text(monkeypatch):
    predictor = Predict("question -> answer")
    client = object.__new__(ModalBatchClient)
    client.provider = "modal"

    @asynccontextmanager
    async def _start():
        yield

    async def _submit_and_wait(batch_request, show_progress=True):
        del batch_request, show_progress
        return BatchReponse(
            batch_id="00000000-0000-0000-0000-000000000654",
            status="completed",
            results=[{"custom_id": "request-0", "response": None, "error": "prompt_too_long"}],
            errors=None,
            raw_response={"source": "modal"},
        )

    client.start = _start
    client.submit_and_wait = _submit_and_wait

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
    assert pred._messages == results[0].messages
    assert pred._outputs == {"text": None, "reasoning_content": None}
    assert results[0].outputs == {"text": None, "reasoning_content": None}


async def test_abatch_preserves_vllm_row_error_text(monkeypatch):
    predictor = Predict("question -> answer")
    client = object.__new__(VLLMBatchClient)
    client.provider = "vllm"
    client.lm = dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct")

    @asynccontextmanager
    async def _start():
        yield

    async def _submit_and_wait(batch_request, show_progress=True):
        del batch_request, show_progress
        return BatchReponse(
            batch_id="00000000-0000-0000-0000-000000000888",
            status="completed",
            results=[{"custom_id": "request-0", "response": None, "error": "prompt_too_long"}],
            errors=None,
            raw_response={"source": "vllm"},
        )

    client.start = _start
    client.submit_and_wait = _submit_and_wait

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
    assert pred._messages == results[0].messages
    assert pred._outputs == {"text": None, "reasoning_content": None}
    assert results[0].outputs == {"text": None, "reasoning_content": None}


async def test_vllm_batch_client_submit_uses_local_runner_and_assigns_custom_ids(monkeypatch):
    client = VLLMBatchClient(dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct", max_tokens=256))
    captured: dict[str, Any] = {"close_calls": 0}

    class _FakeRunner:
        def __init__(
            self,
            model_id: str,
            reasoning_parser: str,
            enforce_eager: bool,
            max_model_len: int | None,
            gpu_memory_utilization: float,
            tensor_parallel_size: int | None,
        ) -> None:
            captured["runner_init"] = {
                "model_id": model_id,
                "reasoning_parser": reasoning_parser,
                "enforce_eager": enforce_eager,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": gpu_memory_utilization,
                "tensor_parallel_size": tensor_parallel_size,
            }

        def generate(self, messages_batch, **kwargs):
            captured["messages_batch"] = messages_batch
            captured["generate_kwargs"] = kwargs
            return (
                [
                    {"response": {"text": "first", "reasoning_content": None}, "error": None},
                    {"response": {"text": "second", "reasoning_content": "trace"}, "error": None},
                ],
                {
                    "total": 2,
                    "failures": 0,
                    "reasoning_rows": 1,
                    "prompt_tokens": 10,
                    "output_tokens": 4,
                    "duration_s": 0.25,
                },
            )

        def close(self) -> None:
            captured["close_calls"] += 1

    monkeypatch.setattr("modaic.batch.clients.vllm.LocalVLLMRunner", _FakeRunner)
    monkeypatch.setattr("modaic.batch.clients.vllm.uuid.uuid4", lambda: UUID("00000000-0000-0000-0000-000000000999"))

    batch_request = {
        "requests": [
            {
                "id": "request-0",
                "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "a"}],
                "lm_kwargs": {},
            },
            {
                "id": "request-1",
                "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "b"}],
                "lm_kwargs": {},
            },
        ],
        "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
        "lm_kwargs": {},
    }

    async with client.start():
        batch_id = await client.submit(batch_request)
        results = await client.get_results(batch_id)

    assert batch_id == "00000000-0000-0000-0000-000000000999"
    assert captured["runner_init"]["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert captured["messages_batch"] == [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ]
    assert captured["generate_kwargs"]["max_tokens"] == 256
    assert [item["custom_id"] for item in results.results] == ["request-0", "request-1"]
    assert client.parse(results.results[0]) == {"text": "first"}
    assert client.parse(results.results[1]) == {"text": "second", "reasoning_content": "trace"}
    assert captured["close_calls"] == 1


async def test_vllm_batch_client2_submit_uses_run_batch_and_assigns_custom_ids(monkeypatch):
    client = VLLMBatchClient2(
        dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct", max_tokens=256, temperature=0.2)
    )
    captured: dict[str, Any] = {"close_calls": 0}

    class _FakeRunner:
        def __init__(
            self,
            model_id: str,
            reasoning_parser: str,
            enforce_eager: bool,
            max_model_len: int | None,
            gpu_memory_utilization: float,
            tensor_parallel_size: int | None,
        ) -> None:
            captured["runner_init"] = {
                "model_id": model_id,
                "reasoning_parser": reasoning_parser,
                "enforce_eager": enforce_eager,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": gpu_memory_utilization,
                "tensor_parallel_size": tensor_parallel_size,
            }

        def run_batch(self, input_path, output_path):
            captured["input_path"] = str(input_path)
            captured["output_path"] = str(output_path)
            captured["input_lines"] = Path(input_path).read_text().strip().splitlines()
            return (
                [
                    {
                        "custom_id": "request-0",
                        "response": {
                            "body": {
                                "choices": [{"message": {"content": "first"}}],
                                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                            }
                        },
                        "error": None,
                    },
                    {
                        "custom_id": "request-1",
                        "response": {
                            "body": {
                                "choices": [{"message": {"content": "second", "reasoning": "trace"}}],
                                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                            }
                        },
                        "error": None,
                    },
                ],
                {
                    "total": 2,
                    "failures": 0,
                    "reasoning_rows": 1,
                    "prompt_tokens": 7,
                    "output_tokens": 3,
                    "duration_s": 0.25,
                },
            )

        def close(self) -> None:
            captured["close_calls"] += 1

    monkeypatch.setattr("modaic.batch.clients.vllm.LocalVLLMRunBatchRunner", _FakeRunner)
    monkeypatch.setattr("modaic.batch.clients.vllm.uuid.uuid4", lambda: UUID("00000000-0000-0000-0000-000000001111"))

    batch_request = {
        "requests": [
            {
                "id": "request-0",
                "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "a"}],
                "lm_kwargs": {"max_tokens": 256, "temperature": 0.2},
            },
            {
                "id": "request-1",
                "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "b"}],
                "lm_kwargs": {"max_tokens": 256, "temperature": 0.2},
            },
        ],
        "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
        "lm_kwargs": {"max_tokens": 256, "temperature": 0.2},
    }

    async with client.start():
        batch_id = await client.submit(batch_request)
        results = await client.get_results(batch_id)

    assert batch_id == "00000000-0000-0000-0000-000000001111"
    assert captured["runner_init"]["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert captured["input_path"].endswith(".input.jsonl")
    assert captured["output_path"].endswith(".output.jsonl")
    first_line = captured["input_lines"][0]
    assert '"custom_id": "request-0"' in first_line
    assert '"url": "/v1/chat/completions"' in first_line
    assert '"model": "meta-llama/Llama-3.1-8B-Instruct"' in first_line
    assert [item["custom_id"] for item in results.results] == ["request-0", "request-1"]
    assert client.parse(results.results[0]) == {"text": "first"}
    assert client.parse(results.results[1]) == {"text": "second", "reasoning_content": "trace"}
    assert captured["close_calls"] == 1


async def test_modal_batch_client_submit_uses_modal_defaults_and_assigns_custom_ids(monkeypatch):
    client = ModalBatchClient(dspy.LM("huggingface/meta-llama/Llama-3.1-8B-Instruct"))
    captured: dict[str, Any] = {}

    class _FakeBatchUpload:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def put_file(self, local_path, remote_path):
            captured["uploaded_local_path"] = local_path
            captured["uploaded_remote_path"] = remote_path

    class _FakeVolume:
        def batch_upload(self, force=True):
            captured["batch_upload_force"] = force
            return _FakeBatchUpload()

        async def _read_file_aio(self, filename):
            captured["read_filename"] = filename
            yield gzip.compress(
                (
                    '{"custom_id":"request-0","response":{"body":{"choices":[{"message":{"content":"first"}}]}},"error":null}\n'
                    '{"custom_id":"request-2","response":{"body":{}},"error":"prompt_too_long"}\n'
                ).encode("utf-8")
            )

        @property
        def read_file(self):
            class _ReadFile:
                aio = self._read_file_aio

            return _ReadFile()

    class _FakeMethod:
        async def _remote_aio(self, **kwargs):
            captured["remote_kwargs"] = kwargs
            return {
                "total": 2,
                "failures": 1,
                "reasoning_rows": 0,
                "prompt_tokens": 3,
                "output_tokens": 1,
                "duration_s": 0.25,
            }

        @property
        def remote(self):
            class _Remote:
                aio = self._remote_aio

            return _Remote()

    class _FakeGenerator:
        generate = _FakeMethod()

    class _FakeResponseGenerator:
        @staticmethod
        def with_options(**kwargs):
            captured["with_options_kwargs"] = kwargs
            def _factory(**params):
                captured["generator_init_kwargs"] = params
                return _FakeGenerator()

            return _factory

    class _FakeRun:
        async def __aenter__(self):
            captured["app_run_entered"] = captured.get("app_run_entered", 0) + 1
            return self

        async def __aexit__(self, exc_type, exc, tb):
            captured["app_run_exited"] = captured.get("app_run_exited", 0) + 1
            return False

    class _FakeEnableOutput:
        def __enter__(self):
            captured["enable_output_entered"] = captured.get("enable_output_entered", 0) + 1
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["enable_output_exited"] = captured.get("enable_output_exited", 0) + 1
            return False

    monkeypatch.setattr("modaic.batch.modal_client.cache_volume", _FakeVolume())
    monkeypatch.setattr("modaic.batch.modal_client.ResponseGenerator", _FakeResponseGenerator)
    monkeypatch.setattr("modaic.batch.modal_client.app.run", lambda: _FakeRun())
    monkeypatch.setattr("modaic.batch.modal_client.modal.enable_output", lambda: _FakeEnableOutput())
    monkeypatch.setattr("modaic.batch.modal_client.uuid.uuid4", lambda: UUID("00000000-0000-0000-0000-000000000123"))

    batch_request = {
        "requests": [
            {
                "id": "request-0",
                "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "a"}],
                "lm_kwargs": {},
            },
            {
                "id": "request-2",
                "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "b"}],
                "lm_kwargs": {},
            },
        ],
        "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
        "lm_kwargs": {},
    }

    async with client.start():
        batch_id = await client.submit(batch_request)
        results = await client.get_results(batch_id)

    assert batch_id == "00000000-0000-0000-0000-000000000123"
    assert captured["with_options_kwargs"]["gpu"] == "A100:2"
    assert captured["enable_output_entered"] == 1
    assert captured["enable_output_exited"] == 1
    assert captured["app_run_entered"] == 1
    assert captured["app_run_exited"] == 1
    assert Path(captured["uploaded_local_path"]).name == "00000000-0000-0000-0000-000000000123.input.jsonl.gz"
    assert captured["uploaded_remote_path"] == "/input.jsonl.gz"
    assert captured["read_filename"] == "output.jsonl.gz"
    assert captured["remote_kwargs"]["input_path"] == "/cache/input.jsonl.gz"
    assert captured["remote_kwargs"]["output_path"] == "/cache/output.jsonl.gz"
    assert captured["generator_init_kwargs"]["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    with gzip.open(captured["uploaded_local_path"], "rt") as f:
        uploaded_lines = [line.strip() for line in f if line.strip()]
    assert '"url": "/v1/chat/completions"' in uploaded_lines[0]
    assert '"custom_id": "request-0"' in uploaded_lines[0]
    assert [item["custom_id"] for item in results.results] == ["request-0", "request-2"]
    assert client.parse(results.results[0]) == {"text": "first"}
    assert results.results[1]["error"] == "prompt_too_long"
    with pytest.raises(ValueError, match="prompt_too_long"):
        client.parse(results.results[1])


async def test_modal_response_generator_reloads_volume_before_generation(monkeypatch):
    call_order: list[str] = []

    monkeypatch.setattr("modaic.batch.modal_job.cache_volume.reload", lambda: call_order.append("reload"))
    monkeypatch.setattr("modaic.batch.modal_job.cache_volume.commit", lambda: call_order.append("commit"))
    monkeypatch.setattr("modaic.batch.modal_job.shutil.which", lambda name: "/usr/bin/vllm")

    def _fake_run(cli_args, check=False, env=None):
        del env
        assert check is False
        assert cli_args[0] == "vllm"
        assert call_order == ["reload"]
        call_order.append("run")
        output_index = cli_args.index("-o") + 1
        with open(cli_args[output_index], "w") as f:
            f.write(
                '{"custom_id":"request-0","response":{"body":{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}},"error":null}\n'
            )
        class _Completed:
            returncode = 0
            stderr = ""
            stdout = ""
        return _Completed()

    monkeypatch.setattr("modaic.batch.modal_job.subprocess.run", _fake_run)
    input_path = Path("/tmp/modal-job-test.input.jsonl.gz")
    output_path = Path("/tmp/modal-job-test.output.jsonl.gz")
    with gzip.open(input_path, "wt") as f:
        f.write(
            '{"custom_id":"request-0","method":"POST","url":"/v1/chat/completions","body":{"model":"x","messages":[{"role":"user","content":"hi"}]}}\n'
        )

    ResponseGenerator().generate.local(
        input_path=str(input_path),
        output_path=str(output_path),
    )

    assert call_order == ["reload", "run", "commit"]
    assert output_path.exists()
    with gzip.open(output_path, "rt") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 1
    input_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)


@pytest.mark.slow
async def test_openai_batch():
    """Test batch processing with OpenAI."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    predictions = await _run_batch("openai/gpt-4o-mini")

    _assert_predictions(predictions)


@pytest.mark.slow
async def test_together_batch():
    """Test batch processing with Together AI."""
    if not os.getenv("TOGETHERAI_API_KEY") and not os.getenv("TOGETHER_API_KEY"):
        pytest.skip("TOGETHERAI_API_KEY or TOGETHER_API_KEY not set")

    predictions = await _run_batch("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo")

    _assert_predictions(predictions)


@pytest.mark.slow
async def test_fireworks_batch():
    """Test batch processing with Fireworks AI."""
    if not os.getenv("FIREWORKS_AI_API_KEY"):
        pytest.skip("FIREWORKS_AI_API_KEY not set")
    if not os.getenv("FIREWORKS_ACCOUNT_ID"):
        pytest.skip("FIREWORKS_ACCOUNT_ID not set")

    predictions = await _run_batch("fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct")

    _assert_predictions(predictions)


# Anthropic is too slow. Leave out for now.
# @pytest.mark.slow
# async def test_anthropic_batch():
#     """Test batch processing with Anthropic."""
#     if not os.getenv("ANTHROPIC_API_KEY"):
#         pytest.skip("ANTHROPIC_API_KEY not set") # noqa: ERA001
#
#     predictions = await _run_batch("anthropic/claude-3-5-haiku-latest") # noqa: ERA001
#
#     _assert_predictions(predictions) # noqa: ERA001
