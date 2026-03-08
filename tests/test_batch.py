"""Tests for batch processing with different LM clients."""

import os
from pathlib import Path
from typing import Any

import dspy
import pytest
from modaic.batch.batch import abatch
from modaic.batch.clients import AnthropicBatchClient, OpenAIBatchClient
from modaic.batch.modal_client import ModalBatchClient
from modaic.batch.types import ABatchResult, FailedPrediction
from modaic.programs.predict import Predict

# Shared predictor and inputs for all tests
PREDICTOR = dspy.Predict("question -> answer")
INPUTS = [
    {"question": "What is the capital of France?"},
    {"question": "What is 2 + 2?"},
    {"question": "Who wrote Romeo and Juliet?"},
]

pytestmark = pytest.mark.asyncio


class _StubBatchClient:
    def __init__(self, raw_results: list[dict[str, Any]], parser_cls: type):
        self._raw_results = raw_results
        self._parser = object.__new__(parser_cls)

    async def submit_and_wait(self, batch_request, show_progress: bool = True):
        from modaic.batch.types import BatchReponse

        return BatchReponse(
            batch_id="batch-test",
            status="completed",
            results=self._raw_results,
            errors=None,
            raw_response={"source": "stub"},
        )

    def parse(self, raw_result: dict[str, Any]):
        return self._parser.parse(raw_result)


def _assert_predictions(results: list[ABatchResult]) -> None:
    assert len(results) == len(INPUTS)
    for res in results:
        pred = res.prediction
        assert not isinstance(pred, FailedPrediction), f"Prediction failed: {pred.error}"
        assert hasattr(pred, "answer")
        assert pred.answer is not None
        assert "messages" in res


async def _run_batch(model: str) -> list[ABatchResult]:
    with dspy.context(lm=dspy.LM(model), adapter=dspy.ChatAdapter()):
        return await abatch(PREDICTOR, INPUTS, show_progress=False)


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
    assert "reasoning_content" not in pred._outputs


async def test_modal_batch_client_requires_huggingface_model():
    with pytest.raises(ValueError, match="huggingface/<repo_path>"):
        ModalBatchClient(dspy.LM("openai/gpt-4o-mini"))


async def test_abatch_uses_explicit_modal_client_without_provider_resolution(monkeypatch):
    predictor = Predict("question -> answer")
    client = object.__new__(ModalBatchClient)
    client.provider = "modal"
    client.parse = lambda raw_result: {"text": raw_result["response"]["text"]}

    async def _submit_and_wait(batch_request, show_progress=True):
        del batch_request, show_progress
        from modaic.batch.types import BatchReponse

        return BatchReponse(
            batch_id="default_batch_id",
            status="completed",
            results=[{"custom_id": "request-0", "response": {"text": '{"answer":"Paris"}'}}],
            errors=None,
            raw_response={"source": "modal"},
        )

    client.submit_and_wait = _submit_and_wait

    monkeypatch.setattr("modaic.batch.batch._get_batch_context", lambda predictor: (_ for _ in ()).throw(AssertionError()))

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.JSONAdapter()):
        results = await predictor.abatch(
            [{"question": "What is the capital of France?"}],
            show_progress=True,
            client=client,
        )

    pred = results[0].prediction
    assert not isinstance(pred, FailedPrediction)
    assert pred.answer == "Paris"


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
            yield b"fake-output"

        @property
        def read_file(self):
            class _ReadFile:
                aio = self._read_file_aio

            return _ReadFile()

    class _FakeMethod:
        async def _remote_aio(self, **kwargs):
            captured["remote_kwargs"] = kwargs

        @property
        def remote(self):
            class _Remote:
                aio = self._remote_aio

            return _Remote()

    class _FakeGenerator:
        def run_generate_responses(self):
            return None

        run_generate_responses = _FakeMethod()

    class _FakeResponseGenerator:
        @staticmethod
        def with_options(**kwargs):
            captured["with_options_kwargs"] = kwargs
            return lambda: _FakeGenerator()

    class _FakeRun:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeEnableOutput:
        def __enter__(self):
            captured["enable_output"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("modaic.batch.modal_client.cache_volume", _FakeVolume())
    monkeypatch.setattr("modaic.batch.modal_client.ResponseGenerator", _FakeResponseGenerator)
    monkeypatch.setattr("modaic.batch.modal_client.app.run", lambda: _FakeRun())
    monkeypatch.setattr("modaic.batch.modal_client.modal.enable_output", lambda: _FakeEnableOutput())
    monkeypatch.setattr(
        "modaic.batch.modal_client.pd.read_parquet",
        lambda path: __import__("pandas").DataFrame(
            [{"response": {"text": "first"}}, {"response": {"text": "second", "reasoning_content": "trace"}}]
        ),
    )
    monkeypatch.setattr(
        "modaic.batch.modal_client.pd.DataFrame.to_parquet",
        lambda self, path, index=False: Path(path).write_bytes(b"input"),
    )

    batch_request = {
        "requests": [
            {"id": "request-0", "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "a"}], "lm_kwargs": {}},
            {"id": "request-1", "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "b"}], "lm_kwargs": {}},
        ],
        "model": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
        "lm_kwargs": {},
    }

    batch_id = await client.submit(batch_request)
    results = await client.get_results(batch_id)

    assert batch_id == "default_batch_id"
    assert captured["with_options_kwargs"] == {"gpu": "A100:2"}
    assert captured["enable_output"] is True
    assert captured["remote_kwargs"]["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert captured["remote_kwargs"]["temperature"] is None
    assert captured["remote_kwargs"]["top_p"] is None
    assert captured["remote_kwargs"]["top_k"] is None
    assert captured["remote_kwargs"]["min_p"] is None
    assert captured["remote_kwargs"]["max_tokens"] is None
    assert captured["remote_kwargs"]["repetition_penalty"] is None
    assert [item["custom_id"] for item in results.results] == ["request-0", "request-1"]
    assert client.parse(results.results[1]) == {"text": "second", "reasoning_content": "trace"}


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
