"""Tests for batch processing with different LM clients."""

import os
from typing import Any

import dspy
import pytest
from modaic.batch.batch import abatch
from modaic.batch.clients import AnthropicBatchClient, OpenAIBatchClient
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
