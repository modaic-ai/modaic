"""Tests for batch processing with different LM clients."""

import os

import dspy
import pytest

from modaic.batch.batch import abatch
from modaic.batch.types import FailedPrediction

# Shared predictor and inputs for all tests
PREDICTOR = dspy.Predict("question -> answer")
INPUTS = [
    {"question": "What is the capital of France?"},
    {"question": "What is 2 + 2?"},
    {"question": "Who wrote Romeo and Juliet?"},
]

pytestmark = pytest.mark.asyncio


def _assert_predictions(predictions: list[dspy.Prediction | FailedPrediction]) -> None:
    assert len(predictions) == len(INPUTS)
    for pred in predictions:
        assert not isinstance(pred, FailedPrediction), f"Prediction failed: {pred.error}"
        assert hasattr(pred, "answer")
        assert pred.answer is not None


async def _run_batch(model: str) -> list[dspy.Prediction | FailedPrediction]:
    with dspy.context(lm=dspy.LM(model), adapter=dspy.ChatAdapter()):
        return await abatch(PREDICTOR, INPUTS, show_progress=False)


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
#         pytest.skip("ANTHROPIC_API_KEY not set")
#
#     predictions = await _run_batch("anthropic/claude-3-5-haiku-latest")
#
#     _assert_predictions(predictions)
