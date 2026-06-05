# ruff: noqa: ANN001, ANN201
import os

import dspy
import modaic
import pytest
from modaic_client.client import ModaicClient

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")

requires_token = pytest.mark.skipif(not MODAIC_TOKEN, reason="MODAIC_TOKEN not set")

TEST_ARBITER_REPO = "modaic/test-arbiter"


class SpamClassifier(dspy.Signature):
    """Classify whether an email is spam or not."""

    subject: str = dspy.InputField(desc="The email subject line")
    body: str = dspy.InputField(desc="The email body")
    is_spam: bool = dspy.OutputField(desc="Whether the email is spam")


@pytest.fixture(scope="session")
def test_arbiter_client():
    return ModaicClient(timeout=120.0)


@pytest.fixture(scope="session")
def test_arbiter(test_arbiter_client):
    if not MODAIC_TOKEN:
        pytest.skip("MODAIC_TOKEN not set")

    test_arbiter_client.create_repo(TEST_ARBITER_REPO, exist_ok=True)

    predictor = modaic.Predict(
        SpamClassifier,
        lm=modaic.SafeLM(model="together_ai/openai/gpt-oss-120b"),
    )
    arbiter = predictor.as_arbiter()
    # push_to_hub is a no-op (warns) when there is nothing to commit, so no special handling needed.
    arbiter.push_to_hub(TEST_ARBITER_REPO, commit_message="test arbiter setup")

    return test_arbiter_client.get_arbiter(TEST_ARBITER_REPO)
