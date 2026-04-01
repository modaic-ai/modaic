import pytest

from modaic_client.client import ArbiterPrediction
from tests.modaic_client.conftest import requires_token


@requires_token
@pytest.mark.slow
class TestPredictIntegration:
    def test_predict_spam(self, test_arbiter_client, test_arbiter):
        result = test_arbiter_client.predict(
            input={"subject": "You won a million dollars!", "body": "Click here to claim your prize..."},
            arbiter=test_arbiter,
        )

        assert isinstance(result, ArbiterPrediction)
        assert result.example_id is not None
        assert result.prediction_id is not None
        assert result.output is not None
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_online_confidence(self, test_arbiter_client, test_arbiter):
        result = test_arbiter_client.predict(
            input={"subject": "Buy cheap watches now!", "body": "Limited time offer, click here!"},
            arbiter=test_arbiter,
        )

        confidence = result.confidence
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_predict_not_spam(self, test_arbiter_client, test_arbiter):
        result = test_arbiter_client.predict(
            input={
                "subject": "Team standup notes - March 31",
                "body": "Hi team, here are the action items from today's standup meeting.",
            },
            arbiter=test_arbiter,
        )

        assert isinstance(result, ArbiterPrediction)
        assert result.example_id is not None
        assert result.prediction_id is not None
        assert result.output is not None
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
