import pytest
from modaic_client.client import ArbiterPrediction, BatchExampleResult, BatchJob

from tests.modaic_client.conftest import requires_token

_BATCH_EXAMPLES = [
    {"input": {"subject": "You won a million dollars!", "body": "Click here to claim your prize..."}},
    {"input": {"subject": "Team standup notes", "body": "Action items from today's standup."}},
    {"input": {"subject": "Buy cheap watches now!", "body": "Limited time offer, click here!"}},
]


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
            compute_confidence=True,
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

    def test_predict_all_with_wait(self, test_arbiter_client, test_arbiter):
        results = test_arbiter_client.predict_all(
            examples=_BATCH_EXAMPLES,
            arbiters=[test_arbiter],
            wait_for_results=True,
            poll_interval=10.0,
        )

        assert isinstance(results, list)
        assert len(results) == len(_BATCH_EXAMPLES)
        for row in results:
            assert isinstance(row, BatchExampleResult)
            assert row.example_id
            assert len(row.predictions) == 1
            pred = row.predictions[0]
            assert isinstance(pred, ArbiterPrediction)
            assert pred.prediction_id
            assert pred.output is not None

    def test_predict_all_no_wait(self, test_arbiter_client, test_arbiter):
        job = test_arbiter_client.predict_all(
            examples=_BATCH_EXAMPLES,
            arbiters=[test_arbiter],
            wait_for_results=False,
        )

        assert isinstance(job, BatchJob)
        assert job.job_id
        assert job.total == len(_BATCH_EXAMPLES)

        results = job.wait(poll_interval=10.0)
        assert len(results) == len(_BATCH_EXAMPLES)
        assert all(isinstance(r, BatchExampleResult) for r in results)

    def test_predict_all_confidence(self, test_arbiter_client, test_arbiter):
        results = test_arbiter_client.predict_all(
            examples=_BATCH_EXAMPLES[:1],
            arbiters=[test_arbiter],
            wait_for_results=True,
            poll_interval=10.0,
        )

        prediction = results[0].predictions[0]
        confidence = prediction.confidence
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_arbiter_predict_all(self, test_arbiter):
        results = test_arbiter.predict_all(
            examples=_BATCH_EXAMPLES[:2],
            wait_for_results=True,
            poll_interval=10.0,
        )

        assert isinstance(results, list)
        assert len(results) == 2
        for row in results:
            assert isinstance(row, BatchExampleResult)
            assert len(row.predictions) == 1
