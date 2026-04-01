from unittest.mock import MagicMock, patch

import pytest
from modaic_client.client import Arbiter, ArbiterPrediction, ModaicClient
from modaic_client.schemas import (
    AnnotateExampleResponse,
    IngestExamplesResponse,
)


@pytest.fixture
def mock_client():
    client = MagicMock(spec=ModaicClient)
    with patch("modaic_client.client.get_modaic_client", return_value=client):
        yield client


@pytest.fixture
def arbiter(mock_client):
    return Arbiter("testuser/testrepo", revision="v1")


# ── __init__ ──────────────────────────────────────────────────────────


def test_init_sets_repo_and_revision(mock_client):
    a = Arbiter("alice/myrepo", revision="v2")
    assert a.repo == "alice/myrepo"
    assert a.revision == "v2"


def test_init_default_revision(mock_client):
    a = Arbiter("alice/myrepo")
    assert a.revision == "main"


# ── properties ────────────────────────────────────────────────────────


def test_repo_user_property(arbiter):
    assert arbiter._repo_user == "testuser"


def test_repo_name_property(arbiter):
    assert arbiter._repo_name == "testrepo"


# ── to_dict / set_client ─────────────────────────────────────────────


def test_to_dict(arbiter):
    assert arbiter.to_dict() == {
        "arbiter_repo": "testuser/testrepo",
        "arbiter_revision": "v1",
    }


def test_set_client(arbiter):
    new_client = MagicMock(spec=ModaicClient)
    arbiter.set_client(new_client)
    assert arbiter.client is new_client


# ── predict ───────────────────────────────────────────────────────────


def test_predict_delegates_to_client(arbiter, mock_client):
    mock_pred = MagicMock(spec=ArbiterPrediction)
    mock_client.predict.return_value = mock_pred

    arbiter.predict(ground_truth="yes", ground_reasoning="because", question="what?")

    mock_client.predict.assert_called_once_with(
        {"question": "what?"},
        arbiter,
        "yes",
        "because",
    )


def test_predict_returns_client_result(arbiter, mock_client):
    mock_pred = MagicMock(spec=ArbiterPrediction)
    mock_client.predict.return_value = mock_pred

    prediction = arbiter.predict(question="what?")
    assert prediction is mock_pred


# ── ingest_examples ──────────────────────────────────────────────────


def test_ingest_examples_sets_arbiter_repo(arbiter, mock_client):
    mock_client.ingest_examples.return_value = MagicMock(spec=IngestExamplesResponse)
    examples = [{"input": "a"}, {"input": "b"}]

    arbiter.ingest_examples(examples)

    for ex in examples:
        assert ex["arbiter_repo"] == "testuser/testrepo"


def test_ingest_examples_preserves_existing_arbiter_repo(arbiter, mock_client):
    mock_client.ingest_examples.return_value = MagicMock(spec=IngestExamplesResponse)
    examples = [{"input": "a", "arbiter_repo": "other/repo"}]

    arbiter.ingest_examples(examples)

    assert examples[0]["arbiter_repo"] == "other/repo"


def test_ingest_examples_delegates_to_client(arbiter, mock_client):
    mock_resp = MagicMock(spec=IngestExamplesResponse)
    mock_client.ingest_examples.return_value = mock_resp
    examples = [{"input": "a"}]

    result = arbiter.ingest_examples(examples)

    mock_client.ingest_examples.assert_called_once_with(examples)
    assert result is mock_resp


# ── list_examples ────────────────────────────────────────────────────


def test_list_examples_delegates_with_correct_args(arbiter, mock_client):
    arbiter.list_examples(page=2, page_size=10, version=3, commit_hash="abc", search="query")

    mock_client.list_examples.assert_called_once_with(
        user="testuser",
        program="testrepo",
        page=2,
        page_size=10,
        version=3,
        commit_hash="abc",
        search="query",
    )


def test_list_examples_default_params(arbiter, mock_client):
    arbiter.list_examples()

    mock_client.list_examples.assert_called_once_with(
        user="testuser",
        program="testrepo",
        page=1,
        page_size=50,
        version=None,
        commit_hash=None,
        search=None,
    )


# ── get_example ──────────────────────────────────────────────────────


def test_get_example_delegates_to_client(arbiter, mock_client):
    arbiter.get_example("ex-123")
    mock_client.get_example.assert_called_once_with("ex-123")


# ── annotate_example ─────────────────────────────────────────────────


def test_annotate_example_with_both_fields(arbiter, mock_client):
    arbiter.annotate_example("ex-123", ground_truth="A", ground_reasoning="reason")

    mock_client.annotate_example.assert_called_once_with(
        "ex-123",
        [{"arbiter_repo": "testuser/testrepo", "ground_truth": "A", "ground_reasoning": "reason"}],
    )


def test_annotate_example_with_only_ground_truth(arbiter, mock_client):
    arbiter.annotate_example("ex-123", ground_truth="A")

    annotation = mock_client.annotate_example.call_args[0][1][0]
    assert annotation["ground_truth"] == "A"
    assert "ground_reasoning" not in annotation


def test_annotate_example_with_only_ground_reasoning(arbiter, mock_client):
    arbiter.annotate_example("ex-123", ground_reasoning="reason")

    annotation = mock_client.annotate_example.call_args[0][1][0]
    assert "ground_truth" not in annotation
    assert annotation["ground_reasoning"] == "reason"


def test_annotate_example_with_neither(arbiter, mock_client):
    arbiter.annotate_example("ex-123")

    annotation = mock_client.annotate_example.call_args[0][1][0]
    assert annotation == {"arbiter_repo": "testuser/testrepo"}


def test_annotate_example_returns_client_result(arbiter, mock_client):
    mock_resp = MagicMock(spec=AnnotateExampleResponse)
    mock_client.annotate_example.return_value = mock_resp

    result = arbiter.annotate_example("ex-123", ground_truth="A")
    assert result is mock_resp
