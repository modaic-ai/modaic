"""Tests for the examples API client methods on ModaicClient and Arbiter."""

import json
from unittest.mock import MagicMock, patch

import pytest
from modaic_client.client import (
    AnnotateExampleResponse,
    Arbiter,
    ExamplesPage,
    IngestExamplesResponse,
    ModaicClient,
    PredictedExample,
)


@pytest.fixture
def client():
    return ModaicClient(modaic_token="test-token")


@pytest.fixture
def arbiter(client):
    return client.get_arbiter("org/my-arbiter", revision="abc123")


# =====================
# ModaicClient.ingest_examples
# =====================


def test_ingest_examples_sends_ndjson(client):
    examples = [
        {"arbiter_repo": "org/my-arbiter", "input": {"text": "hello"}, "output": "positive"},
        {"arbiter_repo": "org/my-arbiter", "input": {"text": "bad"}, "output": "negative"},
    ]
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "queued": True,
        "example_ids": ["id-1", "id-2"],
    }
    mock_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.post", return_value=mock_response) as mock_post:
        result = client.ingest_examples(examples)

    assert isinstance(result, IngestExamplesResponse)
    assert result.queued is True
    assert result.example_ids == ["id-1", "id-2"]

    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["headers"]["Content-Type"] == "text/plain"
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer test-token"
    body = call_kwargs.kwargs["data"]
    lines = body.split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["output"] == "positive"
    assert json.loads(lines[1])["output"] == "negative"


def test_ingest_examples_single(client):
    examples = [{"arbiter_repo": "org/my-arbiter", "input": "test"}]
    mock_response = MagicMock()
    mock_response.json.return_value = {"queued": True, "example_ids": ["id-1"]}
    mock_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.post", return_value=mock_response):
        result = client.ingest_examples(examples)

    assert result.example_ids == ["id-1"]


def test_ingest_examples_raises_on_http_error(client):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("403 Forbidden")

    with patch("modaic_client.client.requests.post", return_value=mock_response):
        with pytest.raises(Exception, match="403 Forbidden"):
            client.ingest_examples([{"arbiter_repo": "org/repo", "input": "x"}])


# =====================
# ModaicClient.list_examples
# =====================


def test_list_examples_basic(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "items": [
            {
                "id": "ex-1",
                "arbiter_repo": "org/my-arbiter",
                "arbiter_hash": "abc",
                "input": {"text": "hello"},
                "output": "positive",
                "split": "none",
                "version": 0,
            }
        ],
        "limit": 50,
        "next_cursor": None,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.get", return_value=mock_response) as mock_get:
        result = client.list_examples(user="org", program="my-arbiter")

    assert isinstance(result, ExamplesPage)
    assert len(result.items) == 1
    assert result.items[0].id == "ex-1"
    assert result.items[0].arbiter_repo == "org/my-arbiter"
    assert result.limit == 50
    assert result.next_cursor is None

    params = mock_get.call_args.kwargs["params"]
    assert params["user"] == "org"
    assert params["program"] == "my-arbiter"
    assert params["limit"] == 50


def test_list_examples_with_all_filters(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"items": [], "limit": 10, "next_cursor": None}
    mock_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.get", return_value=mock_response) as mock_get:
        client.list_examples(
            user="org",
            program="my-arbiter",
            limit=10,
            cursor="abc123",
            version=2,
            search="hello",
        )

    params = mock_get.call_args.kwargs["params"]
    assert params["limit"] == 10
    assert params["cursor"] == "abc123"
    assert params["version"] == 2
    assert params["search"] == "hello"
    assert "commit_hash" not in params


def test_list_examples_with_commit_hash(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"items": [], "limit": 50, "next_cursor": None}
    mock_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.get", return_value=mock_response) as mock_get:
        client.list_examples(user="org", program="repo", commit_hash="deadbeef")

    params = mock_get.call_args.kwargs["params"]
    assert params["commit_hash"] == "deadbeef"
    assert "version" not in params


def test_list_examples_pagination(client):
    page1_response = MagicMock()
    page1_response.json.return_value = {
        "items": [{"id": "ex-1", "arbiter_repo": "org/repo"}],
        "limit": 1,
        "next_cursor": "cursor-token",
    }
    page1_response.raise_for_status = MagicMock()

    page2_response = MagicMock()
    page2_response.json.return_value = {
        "items": [{"id": "ex-2", "arbiter_repo": "org/repo"}],
        "limit": 1,
        "next_cursor": None,
    }
    page2_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.get", side_effect=[page1_response, page2_response]):
        page1 = client.list_examples(user="org", program="repo", limit=1)
        assert page1.next_cursor == "cursor-token"
        assert page1.items[0].id == "ex-1"

        page2 = client.list_examples(user="org", program="repo", limit=1, cursor=page1.next_cursor)
        assert page2.next_cursor is None
        assert page2.items[0].id == "ex-2"


# =====================
# ModaicClient.get_example
# =====================


def test_get_example(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "ex-123",
        "arbiter_repo": "org/my-arbiter",
        "arbiter_hash": "abc123",
        "input": {"text": "hello world"},
        "output": "positive",
        "reasoning": "Greeting detected",
        "ground_truth": "positive",
        "ground_reasoning": "This is indeed a greeting",
        "split": "train",
        "version": 2,
        "confidence": 0.95,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.get", return_value=mock_response) as mock_get:
        result = client.get_example("ex-123")

    assert isinstance(result, PredictedExample)
    assert result.id == "ex-123"
    assert result.output == "positive"
    assert result.ground_truth == "positive"
    assert result.version == 2
    assert result.confidence == 0.95

    url = mock_get.call_args.args[0]
    assert url.endswith("/api/v1/examples/ex-123")


def test_get_example_not_found(client):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")

    with patch("modaic_client.client.requests.get", return_value=mock_response):
        with pytest.raises(Exception, match="404 Not Found"):
            client.get_example("nonexistent-id")


# =====================
# ModaicClient.annotate_example
# =====================


def test_annotate_example(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "success"}
    mock_response.raise_for_status = MagicMock()

    annotations = [
        {
            "arbiter_repo": "org/my-arbiter",
            "ground_truth": "positive",
            "ground_reasoning": "Clearly positive",
        }
    ]

    with patch("modaic_client.client.requests.patch", return_value=mock_response) as mock_patch:
        result = client.annotate_example("ex-123", annotations)

    assert isinstance(result, AnnotateExampleResponse)
    assert result.status == "success"

    call_kwargs = mock_patch.call_args
    url = call_kwargs.args[0]
    assert url.endswith("/api/v1/examples/ex-123/annotation")
    assert call_kwargs.kwargs["json"]["annotations"] == annotations
    assert call_kwargs.kwargs["headers"]["Content-Type"] == "application/json"


def test_annotate_example_multiple_arbiters(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "success"}
    mock_response.raise_for_status = MagicMock()

    annotations = [
        {"arbiter_repo": "org/arbiter-1", "ground_truth": "positive"},
        {"arbiter_repo": "org/arbiter-2", "ground_truth": "greeting"},
    ]

    with patch("modaic_client.client.requests.patch", return_value=mock_response) as mock_patch:
        client.annotate_example("ex-123", annotations)

    sent_annotations = mock_patch.call_args.kwargs["json"]["annotations"]
    assert len(sent_annotations) == 2
    assert sent_annotations[0]["arbiter_repo"] == "org/arbiter-1"
    assert sent_annotations[1]["arbiter_repo"] == "org/arbiter-2"


# =====================
# Arbiter convenience methods
# =====================


def test_arbiter_ingest_examples_sets_repo(arbiter):
    mock_response = MagicMock()
    mock_response.json.return_value = {"queued": True, "example_ids": ["id-1"]}
    mock_response.raise_for_status = MagicMock()

    examples = [{"input": {"text": "hello"}, "output": "positive"}]

    with patch("modaic_client.client.requests.post", return_value=mock_response):
        result = arbiter.ingest_examples(examples)

    assert result.queued is True
    assert examples[0]["arbiter_repo"] == "org/my-arbiter"


def test_arbiter_ingest_examples_does_not_override_existing_repo(arbiter):
    mock_response = MagicMock()
    mock_response.json.return_value = {"queued": True, "example_ids": ["id-1"]}
    mock_response.raise_for_status = MagicMock()

    examples = [{"arbiter_repo": "other/repo", "input": "test"}]

    with patch("modaic_client.client.requests.post", return_value=mock_response):
        arbiter.ingest_examples(examples)

    assert examples[0]["arbiter_repo"] == "other/repo"


def test_arbiter_list_examples_splits_repo(arbiter):
    mock_response = MagicMock()
    mock_response.json.return_value = {"items": [], "limit": 50, "next_cursor": None}
    mock_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.get", return_value=mock_response) as mock_get:
        arbiter.list_examples(limit=20, search="test")

    params = mock_get.call_args.kwargs["params"]
    assert params["user"] == "org"
    assert params["program"] == "my-arbiter"
    assert params["limit"] == 20
    assert params["search"] == "test"


def test_arbiter_get_example(arbiter):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "ex-456",
        "arbiter_repo": "org/my-arbiter",
    }
    mock_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.get", return_value=mock_response):
        result = arbiter.get_example("ex-456")

    assert result.id == "ex-456"


def test_arbiter_annotate_example(arbiter):
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "success"}
    mock_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.patch", return_value=mock_response) as mock_patch:
        result = arbiter.annotate_example(
            "ex-123",
            ground_truth="positive",
            ground_reasoning="Clearly positive sentiment",
        )

    assert result.status == "success"
    sent = mock_patch.call_args.kwargs["json"]["annotations"]
    assert len(sent) == 1
    assert sent[0]["arbiter_repo"] == "org/my-arbiter"
    assert sent[0]["ground_truth"] == "positive"
    assert sent[0]["ground_reasoning"] == "Clearly positive sentiment"


def test_arbiter_annotate_example_partial(arbiter):
    """Only ground_truth, no ground_reasoning — should omit ground_reasoning from payload."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "success"}
    mock_response.raise_for_status = MagicMock()

    with patch("modaic_client.client.requests.patch", return_value=mock_response) as mock_patch:
        arbiter.annotate_example("ex-123", ground_truth="negative")

    sent = mock_patch.call_args.kwargs["json"]["annotations"]
    assert sent[0]["ground_truth"] == "negative"
    assert "ground_reasoning" not in sent[0]


# =====================
# Model validation
# =====================


def test_predicted_example_defaults():
    ex = PredictedExample(arbiter_repo="org/repo")
    assert ex.id is None
    assert ex.alt_id is None
    assert ex.arbiter_hash == ""
    assert ex.input is None
    assert ex.output is None
    assert ex.ground_truth is None
    assert ex.ground_reasoning == ""
    assert ex.messages is None
    assert ex.split is None
    assert ex.version is None
    assert ex.confidence is None


def test_predicted_example_full():
    ex = PredictedExample(
        id="ex-1",
        alt_id="alt-1",
        arbiter_repo="org/repo",
        arbiter_hash="abc123",
        input={"text": "hello"},
        output="positive",
        reasoning="Greeting",
        ground_truth="positive",
        ground_reasoning="Confirmed",
        messages=[{"role": "user", "content": "hello"}],
        split="train",
        version=3,
        prediction_timestamp="2025-01-15T12:00:00Z",
        confidence=0.95,
    )
    assert ex.id == "ex-1"
    assert ex.version == 3
    assert ex.confidence == 0.95
    assert ex.split == "train"


def test_examples_page_with_cursor():
    page = ExamplesPage(
        items=[PredictedExample(arbiter_repo="org/repo", id="ex-1")],
        limit=10,
        next_cursor="some-cursor",
    )
    assert page.next_cursor == "some-cursor"
    assert len(page.items) == 1


def test_examples_page_no_cursor():
    page = ExamplesPage(items=[], limit=50)
    assert page.next_cursor is None
    assert len(page.items) == 0
