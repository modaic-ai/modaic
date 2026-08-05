# ruff: noqa: ANN001, ANN201
"""Unit tests for ModaicClient.annotate_example v1/v2 endpoint routing.

These run without a live server: ``get_client`` is patched to yield a mock
httpx client so we can assert which path the PATCH targets.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from modaic_client.client import ModaicClient
from modaic_client.schemas import AnnotateExampleResponse


@pytest.fixture
def client_and_http():
    """A ModaicClient whose HTTP calls hit a mock returning ``status: success``."""
    mock_http = MagicMock()
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"status": "success"}
    mock_http.patch.return_value = resp

    client = ModaicClient(modaic_token="test-token", base_url="http://test")

    @contextmanager
    def fake_get_client(access_token=None):
        yield mock_http

    client.get_client = fake_get_client
    return client, mock_http


def _patched_path(mock_http):
    return mock_http.patch.call_args[0][0]


def test_dict_ground_truth_uses_v2(client_and_http):
    client, mock_http = client_and_http
    result = client.annotate_example(
        "ex-1",
        [{"arbiter_repo": "u/r", "ground_truth": {"verdict": "A>B"}}],
    )
    assert _patched_path(mock_http) == "/api/v2/examples/ex-1/annotation"
    assert isinstance(result, AnnotateExampleResponse)
    assert result.status == "success"


def test_reasoning_only_uses_v2(client_and_http):
    client, mock_http = client_and_http
    client.annotate_example("ex-2", [{"arbiter_repo": "u/r", "ground_reasoning": "because"}])
    assert _patched_path(mock_http) == "/api/v2/examples/ex-2/annotation"


def test_string_ground_truth_uses_v1_and_warns(client_and_http):
    client, mock_http = client_and_http
    with pytest.warns(DeprecationWarning, match="v1"):
        client.annotate_example("ex-3", [{"arbiter_repo": "u/r", "ground_truth": "A>B"}])
    assert _patched_path(mock_http) == "/api/v1/examples/ex-3/annotation"


def test_dict_ground_truth_does_not_warn(client_and_http):
    client, mock_http = client_and_http
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        client.annotate_example("ex-4", [{"arbiter_repo": "u/r", "ground_truth": {"k": "v"}}])


def test_body_passes_annotations_through(client_and_http):
    client, mock_http = client_and_http
    annotations = [{"arbiter_repo": "u/r", "ground_truth": {"verdict": "A>B"}, "ground_reasoning": "r"}]
    client.annotate_example("ex-5", annotations)
    assert mock_http.patch.call_args.kwargs["json"] == {"annotations": annotations}
