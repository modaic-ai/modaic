"""Integration tests for the examples API client methods."""

import json
import os
import time

import modaic
import httpx
import pytest
from modaic.hub import get_user_info
from modaic_client import ModaicClient
from modaic_client.client import (
    AnnotateExampleResponse,
    ExamplesPage,
    IngestExamplesResponse,
    PredictedExample,
)

from tests.utils import delete_program_repo

SOURCE_REPO = "modaic/preference-arbiter"

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")
if not MODAIC_TOKEN:
    pytest.skip("MODAIC_TOKEN not set", allow_module_level=True)

USERNAME = get_user_info(MODAIC_TOKEN)["login"]
TEST_PROGRAM = "examples-test"
TEST_REPO = f"{USERNAME}/{TEST_PROGRAM}"


def wait_for_example(client, example_id, timeout=15, interval=2):
    """Poll get_example until the example is available or timeout is reached."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return client.get_example(example_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                time.sleep(interval)
                continue
            raise
    raise TimeoutError(f"Example {example_id} not available after {timeout}s")

SAMPLE_INPUT = {
    "question": "What is the capital of France?",
    "response_A": "The capital of France is Paris, which has been the capital since the 10th century.",
    "response_B": "France's capital is Lyon, the second largest city.",
}

SAMPLE_INPUT_2 = {
    "question": "What is 2 + 2?",
    "response_A": "2 + 2 = 4",
    "response_B": "2 + 2 = 5, as famously stated in 1984.",
}

SAMPLE_INPUT_3 = {
    "question": "Who wrote Hamlet?",
    "response_A": "Hamlet was written by William Shakespeare around 1600.",
    "response_B": "Hamlet was written by Christopher Marlowe.",
}


@pytest.fixture(scope="module")
def client():
    # Clean up any leftover repo, then create a fresh one from the source arbiter
    delete_program_repo(username=USERNAME, program_name=TEST_PROGRAM, ignore_errors=True)
    judge = modaic.Predict.from_precompiled(SOURCE_REPO)
    judge.push_to_hub(TEST_REPO)

    yield ModaicClient()

    # Tear down
    delete_program_repo(username=USERNAME, program_name=TEST_PROGRAM, ignore_errors=True)


@pytest.fixture(scope="module")
def arbiter(client):
    return client.get_arbiter(TEST_REPO)


# =====================
# Ingest
# =====================


def test_ingest_examples(client):
    result = client.ingest_examples(
        [
            {
                "arbiter_repo": TEST_REPO,
                "input": json.dumps(SAMPLE_INPUT),
                "output": "A>B",
                "reasoning": "Response A correctly identifies Paris as the capital. Response B incorrectly states Lyon.",
                "arbiter_hash": "abc123",
            },
            {
                "arbiter_repo": TEST_REPO,
                "input": json.dumps(SAMPLE_INPUT_2),
                "output": "A>B",
                "reasoning": "Response A gives the correct answer. Response B is factually wrong.",
                "arbiter_hash": "abc123",
            },
        ]
    )

    assert isinstance(result, IngestExamplesResponse)
    assert result.queued is True
    assert len(result.example_ids) == 2
    assert all(isinstance(eid, str) and len(eid) > 0 for eid in result.example_ids)


def test_ingest_single_example(client):
    result = client.ingest_examples(
        [
            {
                "arbiter_repo": TEST_REPO,
                "input": json.dumps(SAMPLE_INPUT_3),
                "output": "A>B",
                "reasoning": "Shakespeare is the widely accepted author of Hamlet.",
            },
        ]
    )
    assert result.queued is True
    assert len(result.example_ids) == 1


def test_ingest_example_without_output(client):
    """Ingest an example with no output — should create example but no prediction."""
    result = client.ingest_examples(
        [
            {
                "arbiter_repo": TEST_REPO,
                "input": json.dumps(SAMPLE_INPUT),
            },
        ]
    )
    assert result.queued is True
    assert len(result.example_ids) == 1


def test_ingest_example_with_ground_truth(client):
    result = client.ingest_examples(
        [
            {
                "arbiter_repo": TEST_REPO,
                "input": json.dumps(SAMPLE_INPUT),
                "output": "A>B",
                "reasoning": "Paris is correct, Lyon is not the capital.",
                "ground_truth": "A>B",
                "ground_reasoning": "Response A is factually correct about Paris being the capital of France.",
            },
        ]
    )
    assert result.queued is True
    assert len(result.example_ids) == 1


def test_ingest_empty_body_fails(client):
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        client.ingest_examples([])
    assert exc_info.value.response.status_code == 400


def test_ingest_via_arbiter(arbiter):
    """Arbiter.ingest_examples should auto-fill arbiter_repo."""
    examples = [
        {
            "input": json.dumps(SAMPLE_INPUT_2),
            "output": "A>B",
            "reasoning": "Simple arithmetic confirms 2+2=4.",
        },
    ]
    result = arbiter.ingest_examples(examples)
    assert result.queued is True
    assert len(result.example_ids) == 1
    # Should have mutated the dict to include arbiter_repo
    assert examples[0]["arbiter_repo"] == TEST_REPO


# =====================
# List
# =====================


def test_list_examples(arbiter):
    """Wait for async ingestion, then list examples."""
    time.sleep(3)

    result = arbiter.list_examples(page_size=10)

    assert isinstance(result, ExamplesPage)
    assert result.page_size == 10
    assert isinstance(result.items, list)
    assert len(result.items) > 0
    for item in result.items:
        assert isinstance(item, PredictedExample)
        assert item.arbiter_repo == TEST_REPO
        assert item.id is not None


def test_list_examples_with_search(arbiter):
    time.sleep(1)
    result = arbiter.list_examples(search="capital")
    assert isinstance(result, ExamplesPage)
    for item in result.items:
        assert item.arbiter_repo == TEST_REPO


def test_list_examples_pagination(arbiter):
    page1 = arbiter.list_examples(page_size=1)
    assert len(page1.items) <= 1
    if page1.total_pages > 1:
        page2 = arbiter.list_examples(page=2, page_size=1)
        assert isinstance(page2, ExamplesPage)
        if page1.items and page2.items:
            assert page1.items[0].id != page2.items[0].id


def test_list_examples_via_client(client):
    user, program = TEST_REPO.split("/")
    result = client.list_examples(user=user, program=program, page_size=5)
    assert isinstance(result, ExamplesPage)
    assert result.page_size == 5


def test_list_examples_with_commit_hash(arbiter):
    result = arbiter.list_examples(commit_hash="abc123")
    assert isinstance(result, ExamplesPage)
    for item in result.items:
        assert item.arbiter_hash == "abc123"


# =====================
# Get by ID
# =====================


def test_get_example_by_id(client):
    """Ingest an example, wait for flush, then retrieve by ID."""
    ingest_result = client.ingest_examples(
        [
            {
                "arbiter_repo": TEST_REPO,
                "input": json.dumps(SAMPLE_INPUT_3),
                "output": "A>B",
                "reasoning": "Shakespeare is the correct author of Hamlet.",
                "arbiter_hash": "get123",
            },
        ]
    )
    example_id = ingest_result.example_ids[0]

    result = wait_for_example(client, example_id)

    assert isinstance(result, PredictedExample)
    assert result.id == example_id
    assert result.arbiter_repo == TEST_REPO
    assert result.output == "A>B"


def test_get_example_via_arbiter(arbiter):
    ingest_result = arbiter.ingest_examples(
        [
            {
                "input": json.dumps(SAMPLE_INPUT),
                "output": "A>B",
                "reasoning": "Paris is the capital, not Lyon.",
            },
        ]
    )
    example_id = ingest_result.example_ids[0]

    result = wait_for_example(arbiter.client, example_id)
    assert result.id == example_id


def test_get_example_not_found(client):
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        client.get_example("00000000-0000-0000-0000-000000000000")
    assert exc_info.value.response.status_code == 404


# =====================
# Annotate
# =====================


def test_annotate_example(client):
    """Ingest, wait for flush, then annotate with ground truth."""
    ingest_result = client.ingest_examples(
        [
            {
                "arbiter_repo": TEST_REPO,
                "input": json.dumps(SAMPLE_INPUT_2),
                "output": "A>B",
                "reasoning": "2+2=4 is correct.",
                "arbiter_hash": "ann123",
            },
        ]
    )
    example_id = ingest_result.example_ids[0]
    wait_for_example(client, example_id)

    result = client.annotate_example(
        example_id,
        [
            {
                "arbiter_repo": TEST_REPO,
                "ground_truth": "A>B",
                "ground_reasoning": "Response A correctly computes 2+2=4. Response B references a novel, not math.",
            },
        ],
    )

    assert isinstance(result, AnnotateExampleResponse)
    assert result.status == "success"

    updated = client.get_example(example_id)
    assert updated.ground_truth == "A>B"
    assert "correctly computes" in updated.ground_reasoning


def test_annotate_example_via_arbiter(arbiter):
    ingest_result = arbiter.ingest_examples(
        [
            {
                "input": json.dumps(SAMPLE_INPUT_3),
                "output": "A>B",
                "reasoning": "Shakespeare wrote Hamlet.",
            },
        ]
    )
    example_id = ingest_result.example_ids[0]
    wait_for_example(arbiter.client, example_id)

    result = arbiter.annotate_example(
        example_id,
        ground_truth="A>B",
        ground_reasoning="Shakespeare is the universally accepted author of Hamlet.",
    )
    assert result.status == "success"

    updated = arbiter.get_example(example_id)
    assert updated.ground_truth == "A>B"


def test_annotate_nonexistent_example(client):
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        client.annotate_example(
            "00000000-0000-0000-0000-000000000000",
            [{"arbiter_repo": TEST_REPO, "ground_truth": "A>B"}],
        )
    assert exc_info.value.response.status_code == 404
