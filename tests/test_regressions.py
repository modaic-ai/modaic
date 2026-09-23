"""Request-contract regressions, each exercised by both public Python clients."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from typing import Any

import httpx
import pytest
from test_client import (
    EXAMPLE_ID,
    JOB_ID,
    MODEL_ID,
    alignment_json,
    batch_json,
    model_json,
    response_for,
)

from modaic import AsyncModaic, Modaic, ModaicAPIError, ModaicConnectionError, ModaicTimeoutError

QUESTIONS = {
    "refund": {
        "type": "noul",
        "instructions": "Refund eligibility",
        "criteria": {"true": "Eligible", "false": "Not eligible"},
    },
    "priority": {
        "type": "choice",
        "instructions": {"goal": "Prioritize"},
        "criteria": {"low": "Routine", "high": "Urgent"},
    },
    "quality": {"type": "score", "criteria": ["Poor", "Good", "Excellent"]},
}


async def call(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


class Harness:
    def __init__(self, asynchronous: bool) -> None:
        self.requests: list[httpx.Request] = []
        self.handler: Callable[[httpx.Request], httpx.Response] = response_for
        transport = httpx.MockTransport(self.record)
        self.http: Any = (httpx.AsyncClient if asynchronous else httpx.Client)(
            base_url="https://example.test/api/v1", transport=transport
        )
        self.client: Any = (AsyncModaic if asynchronous else Modaic)(
            api_key="test-key", http_client=self.http
        )

    def record(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    async def model(self) -> Any:
        return await call(self.client.models.get, workspace="farouk1", model="support-triage")


@pytest.fixture(params=[False, True], ids=["sync", "async"])
async def api(request: pytest.FixtureRequest) -> AsyncIterator[Harness]:
    harness = Harness(request.param)
    try:
        yield harness
    finally:
        await call(harness.client.close)
        # Injected HTTP clients remain caller-owned.
        assert not harness.http.is_closed
        if isinstance(harness.http, httpx.AsyncClient):
            await harness.http.aclose()
        else:
            harness.http.close()


async def test_create_all_question_types_and_options(api: Harness) -> None:
    questions = deepcopy(QUESTIONS)
    result = await call(
        api.client.models.create,
        workspace="acme",
        slug="support",
        default_branch="review",
        description="",
        model="typesafe/jev-latest",
        questions=questions,
    )
    assert len(api.requests) == 1
    assert json.loads(api.requests[0].content) == {
        "workspace": "acme",
        "slug": "support",
        "defaultBranch": "review",
        "description": "",
        "model": "typesafe/jev-latest",
        "questions": QUESTIONS,
    }
    assert questions == QUESTIONS
    assert result.workspace == "farouk1"
    assert "owner" not in result.model_dump()


async def test_update_branch_null_omission_and_revision_metadata(api: Harness) -> None:
    configuration = {
        "schemaVersion": 1,
        "checkpoint": 0,
        "questions": QUESTIONS,
        "capture": {"sampleRate": 0},
    }
    commit = {"commitSha": "new-sha", "previousSha": "old-sha", "branch": "review/a & b"}
    api.handler = lambda _: httpx.Response(
        200, json={**model_json(), "configuration": configuration, "commit": commit}
    )
    model = await call(
        api.client.models.update,
        MODEL_ID,
        branch=commit["branch"],
        description=None,
        questions=QUESTIONS,
        message="Revise",
    )
    assert api.requests[0].url.params["branch"] == commit["branch"]
    assert json.loads(api.requests[0].content) == {
        "description": None,
        "questions": QUESTIONS,
        "message": "Revise",
    }
    assert model.workspace.slug == "farouk1"
    assert model.configuration.checkpoint == 0
    assert model.configuration.capture == {"sampleRate": 0}
    assert model.configuration.questions == QUESTIONS
    assert model.commit.commit_sha == "new-sha"
    assert model.commit.previous_sha == "old-sha"
    await call(api.client.models.update, MODEL_ID, message="Only message")
    assert not api.requests[1].url.query
    assert json.loads(api.requests[1].content) == {"message": "Only message"}


async def test_multiple_model_handles_stay_isolated(api: Harness) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/models"):
            body = json.loads(request.content)
            return httpx.Response(201, json={**model_json(), **body, "id": body["slug"]})
        if request.url.path.endswith("/decision"):
            return response_for(request)
        # Rewrite only the fixture lookup, preserving the actual request for assertions.
        path = request.url.path.replace("/models/first/", f"/models/{MODEL_ID}/")
        path = path.replace("/models/second/", f"/models/{MODEL_ID}/")
        return response_for(httpx.Request(request.method, request.url.copy_with(path=path)))

    api.handler = handler
    first = await call(api.client.models.create, workspace="team-a", slug="first")
    second = await call(api.client.models.create, workspace="team-b", slug="second")
    api.requests.clear()
    for model in [second, first, second]:
        await call(model.decisions.create, state={})
        await call(model.examples.list)
        await call(model.jobs.alignments.list)
        await call(model.jobs.batch_decisions.list)
    for offset, workspace, model_id in [
        (0, "team-b", "second"),
        (4, "team-a", "first"),
        (8, "team-b", "second"),
    ]:
        assert json.loads(api.requests[offset].content)["model"] == f"{workspace}/{model_id}"
        assert all(
            r.url.path.startswith(f"/api/v1/models/{model_id}/")
            for r in api.requests[offset + 1 : offset + 4]
        )


async def test_ingest_and_annotation_preserve_falsy_values_without_mutation(api: Harness) -> None:
    model = await api.model()
    inputs = [
        {
            "id": EXAMPLE_ID,
            "state": {"nested": [False, 0, None, ""]},
            "annotation": {
                "ground_truth": {"refund": False, "score": 0},
                "ground_reasoning": "",
            },
        },
        {"state": None},
    ]
    before = deepcopy(inputs)
    await call(model.examples.ingest, examples=inputs)
    assert inputs == before
    assert json.loads(api.requests[1].content) == {
        "examples": [
            {
                "id": EXAMPLE_ID,
                "state": before[0]["state"],
                "annotation": {
                    "groundTruth": {"refund": False, "score": 0},
                    "groundReasoning": "",
                },
            },
            {"state": None},
        ]
    }
    await call(
        model.examples.annotate, EXAMPLE_ID, ground_truth={"refund": False}, ground_reasoning=""
    )
    assert json.loads(api.requests[2].content) == {
        "groundTruth": {"refund": False},
        "groundReasoning": "",
    }


@pytest.mark.parametrize(
    "selection,wire",
    [
        ({"example_ids": [EXAMPLE_ID]}, {"exampleIds": [EXAMPLE_ID]}),
        (
            {"examples": [{"id": EXAMPLE_ID, "state": None}]},
            {"examples": [{"id": EXAMPLE_ID, "state": None}]},
        ),
        ({"scope": "all"}, {"scope": "all"}),
    ],
)
async def test_batch_selection_modes(
    api: Harness, selection: dict[str, Any], wire: dict[str, Any]
) -> None:
    model = await api.model()
    before = deepcopy(selection)
    await call(
        model.jobs.batch_decisions.create,
        **selection,
        idempotency_key="selection-123",
        source_commit_sha="pinned",
    )
    assert json.loads(api.requests[1].content) == {
        **wire,
        "branch": "main",
        "sourceCommitSha": "pinned",
    }
    assert api.requests[1].headers["idempotency-key"] == "selection-123"
    assert selection == before


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"example_ids": [EXAMPLE_ID], "scope": "all"},
        {"example_ids": [EXAMPLE_ID], "examples": [{"state": {}}]},
        {"scope": "all", "examples": [{"state": {}}]},
        {"scope": "all", "examples": [{"state": {}}], "example_ids": [EXAMPLE_ID]},
    ],
)
async def test_conflicting_batch_modes_never_send_a_request(
    api: Harness, selection: dict[str, Any]
) -> None:
    model = await api.model()
    with pytest.raises(ValueError, match="exactly one"):
        await call(model.jobs.batch_decisions.create, **selection, idempotency_key="invalid-123")
    assert len(api.requests) == 1


@pytest.mark.parametrize(
    "options,reflection",
    [
        ({}, {"seed": 0}),
        ({"seed": 0}, {"seed": 0}),
        (
            {"reflection_model": "reflect", "reflection_minibatch_size": 2, "seed": 12},
            {"model": "reflect", "minibatchSize": 2, "seed": 12},
        ),
    ],
)
async def test_alignment_defaults_and_overrides(
    api: Harness, options: dict[str, Any], reflection: dict[str, Any]
) -> None:
    model = await api.model()
    await call(
        model.jobs.alignments.create,
        branch="main",
        source_commit_sha="pinned",
        max_metric_calls=10,
        idempotency_key="alignment-123",
        **options,
    )
    assert json.loads(api.requests[1].content) == {
        "branch": "main",
        "sourceCommitSha": "pinned",
        "budget": {"maxMetricCalls": 10},
        "reflection": reflection,
    }
    assert api.requests[1].headers["idempotency-key"] == "alignment-123"


@pytest.mark.parametrize("resource", ["alignments", "batch_decisions"])
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
async def test_wait_polls_through_running_to_each_terminal_state(
    api: Harness, resource: str, status: str
) -> None:
    states = iter(["queued", "running", status])
    factory = alignment_json if resource == "alignments" else batch_json
    error = {"code": "job_failed", "message": "Could not complete"} if status == "failed" else None
    api.handler = lambda _: httpx.Response(200, json={**factory(next(states)), "error": error})
    result = await call(getattr(api.client, resource).wait, JOB_ID, poll_interval=0)
    assert result.status == status
    assert (result.error.model_dump() if result.error else None) == error
    assert len(api.requests) == 3
    assert all(r.method == "GET" for r in api.requests)


@pytest.mark.parametrize("resource", ["alignments", "batch_decisions"])
async def test_wait_timeout_does_not_cancel_job(api: Harness, resource: str) -> None:
    factory = alignment_json if resource == "alignments" else batch_json
    api.handler = lambda _: httpx.Response(200, json=factory("running"))
    with pytest.raises(ModaicTimeoutError):
        await call(getattr(api.client, resource).wait, JOB_ID, timeout=0, poll_interval=0)
    assert len(api.requests) == 1
    assert api.requests[0].method == "GET"


@pytest.mark.parametrize("status", [401, 403, 404, 409, 422, 429, 500])
async def test_bound_resource_errors_preserve_details_without_retries(
    api: Harness, status: int
) -> None:
    model = await api.model()
    api.handler = lambda _: httpx.Response(
        status,
        headers={"x-request-id": "req-123"},
        json={
            "code": "problem",
            "detail": "Request rejected",
            "details": {"reason": "test"},
        },
    )
    for method, kwargs in [
        (model.examples.list, {}),
        (model.jobs.alignments.list, {}),
        (model.jobs.batch_decisions.create, {"scope": "all", "idempotency_key": "error-123"}),
    ]:
        with pytest.raises(ModaicAPIError) as caught:
            await call(method, **kwargs)
        assert caught.value.status_code == status
        assert caught.value.code == "problem"
        assert caught.value.request_id == "req-123"
        assert caught.value.details == {"reason": "test"}
    assert len(api.requests) == 4


async def test_all_answer_types_and_falsy_metadata(api: Harness) -> None:
    answers = {
        "refund": {"type": "noul", "noul": 0},
        "priority": {
            "type": "choice",
            "choice": "low",
            "probabilities": {"low": 1, "high": 0},
            "confidence": 1,
        },
        "quality": {
            "type": "score",
            "score": 0,
            "legend": {"0": "Poor"},
            "probabilities": {"0": 1},
            "confidence": 1,
        },
    }
    payload = {
        "model": "typesafe/jev-latest",
        "answers": answers,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "checkpoint": 0,
        "captured": False,
        "revision": "sha",
        "example_id": EXAMPLE_ID,
        "decision_id": JOB_ID,
    }
    api.handler = lambda _: httpx.Response(200, json=payload)
    result = await call(
        api.client.decisions.create,
        model="typesafe/jev-latest",
        state=None,
        capture=False,
        questions=QUESTIONS,
    )
    assert result.model_dump() == payload
    assert json.loads(api.requests[0].content) == {
        "model": "typesafe/jev-latest",
        "state": None,
        "capture": False,
        "questions": QUESTIONS,
    }
    assert "idempotency-key" not in api.requests[0].headers


async def test_detached_models_do_not_serialize_client_or_resource_state(api: Harness) -> None:
    model = await api.model()
    serialized = model.model_dump_json()
    assert "test-key" not in serialized
    assert not {"decisions", "examples", "jobs"} & json.loads(serialized).keys()
    detached = type(model).model_validate_json(serialized)
    for resource in ["decisions", "examples", "jobs"]:
        with pytest.raises(RuntimeError, match="through a Modaic client"):
            getattr(detached, resource)
    assert len(api.requests) == 1


async def test_encoded_path_segments(api: Harness) -> None:
    api.handler = lambda _: httpx.Response(200, json=model_json())
    await call(api.client.models.get, workspace="team /?", model="model #/%")
    assert api.requests[0].url.raw_path == (
        b"/api/v1/entities/team%20%2F%3F/models/model%20%23%2F%25"
    )


@pytest.mark.parametrize("status,body", [(502, "Bad gateway"), (503, "")])
async def test_non_json_http_errors_preserve_status(api: Harness, status: int, body: str) -> None:
    api.handler = lambda _: httpx.Response(status, text=body, headers={"x-request-id": "req-1"})
    with pytest.raises(ModaicAPIError) as caught:
        await call(api.client.models.list)
    assert caught.value.status_code == status
    assert caught.value.request_id == "req-1"
    assert caught.value.body == (body or None)


@pytest.mark.parametrize(
    "failure,expected",
    [
        (httpx.ReadTimeout, ModaicTimeoutError),
        (httpx.ConnectError, ModaicConnectionError),
    ],
)
async def test_transport_failures(
    api: Harness, failure: type[httpx.HTTPError], expected: type[Exception]
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise failure("Simulated failure", request=request)

    api.handler = fail
    with pytest.raises(expected):
        await call(api.client.models.list)
    assert len(api.requests) == 1


@pytest.mark.parametrize("body", ["not json", '{"models": "wrong-shape"}'])
async def test_malformed_success_responses(api: Harness, body: str) -> None:
    api.handler = lambda _: httpx.Response(200, text=body)
    with pytest.raises(ModaicConnectionError):
        await call(api.client.models.list)


async def test_empty_delete_and_cancel_responses(api: Harness) -> None:
    for method in [
        api.client.models.delete,
        api.client.alignments.cancel,
        api.client.batch_decisions.cancel,
    ]:
        assert (
            await call(method, MODEL_ID if method == api.client.models.delete else JOB_ID) is None
        )
    assert len(api.requests) == 3
    assert all(r.method == "DELETE" for r in api.requests)
