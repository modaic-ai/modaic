from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from modaic import (
    AsyncModaic,
    AsyncModelDecisions,
    AsyncModelExamples,
    AsyncModelJobs,
    CreatedModel,
    Modaic,
    ModaicAPIError,
    Model,
    ModelDecisions,
    ModelExamples,
    ModelJobs,
)

MODEL_ID = "018f5f04-a793-7b21-a1f6-bc6b56789012"
EXAMPLE_ID = "31fcc86c-7f49-4daa-a8b1-577db0c8fe76"
JOB_ID = "9e3d5d17-a8db-4e61-b3c6-c8834f8faf12"


def model_json() -> dict[str, Any]:
    return {
        "id": MODEL_ID,
        "workspace": {
            "id": "workspace-1",
            "kind": "user",
            "slug": "farouk1",
            "name": "Farouk",
            "description": None,
            "avatarUrl": None,
        },
        "slug": "support-triage",
        "description": "Support model",
        "defaultBranch": "main",
        "visibility": "private",
        "createdAt": "2026-09-22T00:00:00Z",
        "updatedAt": "2026-09-22T00:00:00Z",
    }


def example_json() -> dict[str, Any]:
    return {
        "id": EXAMPLE_ID,
        "state": {"ticket": "charged twice"},
        "source": "ingest",
        "imageUrls": [],
        "annotation": {
            "groundTruth": {"refund": True},
            "groundReasoning": "duplicate charge",
            "split": "train",
        },
        "latestDecision": None,
        "decisionCount": 0,
        "createdAt": "2026-09-22T00:00:00Z",
        "updatedAt": "2026-09-22T00:00:00Z",
    }


def batch_json(status: str = "completed") -> dict[str, Any]:
    return {
        "id": JOB_ID,
        "repositoryId": MODEL_ID,
        "status": status,
        "phase": "done" if status == "completed" else "queued",
        "branch": "main",
        "sourceCommitSha": "abc1234",
        "progress": {"total": 1, "completed": 1, "failed": 0},
        "result": {"completed": 1},
        "error": None,
        "createdAt": "2026-09-22T00:00:00Z",
        "updatedAt": "2026-09-22T00:00:00Z",
        "startedAt": "2026-09-22T00:00:00Z",
        "finishedAt": "2026-09-22T00:01:00Z",
    }


def alignment_json(status: str = "completed") -> dict[str, Any]:
    return {
        "id": JOB_ID,
        "repositoryId": MODEL_ID,
        "status": status,
        "phase": "done" if status == "completed" else "queued",
        "branch": "main",
        "sourceCommitSha": "abc1234",
        "resultCommitSha": "def5678",
        "result": {"improved": True},
        "progress": {"stage": "done", "metricCalls": 10, "maxMetricCalls": 10},
        "error": None,
        "createdAt": "2026-09-22T00:00:00Z",
        "updatedAt": "2026-09-22T00:00:00Z",
        "startedAt": "2026-09-22T00:00:00Z",
        "finishedAt": "2026-09-22T00:01:00Z",
    }


def response_for(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    method = request.method
    if path == "/api/v1/systemone":
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-latest",
                "answers": {"refund": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )
    if path == "/api/v1/models" and method == "GET":
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "farouk1/support-triage",
                        "description": "Support model",
                        "type": "repository",
                        "repository_id": MODEL_ID,
                    }
                ]
            },
        )
    if path == "/api/v1/models" and method == "POST":
        return httpx.Response(201, json={**model_json(), "workspace": "farouk1"})
    if path == "/api/v1/entities/farouk1/models/support-triage":
        return httpx.Response(200, json=model_json())
    if path == f"/api/v1/models/{MODEL_ID}" and method == "PATCH":
        return httpx.Response(200, json=model_json())
    if path == f"/api/v1/models/{MODEL_ID}" and method == "DELETE":
        return httpx.Response(204)
    if path == f"/api/v1/models/{MODEL_ID}/examples" and method == "POST":
        return httpx.Response(201, json={"examples": [example_json()]})
    if path == f"/api/v1/models/{MODEL_ID}/examples" and method == "GET":
        return httpx.Response(
            200,
            json={
                "items": [example_json()],
                "page": 1,
                "pageSize": 30,
                "total": 1,
                "totalPages": 1,
            },
        )
    if path == f"/api/v1/models/{MODEL_ID}/examples/{EXAMPLE_ID}/decisions":
        return httpx.Response(200, json={"decisions": []})
    if path == f"/api/v1/models/{MODEL_ID}/examples/{EXAMPLE_ID}":
        return httpx.Response(200, json=example_json())
    if path == f"/api/v1/models/{MODEL_ID}/examples/{EXAMPLE_ID}/annotation":
        return httpx.Response(200, json=example_json())
    if path == f"/api/v1/models/{MODEL_ID}/batch-decisions":
        if method == "GET":
            return httpx.Response(200, json={"batchDecisions": [batch_json()]})
        return httpx.Response(202, json=batch_json("queued"))
    if path == f"/api/v1/batch-decisions/{JOB_ID}" and method == "GET":
        return httpx.Response(200, json=batch_json())
    if path == f"/api/v1/batch-decisions/{JOB_ID}" and method == "DELETE":
        return httpx.Response(204)
    if path == f"/api/v1/models/{MODEL_ID}/alignments":
        if method == "GET":
            return httpx.Response(200, json={"alignments": [alignment_json()]})
        return httpx.Response(202, json=alignment_json("queued"))
    if path == f"/api/v1/alignments/{JOB_ID}" and method == "GET":
        return httpx.Response(200, json=alignment_json())
    if path == f"/api/v1/alignments/{JOB_ID}" and method == "DELETE":
        return httpx.Response(204)
    if path == f"/api/v1/alignments/{JOB_ID}/logs":
        return httpx.Response(200, json={"logs": ["done"], "available": True, "running": False})
    return httpx.Response(404, json={"code": "not_found", "detail": path})


def sync_client(
    handler: Callable[[httpx.Request], httpx.Response] = response_for,
) -> tuple[Modaic, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    http_client = httpx.Client(
        base_url="https://example.test/api/v1", transport=httpx.MockTransport(record)
    )
    return Modaic(api_key="test-key", http_client=http_client), requests


@pytest.mark.parametrize("model", [None, "typesafe/jev-latest"])
def test_create_model_with_questions(model: str | None) -> None:
    client, requests = sync_client()
    params: dict[str, Any] = {
        "workspace": "acme",
        "slug": "support-priority",
        "questions": {
            "priority": {
                "type": "choice",
                "instructions": {"goal": "Route support tickets"},
                "criteria": {"normal": "Routine issue", "high": "Urgent blocker"},
            }
        },
    }
    if model is not None:
        params["model"] = model
    with client:
        created = client.models.create(**params)
        assert created.id == MODEL_ID
        assert created.workspace == "farouk1"
        assert "owner" not in created.model_dump()
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/v1/models"
    assert json.loads(requests[0].content) == params


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, "typesafe/jev-latest"])
async def test_async_create_model_with_questions(model: str | None) -> None:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response_for(request)

    params: dict[str, Any] = {
        "workspace": "acme",
        "slug": "support-priority",
        "questions": {"refund": {"type": "noul", "instructions": {"goal": "Identify refunds"}}},
    }
    if model is not None:
        params["model"] = model
    async with AsyncModaic(
        api_key="test-key",
        http_client=httpx.AsyncClient(
            base_url="https://example.test/api/v1", transport=httpx.MockTransport(record)
        ),
    ) as client:
        created = await client.models.create(**params)
        assert created.id == MODEL_ID
        assert created.workspace == "farouk1"
        assert "owner" not in created.model_dump()
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/v1/models"
    assert json.loads(requests[0].content) == params


def test_update_sends_discard_alignment_only_when_given() -> None:
    questions = {"refund": {"type": "noul", "instructions": "Eligible for a refund?"}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=model_json())

    client, requests = sync_client(handler)
    with client:
        client.models.update(MODEL_ID, questions=questions)
        client.models.update(MODEL_ID, questions=questions, discard_alignment=True)
    assert "discardAlignment" not in json.loads(requests[0].content)
    assert json.loads(requests[1].content)["discardAlignment"] is True


def test_update_surfaces_the_alignment_guardrail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "type": "https://modaic.dev/problems/alignment-would-be-discarded",
                "title": "Conflict",
                "status": 409,
                "code": "alignment_would_be_discarded",
                "detail": "The questions on main were written by alignment (checkpoint 1).",
                "details": {"branch": "main", "commitSha": "abc", "checkpoint": 1},
            },
        )

    client, _ = sync_client(handler)
    with client, pytest.raises(ModaicAPIError) as raised:
        client.models.update(MODEL_ID, questions={"q": {"type": "noul", "instructions": "x"}})
    assert raised.value.status_code == 409
    assert raised.value.code == "alignment_would_be_discarded"


def test_update_reports_no_op_when_configuration_matches() -> None:
    questions = {"refund": {"type": "noul", "instructions": {"goal": "eligibility"}}}
    configuration = {"schemaVersion": 1, "checkpoint": 3, "questions": questions}
    commit = {"commitSha": "head", "previousSha": "head", "branch": "main"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                **model_json(),
                "configuration": configuration,
                "commit": commit,
                "unchanged": True,
            },
        )

    client, requests = sync_client(handler)
    with client:
        model = client.models.update(MODEL_ID, questions=questions)
    assert len(requests) == 1
    assert model.unchanged is True
    assert model.commit is not None
    assert model.commit.commit_sha == model.commit.previous_sha
    assert model.configuration is not None
    assert model.configuration.checkpoint == 3


@pytest.mark.parametrize("method", ["create", "get", "update"])
def test_bound_decisions(method: str) -> None:
    client, requests = sync_client()
    with client:
        model: (
            CreatedModel[ModelDecisions, ModelExamples, ModelJobs]
            | Model[ModelDecisions, ModelExamples, ModelJobs]
        )
        if method == "create":
            model = client.models.create(workspace="farouk1", slug="support-triage")
        elif method == "get":
            model = client.models.get(workspace="farouk1", model="support-triage")
        else:
            model = client.models.update(MODEL_ID, description="Updated")
        assert len(requests) == 1
        serialized = model.model_dump_json()
        assert "decisions" not in json.loads(serialized)
        assert "test-key" not in serialized
        assert "test-key" not in repr(model)
        detached = type(model).model_validate_json(serialized)
        with pytest.raises(RuntimeError, match="through a Modaic client"):
            _ = detached.decisions
        result = model.decisions.create(
            state={"ticket": "charged twice"},
            questions={"refund": {"type": "noul", "instructions": {"goal": "eligibility"}}},
            revision="review",
            example_id=EXAMPLE_ID,
            capture=False,
            idempotency_key="bound-decision-123",
        )
        assert result.answers["refund"].type == "noul"
        assert len(requests) == 2
        request = requests[1]
        assert str(request.url) == "https://example.test/api/v1/systemone"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["idempotency-key"] == "bound-decision-123"
        assert json.loads(request.content) == {
            "state": {"ticket": "charged twice"},
            "model": "farouk1/support-triage",
            "questions": {"refund": {"type": "noul", "instructions": {"goal": "eligibility"}}},
            "revision": "review",
            "example_id": EXAMPLE_ID,
            "capture": False,
        }
        model.decisions.create(state={})
        assert json.loads(requests[2].content) == {"state": {}, "model": "farouk1/support-triage"}
        assert not {"examples", "jobs"} & model.model_dump().keys()
        inputs = [{"state": {}, "annotation": {"ground_truth": {"refund": True}}}]
        assert model.examples.ingest(examples=inputs).examples[0].id == EXAMPLE_ID
        model.examples.list(page=2, page_size=5)
        assert model.examples.get(EXAMPLE_ID).id == EXAMPLE_ID
        model.examples.annotate(EXAMPLE_ID, ground_reasoning="Duplicate charge")
        assert model.examples.list_decisions(EXAMPLE_ID).decisions == []
        model.jobs.batch_decisions.create(
            example_ids=[EXAMPLE_ID],
            branch="review",
            source_commit_sha="abc1234",
            idempotency_key="bound-batch-123",
        )
        model.jobs.batch_decisions.list(limit=7)
        model.jobs.alignments.create(
            branch="review",
            source_commit_sha="abc1234",
            max_metric_calls=10,
            idempotency_key="bound-align-123",
            reflection_model="reflect",
            reflection_minibatch_size=2,
            seed=42,
        )
        model.jobs.alignments.list(limit=8)
        assert len(requests) == 12
        assert_bound_examples_jobs_wire(requests[3:])
        with pytest.raises(ValueError, match="exactly one"):
            model.jobs.batch_decisions.create(idempotency_key="invalid-input")
        assert len(requests) == 12


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["create", "get", "update"])
async def test_async_bound_decisions(method: str) -> None:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response_for(request)

    async with AsyncModaic(
        api_key="test-key",
        http_client=httpx.AsyncClient(
            base_url="https://example.test/api/v1", transport=httpx.MockTransport(record)
        ),
    ) as client:
        model: (
            CreatedModel[AsyncModelDecisions, AsyncModelExamples, AsyncModelJobs]
            | Model[AsyncModelDecisions, AsyncModelExamples, AsyncModelJobs]
        )
        if method == "create":
            model = await client.models.create(workspace="farouk1", slug="support-triage")
        elif method == "get":
            model = await client.models.get(workspace="farouk1", model="support-triage")
        else:
            model = await client.models.update(MODEL_ID, description="Updated")
        assert len(requests) == 1
        serialized = model.model_dump_json()
        assert "decisions" not in json.loads(serialized)
        assert "test-key" not in serialized
        result = await model.decisions.create(
            state={"ticket": "charged twice"},
            questions={"refund": {"type": "noul"}},
            revision="review",
            example_id=EXAMPLE_ID,
            capture=False,
            idempotency_key="async-bound-123",
        )
        assert result.answers["refund"].type == "noul"
        assert len(requests) == 2
        request = requests[1]
        assert str(request.url) == "https://example.test/api/v1/systemone"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["idempotency-key"] == "async-bound-123"
        assert json.loads(request.content) == {
            "state": {"ticket": "charged twice"},
            "model": "farouk1/support-triage",
            "questions": {"refund": {"type": "noul"}},
            "revision": "review",
            "example_id": EXAMPLE_ID,
            "capture": False,
        }
        await model.decisions.create(state={})
        assert json.loads(requests[2].content) == {"state": {}, "model": "farouk1/support-triage"}
        assert not {"examples", "jobs"} & model.model_dump().keys()
        inputs = [{"state": {}, "annotation": {"ground_truth": {"refund": True}}}]
        assert (await model.examples.ingest(examples=inputs)).examples[0].id == EXAMPLE_ID
        await model.examples.list(page=2, page_size=5)
        assert (await model.examples.get(EXAMPLE_ID)).id == EXAMPLE_ID
        await model.examples.annotate(EXAMPLE_ID, ground_reasoning="Duplicate charge")
        assert (await model.examples.list_decisions(EXAMPLE_ID)).decisions == []
        await model.jobs.batch_decisions.create(
            example_ids=[EXAMPLE_ID],
            branch="review",
            source_commit_sha="abc1234",
            idempotency_key="bound-batch-123",
        )
        await model.jobs.batch_decisions.list(limit=7)
        await model.jobs.alignments.create(
            branch="review",
            source_commit_sha="abc1234",
            max_metric_calls=10,
            idempotency_key="bound-align-123",
            reflection_model="reflect",
            reflection_minibatch_size=2,
            seed=42,
        )
        await model.jobs.alignments.list(limit=8)
        assert len(requests) == 12
        assert_bound_examples_jobs_wire(requests[3:])
        with pytest.raises(ValueError, match="exactly one"):
            await model.jobs.batch_decisions.create(idempotency_key="invalid-input")
        assert len(requests) == 12


def assert_bound_examples_jobs_wire(requests: list[httpx.Request]) -> None:
    base = f"https://example.test/api/v1/models/{MODEL_ID}"
    expected = [
        ("POST", "/examples"),
        ("GET", "/examples?page=2&pageSize=5"),
        ("GET", f"/examples/{EXAMPLE_ID}"),
        ("PATCH", f"/examples/{EXAMPLE_ID}/annotation"),
        ("GET", f"/examples/{EXAMPLE_ID}/decisions"),
        ("POST", "/batch-decisions"),
        ("GET", "/batch-decisions?limit=7"),
        ("POST", "/alignments"),
        ("GET", "/alignments?limit=8"),
    ]
    assert [(r.method, str(r.url)) for r in requests] == [(m, base + p) for m, p in expected]
    assert all(r.headers["authorization"] == "Bearer test-key" for r in requests)
    assert json.loads(requests[0].content) == {
        "examples": [{"state": {}, "annotation": {"groundTruth": {"refund": True}}}],
    }
    assert json.loads(requests[3].content) == {"groundReasoning": "Duplicate charge"}
    assert requests[5].headers["idempotency-key"] == "bound-batch-123"
    assert json.loads(requests[5].content) == {
        "branch": "review",
        "sourceCommitSha": "abc1234",
        "exampleIds": [EXAMPLE_ID],
    }
    assert requests[7].headers["idempotency-key"] == "bound-align-123"
    assert json.loads(requests[7].content) == {
        "branch": "review",
        "sourceCommitSha": "abc1234",
        "budget": {"maxMetricCalls": 10},
        "reflection": {"model": "reflect", "minibatchSize": 2, "seed": 42},
    }


def test_bound_decisions_preserve_api_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/systemone"):
            return httpx.Response(429, json={"code": "rate_limited", "detail": "Slow down"})
        return response_for(request)

    client, _ = sync_client(handler)
    with client:
        model = client.models.get(workspace="farouk1", model="support-triage")
        with pytest.raises(ModaicAPIError) as error:
            model.decisions.create(state={})
        assert error.value.status_code == 429
        assert error.value.code == "rate_limited"


def test_sync_client_covers_the_documented_api_surface() -> None:
    client, requests = sync_client()

    result = client.decisions.create(
        state={"ticket": "charged twice"},
        model="typesafe/jev-latest",
        questions={"refund": {"type": "noul"}},
        example_id=EXAMPLE_ID,
        idempotency_key="decision-1",
    )
    assert result.answers["refund"].type == "noul"
    assert client.models.list().models[0].repository_id == MODEL_ID
    client.models.create(workspace="farouk1", slug="support-triage", default_branch="main")
    client.models.get(workspace="farouk1", model="support-triage")
    client.models.update(
        MODEL_ID, description=None, model="typesafe/jev-latest", message="configure"
    )
    client.models.delete(MODEL_ID)

    client.examples.ingest(
        MODEL_ID,
        examples=[
            {
                "state": {"ticket": "charged twice"},
                "annotation": {
                    "ground_truth": {"refund": True},
                    "ground_reasoning": "duplicate",
                },
            }
        ],
    )
    client.examples.list(MODEL_ID)
    client.examples.get(MODEL_ID, EXAMPLE_ID)
    client.examples.annotate(MODEL_ID, EXAMPLE_ID, ground_truth={"refund": True})
    client.examples.list_decisions(MODEL_ID, EXAMPLE_ID)

    client.batch_decisions.create(
        MODEL_ID,
        example_ids=[EXAMPLE_ID],
        idempotency_key="batch-123",
    )
    client.batch_decisions.list(MODEL_ID)
    client.batch_decisions.get(JOB_ID)
    client.batch_decisions.cancel(JOB_ID)
    assert client.batch_decisions.wait(JOB_ID, poll_interval=0).status == "completed"

    client.alignments.create(
        MODEL_ID,
        branch="main",
        source_commit_sha="abc1234",
        max_metric_calls=100,
        idempotency_key="alignment-123",
    )
    client.alignments.list(MODEL_ID)
    client.alignments.get(JOB_ID)
    client.alignments.logs(JOB_ID)
    client.alignments.cancel(JOB_ID)
    assert client.alignments.wait(JOB_ID, poll_interval=0).status == "completed"

    assert all(request.headers["authorization"] == "Bearer test-key" for request in requests)
    decision_request = requests[0]
    assert json.loads(decision_request.content)["example_id"] == EXAMPLE_ID
    annotation_request = next(
        request for request in requests if request.url.path.endswith("/annotation")
    )
    assert json.loads(annotation_request.content) == {"groundTruth": {"refund": True}}
    assert {request.url.path for request in requests} >= {
        "/api/v1/systemone",
        "/api/v1/models",
        f"/api/v1/models/{MODEL_ID}/examples",
        f"/api/v1/models/{MODEL_ID}/batch-decisions",
        f"/api/v1/models/{MODEL_ID}/alignments",
    }


@pytest.mark.asyncio
async def test_async_client_uses_the_same_contract() -> None:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response_for(request)

    http_client = httpx.AsyncClient(
        base_url="https://example.test/api/v1", transport=httpx.MockTransport(record)
    )
    client = AsyncModaic(api_key="test-key", http_client=http_client)
    decision = await client.decisions.create(
        state={}, model="typesafe/jev-latest", questions={"ok": {"type": "noul"}}
    )
    models = await client.models.list()
    example = await client.examples.get(MODEL_ID, EXAMPLE_ID)
    batch = await client.batch_decisions.wait(JOB_ID, poll_interval=0)
    alignment = await client.alignments.wait(JOB_ID, poll_interval=0)
    await client.close()

    assert decision.model == "typesafe/jev-latest"
    assert models.models[0].repository_id == MODEL_ID
    assert example.id == EXAMPLE_ID
    assert batch.status == alignment.status == "completed"
    assert len(requests) == 5


def test_api_errors_preserve_problem_details() -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            headers={"x-request-id": "request-1"},
            json={
                "code": "validation_error",
                "detail": "Bad question.",
                "details": [{"path": ["questions"]}],
            },
        )

    client, _ = sync_client(fail)
    with pytest.raises(ModaicAPIError) as caught:
        client.models.list()
    assert caught.value.status_code == 422
    assert caught.value.code == "validation_error"
    assert caught.value.request_id == "request-1"
    assert str(caught.value) == "Bad question."


def test_batch_selection_is_exactly_one_mode() -> None:
    client, _ = sync_client()
    with pytest.raises(ValueError, match="exactly one"):
        client.batch_decisions.create(MODEL_ID, idempotency_key="batch-123")
    with pytest.raises(ValueError, match="exactly one"):
        client.batch_decisions.create(
            MODEL_ID,
            idempotency_key="batch-123",
            example_ids=[EXAMPLE_ID],
            scope="all",
        )
