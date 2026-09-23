"""Opt-in live tests. Create and retain test models/examples for inspection.

Set MODAIC_E2E=1, MODAIC_API_KEY, MODAIC_API_URL, and MODAIC_E2E_WORKSPACE.
These tests make billable inference requests; they never run in ordinary CI.
"""

import asyncio
import inspect
import os
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest
from pydantic import Field

from modaic import (
    AsyncModaic,
    Choice,
    ChoiceAnswer,
    DecisionResponse,
    Modaic,
    ModaicAPIError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)

pytestmark = pytest.mark.skipif(os.getenv("MODAIC_E2E") != "1", reason="Live API opt-in required")

QUESTIONS = {
    "urgent": Noul(instructions="Does the ticket need urgent attention?"),
    "category": Choice(
        instructions="Choose the ticket category.",
        criteria={"billing": "Payment or refund issue", "other": "Anything else"},
    ),
    "severity": Score(
        instructions="Rate the severity of the ticket.",
        criteria=["Informational", "Routine issue", "Urgent financial impact"],
    ),
}


class TicketResponse(DecisionResponse):
    urgent: NoulAnswer
    category: ChoiceAnswer
    severity: ScoreAnswer


class EdgeTicketResponse(TicketResponse):
    unnamed: NoulAnswer = Field(alias="")


async def call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    result = fn(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


async def eventually(fn: Callable[..., Any], predicate: Callable[[Any], bool]) -> Any:
    for _ in range(60):
        try:
            result = await call(fn)
            if predicate(result):
                return result
        except ModaicAPIError as error:
            if error.status_code != 404:
                raise
        await asyncio.sleep(0.5)
    raise AssertionError("Expected persisted state was not visible within 30 seconds")


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_live_model_lifecycle(mode: str) -> None:
    workspace = os.environ["MODAIC_E2E_WORKSPACE"]
    slug = f"e2e-systemone-py-{mode}-{uuid4().hex[:10]}"
    client = (Modaic if mode == "sync" else AsyncModaic)(
        api_key=os.environ["MODAIC_API_KEY"],
        base_url=os.environ["MODAIC_API_URL"],
        timeout=60,
    )
    try:
        created = await call(
            client.models.create,
            workspace=workspace,
            slug=slug,
            description="SDK end-to-end test; retained for inspection",
            model="typesafe/jev-latest",
            questions=QUESTIONS,
        )
        print(f"Created {workspace}/{slug} ({created.id})", flush=True)
        assert created.workspace == workspace
        model = await call(client.models.get, workspace=workspace, model=slug)
        assert model.id == created.id
        assert model.workspace.slug == workspace
        catalog = await call(client.models.list)
        assert any(item.repository_id == model.id for item in catalog.models)

        ids = [str(uuid4()), str(uuid4())]
        state = {"ticket": "I was charged twice. Please refund the extra payment urgently."}
        truth = {"urgent": False, "category": "other", "severity": 0}
        ingested = await call(
            created.examples.ingest,
            examples=[
                {"id": ids[0], "state": state},
                {
                    "id": ids[1],
                    "state": {"ticket": "Thanks for the information."},
                    "annotation": {
                        "ground_truth": truth,
                        "ground_reasoning": "No action requested.",
                    },
                },
            ],
        )
        assert {item.id for item in ingested.examples} == set(ids)
        labeled = await eventually(
            lambda: model.examples.get(ids[1]), lambda e: e.annotation is not None
        )
        assert labeled.annotation.ground_truth == truth
        assert labeled.annotation.ground_reasoning == "No action requested."

        truth = {"urgent": True, "category": "billing", "severity": 2}
        annotated = await call(
            model.examples.annotate,
            ids[0],
            ground_truth=truth,
            ground_reasoning="Duplicate charge requires prompt help.",
        )
        split = annotated.annotation.split
        await call(model.examples.annotate, ids[0], ground_reasoning="Confirmed duplicate charge.")
        stored = await eventually(
            lambda: model.examples.get(ids[0]),
            lambda e: e.annotation.ground_reasoning == "Confirmed duplicate charge.",
        )
        assert stored.annotation.ground_truth == truth
        assert stored.annotation.split == split

        key = f"e2e-decision-{uuid4()}"
        result = await call(
            created.decisions.create,
            state=state,
            example_id=ids[0],
            idempotency_key=key,
            response_model=TicketResponse,
        )
        assert isinstance(result, TicketResponse)
        assert result.urgent == result.nouls["urgent"]
        assert result.category == result.choices["category"]
        assert result.severity == result.scores["severity"]
        assert result.request_id
        assert result.captured is True
        assert result.example_id == ids[0]
        assert result.decision_id
        assert result.checkpoint is not None and result.revision
        assert {name: answer.type for name, answer in result.answers.items()} == {
            name: question.type for name, question in QUESTIONS.items()
        }
        history = await eventually(
            lambda: model.examples.list_decisions(ids[0]),
            lambda h: any(d.id == result.decision_id for d in h.decisions),
        )
        assert len(history.decisions) == 1
        assert history.decisions[0].source == "live"
        assert history.decisions[0].request["questions"] == {
            name: {"type": question.type, **question.model_dump(exclude_unset=True)}
            for name, question in QUESTIONS.items()
        }
        assert history.decisions[0].answers == result.model_dump()["answers"]
        replay = await call(
            created.decisions.create,
            state=state,
            example_id=ids[0],
            idempotency_key=key,
            response_model=TicketResponse,
        )
        assert replay.model_dump() == result.model_dump()
        assert len((await call(model.examples.list_decisions, ids[0])).decisions) == 1

        automatic = await call(
            model.decisions.create,
            state={"ticket": "Please explain my invoice."},
            idempotency_key=f"e2e-auto-{uuid4()}",
        )
        assert automatic.captured is True and automatic.example_id
        await eventually(
            lambda: model.examples.get(automatic.example_id),
            lambda e: e.source == "live" and e.decision_count == 1,
        )
        page = await eventually(lambda: model.examples.list(page_size=1), lambda p: p.total == 3)
        assert len(page.items) == 1 and page.total_pages == 3
        uncaptured_id = str(uuid4())
        uncaptured = await call(
            model.decisions.create,
            state={"ticket": "No storage test."},
            example_id=uncaptured_id,
            capture=False,
            idempotency_key=f"e2e-no-capture-{uuid4()}",
        )
        assert uncaptured.captured is False
        direct = await call(
            client.decisions.create,
            model="typesafe/jev-latest",
            state=state,
            questions=QUESTIONS,
            response_model=TicketResponse,
        )
        assert isinstance(direct, TicketResponse)
        assert 0 <= direct.urgent.noul <= 1
        assert direct.category == direct.choices["category"]
        assert direct.severity == direct.scores["severity"]
        with pytest.raises(ModaicAPIError) as missing:
            await call(model.examples.get, uncaptured_id)
        assert missing.value.status_code == 404
        assert (await call(model.examples.list)).total == 3
        updated = await call(
            client.models.update, model.id, description="Verified E2E model and examples"
        )
        assert updated.id == model.id
        assert (
            await call(client.models.get, workspace=workspace, model=slug)
        ).description == updated.description
        edge_questions = {
            "urgent": Noul(instructions=0),
            "category": Choice(instructions=False, criteria={"only": None}),
            "severity": Score(instructions=True, criteria=["Only"]),
            "": Noul(),
        }
        edge = await call(
            model.decisions.create,
            state=False,
            questions=edge_questions,
            response_model=EdgeTicketResponse,
        )
        assert edge.captured is True and edge.example_id
        assert edge.unnamed == edge.nouls[""]
        edge_example = await eventually(
            lambda: model.examples.get(edge.example_id), lambda e: e.latest_decision is not None
        )
        assert edge_example.state is False
        assert "" in edge_example.latest_decision.answers
        edge_history = await eventually(
            lambda: model.examples.list_decisions(edge.example_id), lambda h: len(h.decisions) == 1
        )
        assert edge_history.decisions[0].request["state"] is False
        assert edge_history.decisions[0].request["questions"] == {
            name: {"type": question.type, **question.model_dump(exclude_unset=True)}
            for name, question in edge_questions.items()
        }
        print(
            f"PASS {mode}: {workspace}/{slug}; "
            "4 stored examples, capture, annotations, history, replay, preserved edge inputs",
            flush=True,
        )
    finally:
        await call(client.close)
