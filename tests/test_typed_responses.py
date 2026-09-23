"""Question objects and named Pydantic answers share the existing wire contract."""

import json
from copy import deepcopy

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import (
    Choice,
    ChoiceAnswer,
    DecisionResponse,
    ModaicConnectionError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)


class BillingResponse(DecisionResponse):
    billing: NoulAnswer
    category: ChoiceAnswer
    severity: ScoreAnswer


QUESTIONS = {
    "billing": Noul(instructions="Is this about billing?"),
    "category": Choice(criteria={"billing": "Payments", "other": None}),
    "severity": Score(criteria=["Routine", "Urgent"]),
    "legacy": {"type": "noul"},
}
WIRE = {
    "billing": {"type": "noul", "instructions": "Is this about billing?"},
    "category": {"type": "choice", "criteria": {"billing": "Payments", "other": None}},
    "severity": {"type": "score", "criteria": ["Routine", "Urgent"]},
    "legacy": {"type": "noul"},
}
BODY = {
    "model": "typesafe/jev-latest",
    "answers": {
        "billing": {"type": "noul", "noul": 0.95},
        "category": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "other": 0.1},
            "confidence": 0.9,
        },
        "severity": {
            "type": "score",
            "score": 0.7,
            "legend": {"0": "Routine", "1": "Urgent"},
            "probabilities": {"0": 0.3, "1": 0.7},
            "confidence": 0.7,
        },
        "legacy": {"type": "noul", "noul": 0.2},
    },
    "usage": {"input_tokens": 20, "output_tokens": 10},
    "captured": True,
    "example_id": "example-1",
    "decision_id": "decision-1",
    "checkpoint": 2,
    "revision": "abc123",
    "future_metadata": True,
}


@pytest.mark.parametrize("bound", [False, True])
async def test_typed_response(api: Harness, bound: bool) -> None:
    resource = (await api.model()).decisions if bound else api.client.decisions
    api.handler = lambda _: httpx.Response(200, json=BODY, headers={"x-request-id": "req-123"})
    result = await call(
        resource.create,
        state="Charged twice",
        questions=QUESTIONS,
        response_model=BillingResponse,
        **({} if bound else {"model": "base"}),
    )
    assert isinstance(result, BillingResponse)
    assert result.billing == result.nouls["billing"] == result.answers["billing"]
    assert result.category == result.choices["category"]
    assert result.severity == result.scores["severity"]
    assert set(result.nouls) == {"billing", "legacy"}
    assert result.request_id == "req-123"
    assert result.usage.input_tokens == 20
    assert result.captured and result.example_id == "example-1"
    assert result.decision_id == "decision-1" and result.checkpoint == 2
    assert result.revision == "abc123" and result.future_metadata is True
    assert "request_id" not in result.model_dump()
    body = json.loads(api.requests[-1].content)
    assert body["questions"] == WIRE
    assert "response_model" not in body
    assert api.requests[-1].url.path.endswith("/systemone")


async def test_question_objects_in_model_writes(api: Harness) -> None:
    model = await call(
        api.client.models.create, workspace="acme", slug="support", questions=QUESTIONS
    )
    assert json.loads(api.requests[-1].content)["questions"] == WIRE
    await call(api.client.models.update, model.id, questions=QUESTIONS)
    assert json.loads(api.requests[-1].content)["questions"] == WIRE


async def test_omission_null_and_json_instructions(api: Harness) -> None:
    await call(
        api.client.decisions.create,
        state=None,
        model="base",
        questions={
            "empty": Noul(),
            "null": Noul(instructions=None, criteria=None),
            "json": Noul(instructions={"rules": ["one", "two"]}),
        },
    )
    assert json.loads(api.requests[-1].content)["questions"] == {
        "empty": {"type": "noul"},
        "null": {"type": "noul", "instructions": None, "criteria": None},
        "json": {"type": "noul", "instructions": {"rules": ["one", "two"]}},
    }


@pytest.mark.parametrize("invalid", ["missing", "wrong_type"])
async def test_typed_answer_validation(api: Harness, invalid: str) -> None:
    body = deepcopy(BODY)
    if invalid == "missing":
        del body["answers"]["billing"]
    else:
        body["answers"]["billing"] = body["answers"]["category"]
    api.handler = lambda _: httpx.Response(200, json=body)
    with pytest.raises(ModaicConnectionError, match="response"):
        await call(
            api.client.decisions.create, state=None, model="base", response_model=BillingResponse
        )


@pytest.mark.parametrize("invalid", [BaseModel, {}, None])
async def test_invalid_response_model_rejected_before_http(api: Harness, invalid: object) -> None:
    with pytest.raises(TypeError, match="DecisionResponse subclass"):
        await call(api.client.decisions.create, state=None, model="base", response_model=invalid)
    assert not api.requests


def test_alias_and_optional_answer() -> None:
    class Response(DecisionResponse):
        is_billing: NoulAnswer = Field(alias="billing")
        absent: NoulAnswer | None = None

    response = Response.model_validate(BODY)
    assert response.is_billing == response.nouls["billing"]
    assert response.absent is None
    assert response.request_id is None


def test_question_validation() -> None:
    with pytest.raises(ValidationError):
        Choice(criteria={})
    with pytest.raises(ValidationError):
        Score(criteria=[])
    with pytest.raises(ValidationError):
        Choice()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Noul(type="score")  # type: ignore[arg-type]


def test_validation_alias_and_missing_answer() -> None:
    class Response(DecisionResponse):
        is_billing: NoulAnswer = Field(validation_alias="billing")

    assert (
        Response.model_validate(BODY).is_billing
        == DecisionResponse.model_validate(BODY).nouls["billing"]
    )
    body = deepcopy(BODY)
    del body["answers"]["billing"]
    body["billing"] = {"type": "noul", "noul": 0.9}
    with pytest.raises(ValidationError):
        Response.model_validate(body)


def test_answer_alias_cannot_override_metadata() -> None:
    class Response(DecisionResponse):
        model_answer: NoulAnswer = Field(alias="model")

    with pytest.raises(ValidationError, match="conflicts with response metadata"):
        Response.model_validate(BODY)


def test_empty_question_id_alias() -> None:
    class Response(DecisionResponse):
        empty: NoulAnswer = Field(alias="")

    body = {**BODY, "answers": {"": {"type": "noul", "noul": 0.25}}}
    result = Response.model_validate(body)
    assert result.empty == result.nouls[""]
