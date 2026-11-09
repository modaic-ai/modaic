from collections.abc import Callable
from datetime import date
from typing import Any, Literal

import httpx
import pytest
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import (
    Choice,
    ChoiceAnswer,
    DecisionResponse,
    ModaicConnectionError,
    Noul,
    Nullable,
    Question,
    Signature,
)


class Review(Signature):
    country: Literal["fr", "us"] | None = Question(
        instructions="Which country?", criteria=["France", "United States"]
    )
    relevant: bool = Question(instructions="Relevant?")
    count: int = Question(instructions="How many?", min=0)
    price: float = Question(instructions="Price?", step=0.01)
    published: date = Question(instructions="Publication date?", format="YYYY-MM-DD")
    summary: str = Question(instructions="Summary?", nullable=0.8)


def response() -> dict[str, Any]:
    return {
        "model": "mo-latest",
        "answers": {
            "country": {
                "type": "choice",
                "choice": "fr",
                "probabilities": {"fr": 1},
                "confidence": 1,
            },
            "__modaic_answerable_country": {"type": "noul", "noul": 0.5},
            "relevant": {"type": "noul", "noul": 0},
            "count": {
                "type": "number",
                "number": 0,
                "probabilities": {"0": 1},
                "variance": 0,
                "confidence": 0.91,
            },
            "price": {
                "type": "number",
                "number": 1.25,
                "probabilities": {"1.25": 1},
                "variance": 0,
                "confidence": 0.91,
            },
            "published": {
                "type": "datetime",
                "datetime": "2026-11-07",
                "probabilities": {"2026-11-07": 1},
                "confidence": 1,
            },
            "summary": {"type": "text", "text": "", "confidence": 1},
            "__modaic_answerable_summary": {"type": "noul", "noul": 0.7},
        },
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }


@pytest.mark.parametrize("bound", [False, True])
async def test_signature(api: Harness, bound: bool) -> None:
    import json

    resource = (await api.model()).decisions if bound else api.client.decisions
    api.handler = lambda _: httpx.Response(200, json=response())
    result = await call(
        resource.create, state=False, questions=Review, **({} if bound else {"model": "mo-latest"})
    )
    assert isinstance(result, Review)
    assert result.country == "fr"
    assert result.relevant is False
    assert result.count == 0 and type(result.count) is int
    assert result.price == 1.25 and type(result.price) is float
    assert result.published == date(2026, 11, 7)
    assert result.summary is None
    wire = json.loads(api.requests[-1].content)
    assert wire["state"] is False
    assert wire["questions"]["count"]["criteria"] == {"min": "0", "step": "1"}
    assert wire["questions"]["price"]["criteria"] == {"step": "0.01"}
    assert wire["questions"]["country"]["criteria"] == {"fr": "France", "us": "United States"}
    assert len(api.requests) == (2 if bound else 1)


@pytest.mark.parametrize("wrapper", [lambda q: q.nullable(), lambda q: q | None, Nullable])
async def test_dictionary_nullable(api: Harness, wrapper: Callable[[Choice], Nullable]) -> None:
    class Output(DecisionResponse):
        country: ChoiceAnswer | None

    body = response()
    body["answers"] = {
        k: v for k, v in body["answers"].items() if k in ("country", "__modaic_answerable_country")
    }
    body["answers"]["__modaic_answerable_country"]["noul"] = 0.49
    api.handler = lambda _: httpx.Response(200, json=body)
    result = await call(
        api.client.decisions.create,
        model="mo-latest",
        state=None,
        questions={"country": wrapper(Choice(criteria={"fr": "France"}))},
        response_model=Output,
    )
    assert result.country is None
    assert result.answers == {"country": None}
    assert result.choices == {}


@pytest.mark.parametrize(
    "gate", [None, {"type": "choice"}, {"type": "noul", "noul": True}, {"type": "noul", "noul": 2}]
)
async def test_invalid_gate(api: Harness, gate: Any) -> None:
    body = response()
    body["answers"]["__modaic_answerable_country"] = gate
    api.handler = lambda _: httpx.Response(200, json=body)
    with pytest.raises(ModaicConnectionError):
        await call(api.client.decisions.create, model="mo-latest", state={}, questions=Review)


@pytest.mark.parametrize("threshold", [-1, 1.01, float("nan"), float("inf"), True])
def test_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        Noul().nullable(threshold)
    with pytest.raises(ValueError):
        Question(nullable=threshold)


async def test_gate_collision(api: Harness) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        await call(
            api.client.decisions.create,
            model="mo-latest",
            state={},
            questions={"a": Noul().nullable(), "__modaic_answerable_a": Noul()},
        )
    assert api.requests == []


async def test_optional_custom_threshold_and_score(api: Harness) -> None:
    class Output(Signature):
        flag: bool | None = Question(nullable=0.9)
        severity: int = Question(criteria=["Low", "High"])

    body = {
        "model": "mo-latest",
        "answers": {
            "flag": {"type": "noul", "noul": 0.9},
            "__modaic_answerable_flag": {"type": "noul", "noul": 0.8},
            "severity": {
                "type": "score",
                "score": 0.7,
                "legend": {"0": "Low", "1": "High"},
                "probabilities": {"0": 0.3, "1": 0.7},
                "confidence": 0.7,
            },
        },
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    api.handler = lambda _: httpx.Response(200, json=body)
    result = await call(api.client.decisions.create, model="mo-latest", state={}, questions=Output)
    assert result.flag is None
    assert result.severity == 1


async def test_required_signature_and_invalid_response(api: Harness) -> None:
    class Output(Signature):
        count: int = Question(instructions="Count?")
        relevant: bool = Question(instructions=False)

    body: dict[str, Any] = {
        "model": "mo-latest",
        "answers": {
            "count": {
                "type": "number",
                "number": 2,
                "probabilities": {"2": 1},
                "variance": 0,
                "confidence": 0.91,
            },
            "relevant": {"type": "noul", "noul": 0.5},
        },
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    api.handler = lambda _: httpx.Response(200, json=body)
    result = await call(api.client.decisions.create, model="mo-latest", state=0, questions=Output)
    assert result.count == 2 and result.relevant is True
    body["answers"]["count"]["number"] = 2.5
    with pytest.raises(ModaicConnectionError):
        await call(api.client.decisions.create, model="mo-latest", state=0, questions=Output)
    body["answers"] = None
    with pytest.raises(ModaicConnectionError):
        await call(api.client.decisions.create, model="mo-latest", state=0, questions=Output)


@pytest.mark.parametrize(
    "threshold, probability, expected_null", [(0, 0, False), (1, 0.99, True), (1, 1, False)]
)
async def test_threshold_endpoints(
    api: Harness, threshold: float, probability: float, expected_null: bool
) -> None:
    body = {
        "model": "mo-latest",
        "answers": {
            "flag": {"type": "noul", "noul": 0},
            "__modaic_answerable_flag": {"type": "noul", "noul": probability},
        },
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    api.handler = lambda _: httpx.Response(200, json=body)
    result = await call(
        api.client.decisions.create,
        model="mo-latest",
        state={},
        questions={"flag": Noul().nullable(threshold)},
    )
    assert (result.answers["flag"] is None) == expected_null
    if not expected_null:
        assert result.answers["flag"].noul == 0
