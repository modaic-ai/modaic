import json
from typing import Any, Literal

import httpx
import pytest
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import Choice, DecisionResponse, Noul, NoulAnswer, Question, Signature


class Review(Signature):
    relevant: Noul = Question(instructions="Relevant?")
    country: Choice[Literal["fr", "us"]] | None = Question(criteria=["France", "United States"])


def body() -> dict[str, Any]:
    return {
        "model": "mo-2.0",
        "answers": {
            "relevant": {"type": "noul", "noul": 0.9},
            "country": {
                "type": "choice",
                "choice": "fr",
                "probabilities": {"fr": 1},
                "confidence": 1,
            },
            "__modaic_answerable_country": {"type": "noul", "noul": 0.9},
        },
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "captured": True,
        "example_id": "example",
        "decision_id": "decision",
        "revision": "sha",
        "checkpoint": 3,
        "thinking": {},
        "future_metadata": "preserved",
    }


@pytest.mark.parametrize("bound", [False, True])
async def test_signature_retains_envelope_and_adds_aliases(api: Harness, bound: bool) -> None:
    resource = (await api.model()).decisions if bound else api.client.decisions
    payload = body()
    api.handler = lambda _: httpx.Response(200, json=payload, headers={"x-request-id": "request"})
    result = await call(
        resource.create, state={}, questions=Review, **({} if bound else {"model": "mo-2.0"})
    )
    assert isinstance(result, Review) and isinstance(result, DecisionResponse)
    assert result.relevant is result.answers["relevant"] is result.nouls["relevant"]
    assert result.country is result.answers["country"] is result.choices["country"]
    assert result.relevant.noul == 0.9
    assert result.model == "mo-2.0" and result.usage.input_tokens == 10
    assert result.captured and result.example_id == "example" and result.decision_id == "decision"
    assert result.revision == "sha" and result.checkpoint == 3 and result.request_id == "request"
    assert result.thinking == {} and result.future_metadata == "preserved"
    assert set(json.loads(api.requests[-1].content)["questions"]) == {
        "relevant",
        "country",
        "__modaic_answerable_country",
    }
    dumped = result.model_dump()
    assert "request_id" not in dumped
    assert dumped["answers"]["relevant"] == dumped["relevant"]
    result.relevant.noul = 0.8
    assert result.answers["relevant"].noul == 0.8
    assert result.model_dump()["answers"]["relevant"]["noul"] == 0.8
    payload["answers"]["__modaic_answerable_country"]["noul"] = 0.1
    rejected = await call(
        resource.create, state={}, questions=Review, **({} if bound else {"model": "mo-2.0"})
    )
    assert rejected.country is None and rejected.answers["country"] is None
    assert "__modaic_answerable_country" not in rejected.answers


async def test_response_model_and_signature_have_same_access_paths(api: Harness) -> None:
    class TypedResponse(DecisionResponse):
        relevant: NoulAnswer

    api.handler = lambda _: httpx.Response(
        200,
        json={
            "model": "mo-2.0",
            "answers": {"relevant": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )

    class TypedSignature(Signature):
        relevant: Noul = Question(instructions="Relevant?")

    signature = await call(
        api.client.decisions.create, model="mo-2.0", state={}, questions=TypedSignature
    )
    explicit = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        questions={"relevant": Noul(instructions="Relevant?")},
        response_model=TypedResponse,
    )
    for result in (signature, explicit):
        assert result.relevant is result.answers["relevant"]
        assert result.relevant.noul == 0.9
        assert result.model == "mo-2.0" and result.usage.input_tokens == 1


async def test_native_signature_annotations_keep_values_and_envelope(api: Harness) -> None:
    class Legacy(Signature):
        relevant: bool = Question()

    api.handler = lambda _: httpx.Response(
        200,
        json={
            "model": "mo-2.0",
            "answers": {"relevant": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    result = await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Legacy)
    assert result.relevant is True
    assert isinstance(result.answers["relevant"], Noul)
    assert result.answers["relevant"].noul == 0.9
    assert result.usage.input_tokens == 1


async def test_signature_rejects_reserved_question_names(api: Harness) -> None:
    class Invalid(Signature):
        model: str = Question(instructions="Question named model")

    with pytest.raises(ValueError, match="conflicts with response metadata"):
        await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Invalid)
    assert not api.requests
