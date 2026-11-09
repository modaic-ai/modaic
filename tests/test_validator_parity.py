import json
from typing import Any

import httpx
import pytest
from pydantic import field_validator, model_validator
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import Box, Choice, ChoiceAnswer, DecisionResponse, Question, Score, Signature


def payload() -> dict[str, Any]:
    return {
        "model": "mo-2.0",
        "answers": {
            "country": {
                "type": "choice",
                "choice": "FR",
                "probabilities": {"FR": 1},
                "confidence": 1,
            }
        },
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.mark.parametrize("inline", [True, False])
async def test_before_validators_receive_wire_dicts(api: Harness, inline: bool) -> None:
    seen: list[str] = []

    class Review(DecisionResponse):
        country: ChoiceAnswer

        @model_validator(mode="before")
        @classmethod
        def check_model(cls, value: Any) -> Any:
            assert isinstance(value["answers"]["country"], dict)
            seen.append("model")
            return value

        @field_validator("answers", mode="before")
        @classmethod
        def check_answers(cls, value: Any) -> Any:
            assert isinstance(value["country"], dict)
            seen.append("answers")
            return value

        @field_validator("country", mode="before")
        @classmethod
        def normalize(cls, value: Any) -> Any:
            assert isinstance(value, dict)
            seen.append("country")
            return {**value, "choice": value["choice"].lower()}

    api.handler = lambda _: httpx.Response(200, json=payload())
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        response_model=Review,
        **({"questions": {"country": Choice(criteria={"FR": "France"})}} if inline else {}),
    )
    assert set(seen) == {"model", "answers", "country"}
    assert isinstance(result.country, Choice) and result.country.choice == "fr"
    assert result.country is result.answers["country"]


async def test_custom_answer_subclass_is_preserved(api: Harness) -> None:
    class CustomChoice(ChoiceAnswer):
        domain: str = "custom"

        @field_validator("choice")
        @classmethod
        def normalize(cls, value: str) -> str:
            return value.lower()

    class Review(DecisionResponse):
        country: CustomChoice

    api.handler = lambda _: httpx.Response(200, json=payload())
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        questions={"country": Choice(instructions="Country?", criteria={"FR": "France"})},
        response_model=Review,
    )
    assert type(result.country) is CustomChoice
    assert result.country.choice == "fr" and result.country.domain == "custom"
    assert result.country.name == "country" and result.country.instructions == "Country?"
    assert result.country is result.answers["country"]


async def test_public_primitive_response_model_and_signature_before_validators(
    api: Harness,
) -> None:
    class Review(DecisionResponse):
        country: Choice

        @field_validator("country", mode="before")
        @classmethod
        def normalize(cls, value: Any) -> Any:
            assert isinstance(value, dict)
            return {**value, "choice": value["choice"].lower()}

    class Questions(Signature):
        country: Choice = Question(criteria={"FR": "France"})

        @field_validator("country", mode="before")
        @classmethod
        def normalize(cls, value: Any) -> Any:
            assert isinstance(value, dict)
            return {**value, "choice": value["choice"].lower()}

    api.handler = lambda _: httpx.Response(200, json=payload())
    explicit = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        questions={"country": Choice(criteria={"FR": "France"})},
        response_model=Review,
    )
    signature = await call(
        api.client.decisions.create, model="mo-2.0", state={}, questions=Questions
    )
    assert explicit.country.choice == signature.country.choice == "fr"
    assert explicit.country is explicit.answers["country"]
    assert signature.country is signature.answers["country"]


async def test_integer_score_maps_and_json_roundtrip(api: Harness) -> None:
    raw = {
        "type": "score",
        "score": 0.7,
        "legend": {"0": "Low", "1": "High"},
        "probabilities": {"0": 0.3, "1": 0.7},
        "confidence": 0.8,
    }
    api.handler = lambda _: httpx.Response(
        200,
        json={
            "model": "mo-2.0",
            "answers": {"rating": raw},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        questions={"rating": Score(criteria=["Low", "High"])},
    )
    answer = result.answers["rating"]
    assert answer.legend == {0: "Low", 1: "High"}
    assert answer.probabilities == {0: 0.3, 1: 0.7}
    assert answer.model_dump()["legend"] == {0: "Low", 1: "High"}
    assert json.loads(answer.model_dump_json()) == raw
    assert result.model_dump(mode="json")["answers"]["rating"] == raw


async def test_box_null_confidence(api: Harness) -> None:
    raw = {
        "type": "box",
        "box": {"top": 0.1, "bottom": 0.8, "left": 0.2, "right": 0.9},
        "confidence": None,
    }
    api.handler = lambda _: httpx.Response(
        200,
        json={
            "model": "mo-2.0",
            "answers": {"region": raw},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    result = await call(api.client.decisions.create, model="acme/image", state={})
    assert type(result.answers["region"]) is Box
    assert result.answers["region"].confidence is None
    assert result.answers["region"].model_dump() == raw


async def test_after_model_validator_transformations_are_retained(api: Harness) -> None:
    class Review(DecisionResponse):
        country: ChoiceAnswer

        @model_validator(mode="after")
        def normalize(self) -> Any:
            self.country.choice = self.country.choice.lower()
            return self

    api.handler = lambda _: httpx.Response(200, json=payload())
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        questions={"country": Choice(criteria={"FR": "France"})},
        response_model=Review,
    )
    assert result.country.choice == "fr"
    assert result.country is result.answers["country"]
    assert result.model_dump()["answers"]["country"]["choice"] == "fr"
