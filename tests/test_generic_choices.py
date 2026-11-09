import json
from typing import Any, Literal

import httpx
import pytest
from pydantic import ValidationError
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import (
    Choice,
    ChoiceAnswer,
    Date,
    Datetime,
    ModaicConnectionError,
    Noul,
    Question,
    Signature,
    Time,
)


def choice_answer(value: str) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": value,
        "probabilities": {"fr": 0.9, "us": 0.1},
        "confidence": 0.8,
    }


def body(answers: dict[str, Any]) -> dict[str, Any]:
    return {"model": "mo-2.0", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}


@pytest.mark.parametrize(
    "criteria", [None, ["France", "United States"], {"us": "United States", "fr": "France"}]
)
async def test_generic_choice_signature(api: Harness, criteria: Any) -> None:
    class Review(Signature):
        country: Choice[Literal["fr", "us"]] | None = Question(
            instructions="Country?", criteria=criteria
        )
        relevant: Noul = Question(instructions="Relevant?")

    answers = {
        "country": choice_answer("fr"),
        "__modaic_answerable_country": {"type": "noul", "noul": 0.9},
        "relevant": {"type": "noul", "noul": 0.8},
    }
    api.handler = lambda _: httpx.Response(200, json=body(answers))
    result = await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Review)
    assert type(result.country) is Choice[Literal["fr", "us"]]
    assert isinstance(result.country, ChoiceAnswer)
    assert result.country.choice == "fr"
    assert result.country.model_dump(mode="json") == choice_answer("fr")
    assert result.country.name == "country" and result.country.instructions == "Country?"
    assert type(result.relevant) is Noul and result.relevant.noul == 0.8
    wire = json.loads(api.requests[-1].content)
    assert set(wire["questions"]["country"]["criteria"]) == {"fr", "us"}
    if criteria is not None:
        assert wire["questions"]["country"]["criteria"] == {"fr": "France", "us": "United States"}
    answers["__modaic_answerable_country"]["noul"] = 0.1
    result = await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Review)
    assert result.country is None


async def test_generic_choice_dictionary_retains_specialization(api: Harness) -> None:
    api.handler = lambda _: httpx.Response(200, json=body({"country": choice_answer("fr")}))
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        questions={"country": Choice[Literal["fr", "us"]](instructions="Country?")},
    )
    assert type(result.answers["country"]) is Choice[Literal["fr", "us"]]
    assert result.answers["country"].choice == "fr"


@pytest.mark.parametrize(
    "criteria", [["France"], {"fr": "France"}, {"fr": "France", "de": "Germany"}]
)
async def test_generic_choice_invalid_criteria(api: Harness, criteria: Any) -> None:
    class Review(Signature):
        country: Choice[Literal["fr", "us"]] = Question(criteria=criteria)

    with pytest.raises(ValidationError):
        await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Review)
    assert not api.requests


async def test_generic_choice_rejects_unknown_response_value(api: Harness) -> None:
    class Review(Signature):
        country: Choice[Literal["fr", "us"]] = Question()

    api.handler = lambda _: httpx.Response(200, json=body({"country": choice_answer("de")}))
    with pytest.raises(ModaicConnectionError):
        await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Review)
    with pytest.raises(ModaicConnectionError):
        await call(
            api.client.decisions.create,
            model="mo-2.0",
            state={},
            questions={"country": Choice[Literal["fr", "us"]]()},
        )


@pytest.mark.parametrize(
    "primitive,pattern,value",
    [
        (Date, "YYYY-MM-DD", "2024-02-29"),
        (Time, "HH:MM", "20:05"),
        (Datetime, "YYYY-MM-DDTHH:MM", "2024-02-29T20:05"),
    ],
)
@pytest.mark.parametrize("signature", [False, True])
async def test_datetime_public_subclasses(
    api: Harness, primitive: Any, pattern: str, value: str, signature: bool
) -> None:
    from pydantic import create_model

    questions = (
        create_model("Review", __base__=Signature, when=(primitive, Question(instructions="When?")))
        if signature
        else {"when": primitive(instructions="When?")}
    )
    api.handler = lambda _: httpx.Response(
        200,
        json=body(
            {
                "when": {
                    "type": "datetime",
                    "datetime": value,
                    "probabilities": {value: 1},
                    "confidence": 1,
                }
            }
        ),
    )
    result = await call(api.client.decisions.create, model="mo-2.0", state={}, questions=questions)
    answer = result.when if signature else result.answers["when"]
    assert type(answer) is primitive and isinstance(answer, Datetime)
    assert answer.datetime == value and answer.name == "when" and answer.instructions == "When?"
    wire = json.loads(api.requests[-1].content)
    assert wire["questions"]["when"]["criteria"] == {"format": pattern}


def test_date_time_explicit_precision_and_bounds() -> None:
    assert Date(instructions="Month?", format="YYYY-MM").to_question()["criteria"] == {
        "format": "YYYY-MM"
    }
    assert Time(instructions="Hour?", format="HH").to_question()["criteria"] == {"format": "HH"}
    assert Time(instructions="Exact time?", format="HH:MM:SS").to_question()["criteria"] == {
        "format": "HH:MM:SS"
    }
    with pytest.raises(ValidationError):
        Date(instructions="When?", format="HH:MM")
    with pytest.raises(ValidationError):
        Time(instructions="When?", format="YYYY-MM-DD")


def test_reordered_literals_preserve_description_order() -> None:
    forward = Choice[Literal["fr", "us"]](criteria=["France", "United States"])
    reverse = Choice[Literal["us", "fr"]](criteria=["United States", "France"])
    assert forward.to_question()["criteria"] == {"fr": "France", "us": "United States"}
    assert reverse.to_question()["criteria"] == {"us": "United States", "fr": "France"}
    assert type(forward).literal_keys() == ("fr", "us")
    assert type(reverse).literal_keys() == ("us", "fr")
