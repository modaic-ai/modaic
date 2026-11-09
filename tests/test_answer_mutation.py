import json
from typing import Any

import httpx
import pytest
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import Box, Choice, ChoiceAnswer, DecisionResponse, Noul, Question, Signature


def wire_response(answers: dict[str, Any]) -> dict[str, Any]:
    return {"model": "mo-2.0", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}


async def test_mutated_choice_serializes_current_values(api: Harness) -> None:
    class Review(DecisionResponse):
        country: ChoiceAnswer

    raw = {
        "type": "choice",
        "choice": "fr",
        "probabilities": {"fr": 0.9, "us": 0.1},
        "confidence": 0.8,
        "future_extension": "kept",
    }
    api.handler = lambda _: httpx.Response(200, json=wire_response({"country": raw}))
    question = Choice(instructions="Country?", criteria={"fr": "France", "us": "United States"})
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        questions={"country": question},
        response_model=Review,
    )
    answer = result.choices["country"]
    assert answer is result.country is result.answers["country"]
    answer.choice = "us"
    answer.confidence = 0.6
    answer.probabilities["us"] = 0.8
    assert answer.model_dump()["choice"] == "us"
    assert answer.model_dump()["confidence"] == 0.6
    assert answer.model_dump()["probabilities"]["us"] == 0.8
    assert answer.model_dump()["future_extension"] == "kept"
    assert json.loads(answer.model_dump_json())["choice"] == "us"
    assert result.model_dump()["answers"]["country"]["choice"] == "us"
    assert json.loads(result.model_dump_json())["answers"]["country"]["choice"] == "us"
    assert answer.model_dump(include={"choice"}) == {"choice": "us"}
    assert "probabilities" not in answer.model_dump(exclude={"probabilities"})
    assert "'us'" in repr(answer)
    assert isinstance(answer, Choice)
    assert answer.to_question() == question.to_question()
    assert answer.model_copy(update={"choice": "fr"}).model_dump()["choice"] == "fr"
    assert answer.choice == "us"


def test_mutating_box_model_serializes_nested_current_fields() -> None:
    raw = {
        "type": "box",
        "box": {"top": 0.1, "bottom": 0.8, "left": 0.2, "right": 0.9},
        "confidence": 0.9,
    }
    answer = Box.from_answer(raw)
    answer.box.top = 0.3
    assert answer.model_dump()["box"]["top"] == 0.3
    answer.box = answer.box.model_copy(update={"top": 0.4})
    assert answer.model_dump()["box"]["top"] == 0.4


@pytest.mark.parametrize(
    "probability,expected", [(0, False), (0.49, False), (0.5, False), (0.50001, True), (1, True)]
)
@pytest.mark.parametrize("signature", [False, True])
async def test_noul_truthiness(
    api: Harness, probability: float, expected: bool, signature: bool
) -> None:
    class Review(Signature):
        relevant: Noul = Question(instructions="Relevant?")

    api.handler = lambda _: httpx.Response(
        200, json=wire_response({"relevant": {"type": "noul", "noul": probability}})
    )
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        questions=Review if signature else {"relevant": Noul(instructions="Relevant?")},
    )
    answer = result.relevant if signature else result.answers["relevant"]
    assert bool(answer) is expected
    assert ("yes" if answer else "no") == ("yes" if expected else "no")
    assert answer.noul == probability
    answer.noul = 0.9 if not expected else 0.1
    assert bool(answer) is not expected
    assert answer.model_dump()["noul"] == answer.noul


def test_unanswered_noul_has_no_truth_value() -> None:
    with pytest.raises(AttributeError):
        bool(Noul(instructions="Relevant?"))
