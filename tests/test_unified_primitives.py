import json
from typing import Any

import httpx
import pytest
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import (
    Choice,
    ChoiceAnswer,
    Datetime,
    DecisionResponse,
    ModaicConnectionError,
    Noul,
    Number,
    Question,
    Score,
    Signature,
    Span,
    SpanAnswer,
    Text,
)
from modaic._fine_tuning import _questions
from modaic._resources import _questions_payload

CASES = [
    (Noul(instructions="Is this relevant?"), {"type": "noul", "noul": 0.9}, "noul", 0.9),
    (
        Choice(instructions="Country?", criteria={"fr": "France"}),
        {"type": "choice", "choice": "fr", "probabilities": {"fr": 1}, "confidence": 1},
        "choice",
        "fr",
    ),
    (
        Score(instructions="Severity?", criteria=["Low", "High"]),
        {
            "type": "score",
            "score": 0.7,
            "legend": {"0": "Low", "1": "High"},
            "probabilities": {"0": 0.3, "1": 0.7},
            "confidence": 0.7,
        },
        "score",
        0.7,
    ),
    (Text(instructions="Summarize"), {"type": "text", "text": "", "confidence": 1}, "text", ""),
    (
        Number(instructions="How many?"),
        {"type": "number", "value": 0, "probabilities": {"0": 1}, "variance": 0},
        "value",
        0,
    ),
    (
        Span(instructions="Find a city"),
        {"type": "span", "span": "Paris", "start": 0, "end": 5, "confidence": 1},
        "span",
        "Paris",
    ),
    (
        Datetime(instructions="When?", format="YYYY-MM-DD"),
        {
            "type": "datetime",
            "datetime": "2024-02-29",
            "probabilities": {"2024-02-29": 1},
            "confidence": 1,
        },
        "datetime",
        "2024-02-29",
    ),
]


def body(answers: dict[str, Any]) -> dict[str, Any]:
    return {"model": "mo-2.0", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}


@pytest.mark.parametrize("question,answer,attribute,expected", CASES)
@pytest.mark.parametrize("bound", [False, True])
async def test_dictionary_primitive(
    api: Harness, question: Any, answer: Any, attribute: str, expected: Any, bound: bool
) -> None:
    resource = (await api.model()).decisions if bound else api.client.decisions
    question = question.model_copy(update={"name": "evidence"})
    wire = question.to_question()
    api.handler = lambda _: httpx.Response(200, json=body({"evidence": answer}))
    result = await call(
        resource.create,
        state="Paris",
        questions={"evidence": question},
        **({} if bound else {"model": "mo-2.0"}),
    )
    result = result.answers["evidence"]
    assert type(result) is type(question)
    assert getattr(result, attribute) == expected
    assert result.name == "evidence" and result.instructions == question.instructions
    assert result.model_dump(mode="json") == answer
    assert json.loads(api.requests[-1].content)["questions"] == {"evidence": wire}
    assert _questions_payload({"evidence": result}) == {"evidence": wire}
    assert _questions({"evidence": result}) == {"evidence": wire}
    await call(
        resource.create,
        state="Paris",
        questions={"evidence": result},
        **({} if bound else {"model": "mo-2.0"}),
    )
    assert json.loads(api.requests[-1].content)["questions"] == {"evidence": wire}
    with pytest.raises(AttributeError):
        getattr(question, attribute)


async def test_dictionary_answer_compatibility(api: Harness) -> None:
    class Typed(DecisionResponse):
        country: ChoiceAnswer
        quote: SpanAnswer

    answers = {"country": CASES[1][1], "quote": CASES[5][1]}
    api.handler = lambda _: httpx.Response(200, json=body(answers))
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state="Paris",
        questions={
            "country": CASES[1][0],
            "quote": CASES[5][0],
        },
        response_model=Typed,
    )
    assert type(result.country) is Choice and isinstance(result.country, ChoiceAnswer)
    assert type(result.quote) is Span and isinstance(result.quote, SpanAnswer)
    assert result.quote is result.answers["quote"]
    assert result.quote.name == "quote" and result.quote.instructions == "Find a city"
    assert result.model_dump()["answers"] == answers


async def test_signature_primitive_annotations(api: Harness) -> None:
    class Document(Signature):
        country: Choice = Question(instructions="Country?", criteria={"fr": "France"})
        severity: Score = Question(instructions="Severity?", criteria=["Low", "High"])
        quote: Span = Question(instructions="Find a city")
        summary: Text = Question(instructions="Summarize")

    api.handler = lambda _: httpx.Response(
        200,
        json=body(
            {
                "country": CASES[1][1],
                "severity": CASES[2][1],
                "quote": CASES[5][1],
                "summary": CASES[3][1],
            }
        ),
    )
    result = await call(
        api.client.decisions.create, model="mo-2.0", state="Paris", questions=Document
    )
    assert type(result.country) is Choice and result.country.choice == "fr"
    assert type(result.severity) is Score and result.severity.score == 0.7
    assert type(result.quote) is Span and result.quote.start == 0 and result.quote.end == 5
    assert result.quote.instructions == "Find a city" and result.quote.name == "quote"
    assert result.summary.text == ""


async def test_dictionary_nullable_and_inferred_empty_name(api: Harness) -> None:
    answers = {"": CASES[5][1], "__modaic_answerable_": {"type": "noul", "noul": 0.1}}
    api.handler = lambda _: httpx.Response(200, json=body(answers))
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state="Paris",
        questions={"": Span(name="ignored", instructions="Find").nullable()},
    )
    assert result.answers[""] is None
    answers["__modaic_answerable_"]["noul"] = 0.9
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state="Paris",
        questions={"": Span(name="ignored", instructions="Find").nullable()},
    )
    assert type(result.answers[""]) is Span and result.answers[""].name == ""


async def test_saved_model_answers_are_unified(api: Harness) -> None:
    api.handler = lambda _: httpx.Response(200, json=body({"evidence": CASES[5][1]}))
    result = await call(api.client.decisions.create, model="acme/model", state="Paris")
    assert type(result.answers["evidence"]) is Span
    assert result.answers["evidence"].name == "evidence"
    assert result.answers["evidence"].instructions is None
    with pytest.raises(ValueError, match="unavailable"):
        _questions_payload({"evidence": result.answers["evidence"]})


async def test_wrong_dictionary_answer_type(api: Harness) -> None:
    api.handler = lambda _: httpx.Response(200, json=body({"answer": CASES[0][1]}))
    with pytest.raises(ModaicConnectionError):
        await call(
            api.client.decisions.create,
            model="mo-2.0",
            state="Paris",
            questions={"answer": Span(instructions="Find")},
        )


async def test_signature_answer_annotations_return_public_classes(api: Harness) -> None:
    class Document(Signature):
        country: ChoiceAnswer = Question(criteria={"fr": "France"})
        quote: SpanAnswer = Question(instructions="Find")

    api.handler = lambda _: httpx.Response(
        200, json=body({"country": CASES[1][1], "quote": CASES[5][1]})
    )
    result = await call(
        api.client.decisions.create, model="mo-2.0", state="Paris", questions=Document
    )
    assert type(result.country) is Choice and isinstance(result.country, ChoiceAnswer)
    assert type(result.quote) is Span and isinstance(result.quote, SpanAnswer)


@pytest.mark.parametrize(
    "questions", [Span(instructions="Find"), Span, Span(instructions="Find").nullable()]
)
async def test_reject_direct_questions(api: Harness, questions: Any) -> None:
    with pytest.raises(TypeError, match="dictionary or Signature"):
        await call(api.client.decisions.create, model="mo-2.0", state="Paris", questions=questions)
    assert not api.requests


async def test_dictionary_key_overrides_optional_name(api: Harness) -> None:
    api.handler = lambda _: httpx.Response(200, json=body({"evidence": CASES[5][1]}))
    question = Span(name="other", instructions="Find")
    result = await call(
        api.client.decisions.create, model="mo-2.0", state="Paris", questions={"evidence": question}
    )
    assert result.answers["evidence"].name == "evidence"
    assert question.name == "other"
