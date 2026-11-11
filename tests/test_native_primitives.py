import json
from datetime import date, datetime, time
from typing import Any

import httpx
import pytest
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import (
    Datetime,
    DatetimeAnswer,
    ModaicConnectionError,
    Question,
    Signature,
    Span,
    SpanAnswer,
    Text,
)

FORMATS = {
    "YYYY": "2024",
    "YYYY-MM": "2024-02",
    "YYYY-MM-DD": "2024-02-29",
    "HH": "20",
    "HH:MM": "20:05",
    "HH:MM:SS": "20:05:07",
    "YYYY-MM-DDTHH": "2024-02-29T20",
    "YYYY-MM-DDTHH:MM": "2024-02-29T20:05",
    "YYYY-MM-DDTHH:MM:SS": "2024-02-29T20:05:07",
}


def body(answers: dict[str, Any]) -> dict[str, Any]:
    return {"model": "mo-2.0", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}


def datetime_answer(value: str) -> dict[str, Any]:
    return {"type": "datetime", "datetime": value, "probabilities": {value: 1}, "confidence": 1}


@pytest.mark.parametrize("pattern,value", list(FORMATS.items()))
async def test_native_datetime_formats(api: Harness, pattern: Any, value: str) -> None:
    class Output(Signature):
        value: str = Question(instructions="When?", format=pattern)

    api.handler = lambda _: httpx.Response(200, json=body({"value": datetime_answer(value)}))
    result = await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Output)
    assert result.value == value
    wire = json.loads(api.requests[-1].content)
    assert wire["questions"]["value"] == {
        "type": "datetime",
        "instructions": "When?",
        "criteria": {"format": pattern},
    }


async def test_native_python_values_text_and_span(api: Harness) -> None:
    class Output(Signature):
        day: date = Question(format="YYYY-MM-DD")
        stamp: datetime | None = Question(format="YYYY-MM-DDTHH:MM:SS")
        clock: time = Question(format="HH:MM")
        summary: str = Question()
        quote: SpanAnswer = Question()
        quote_text: str = Question(type="span")

    span = {
        "type": "span",
        "span": {"text": "Paris", "start": 0, "end": 5, "confidence": 1},
        "spans": [{"text": "Paris", "start": 0, "end": 5, "confidence": 1}],
        "confidence": 1,
    }
    answers = {
        "day": datetime_answer("2024-02-29"),
        "stamp": datetime_answer("2024-02-29T20:05:07"),
        "__modaic_answerable_stamp": {"type": "noul", "noul": 0.9},
        "clock": datetime_answer("20:05"),
        "summary": {"type": "text", "text": "", "confidence": 1},
        "quote": span,
        "quote_text": span,
    }
    api.handler = lambda _: httpx.Response(200, json=body(answers))
    result = await call(
        api.client.decisions.create, model="mo-2.0", state="Paris", questions=Output
    )
    assert result.day == date(2024, 2, 29)
    assert result.stamp == datetime(2024, 2, 29, 20, 5, 7)
    assert result.stamp.tzinfo is None
    assert result.clock == time(20, 5)
    assert result.summary == ""
    assert (
        result.quote.span.start == 0
        and result.quote.span.end == 5
        and result.quote.span.text == "Paris"
    )
    assert result.quote_text == "Paris"
    answers["__modaic_answerable_stamp"]["noul"] = 0.1
    result = await call(
        api.client.decisions.create, model="mo-2.0", state="Paris", questions=Output
    )
    assert result.stamp is None


@pytest.mark.parametrize(
    "annotation,pattern,value,expected",
    [
        (date, "YYYY-MM-DD", "2024-02-29", date(2024, 2, 29)),
        (time, "HH:MM", "20:05", time(20, 5)),
        (datetime, "YYYY-MM-DDTHH:MM", "2024-02-29T20:05", datetime(2024, 2, 29, 20, 5)),
    ],
)
@pytest.mark.parametrize("nullable", [False, True])
async def test_native_format_defaults(
    api: Harness, annotation: Any, pattern: str, value: str, expected: Any, nullable: bool
) -> None:
    from pydantic import create_model

    output = create_model(
        "Output",
        __base__=Signature,
        value=(annotation | None if nullable else annotation, Question()),
    )
    answers = {"value": datetime_answer(value)}
    if nullable:
        answers["__modaic_answerable_value"] = {"type": "noul", "noul": 0.9}
    api.handler = lambda _: httpx.Response(200, json=body(answers))
    result = await call(api.client.decisions.create, model="mo-2.0", state={}, questions=output)
    assert result.value == expected and type(result.value) is type(expected)
    wire = json.loads(api.requests[-1].content)
    assert wire["questions"]["value"]["criteria"] == {"format": pattern}
    if nullable:
        answers["__modaic_answerable_value"]["noul"] = 0.1
        result = await call(api.client.decisions.create, model="mo-2.0", state={}, questions=output)
        assert result.value is None


@pytest.mark.parametrize(
    "pattern,value",
    [
        ("YYYY-MM-DD", "2023-02-29"),
        ("HH:MM", "24:00"),
        ("YYYY-MM-DDTHH:MM:SS", "2024-02-29T20:05:07Z"),
        ("YYYY-MM", "2024-13"),
    ],
)
async def test_malformed_datetime(api: Harness, pattern: Any, value: str) -> None:
    class Output(Signature):
        value: str = Question(format=pattern)

    api.handler = lambda _: httpx.Response(200, json=body({"value": datetime_answer(value)}))
    with pytest.raises(ModaicConnectionError):
        await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Output)


async def test_dictionary_native_constructors(api: Harness) -> None:
    api.handler = lambda _: httpx.Response(
        200,
        json=body(
            {
                "when": datetime_answer("2024-02-29"),
                    "quote": {
                        "type": "span",
                        "span": {"text": "Paris", "start": 0, "end": 5, "confidence": 1},
                        "spans": [
                            {"text": "Paris", "start": 0, "end": 5, "confidence": 1}
                        ],
                        "confidence": 1,
                    },
                "summary": {"type": "text", "text": "France", "confidence": 1},
            }
        ),
    )
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state="Paris",
        questions={
            "when": Datetime(instructions="When?", format="YYYY-MM-DD"),
            "quote": Span(instructions="Find the city"),
            "summary": Text(instructions="Summarize"),
        },
    )
    assert isinstance(result.answers["when"], DatetimeAnswer)
    assert result.answers["quote"].span.text == "Paris"
    wire = json.loads(api.requests[-1].content)
    assert wire["questions"]["when"] == {
        "type": "datetime",
        "instructions": "When?",
        "criteria": {"format": "YYYY-MM-DD"},
    }
    assert wire["questions"]["quote"]["type"] == "span"


async def test_explicit_datetime_requires_format(api: Harness) -> None:
    class Output(Signature):
        value: str = Question(type="datetime")

    with pytest.raises(ValueError, match="requires format"):
        await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Output)
    assert not api.requests


@pytest.mark.parametrize("pattern", ["YYYY-MM-DDTHH:MM:SSZ", "%Y-%m-%d", "", "MM-DD-YYYY"])
async def test_invalid_format_rejected_locally(api: Harness, pattern: Any) -> None:
    class Output(Signature):
        value: datetime = Question(format=pattern)

    with pytest.raises(ValueError, match="requires format"):
        await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Output)
    assert not api.requests


async def test_native_default_formats_can_be_overridden(api: Harness) -> None:
    class Output(Signature):
        stamp: datetime = Question(format="YYYY-MM-DDTHH:MM:SS")
        clock: time = Question(format="HH:MM:SS")

    api.handler = lambda _: httpx.Response(
        200,
        json=body(
            {
                "stamp": datetime_answer("2024-02-29T20:05:07"),
                "clock": datetime_answer("20:05:07"),
            }
        ),
    )
    result = await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Output)
    assert result.stamp.second == result.clock.second == 7
    wire = json.loads(api.requests[-1].content)
    assert wire["questions"]["stamp"]["criteria"]["format"] == "YYYY-MM-DDTHH:MM:SS"
    assert wire["questions"]["clock"]["criteria"]["format"] == "HH:MM:SS"


async def test_native_formats_can_reduce_granularity(api: Harness) -> None:
    class Output(Signature):
        stamp: datetime = Question(format="YYYY-MM-DDTHH")
        clock: time = Question(format="HH")

    api.handler = lambda _: httpx.Response(
        200,
        json=body(
            {
                "stamp": datetime_answer("2024-02-29T20"),
                "clock": datetime_answer("20"),
            }
        ),
    )
    result = await call(api.client.decisions.create, model="mo-2.0", state={}, questions=Output)
    assert result.stamp == datetime(2024, 2, 29, 20)
    assert result.clock == time(20)
    wire = json.loads(api.requests[-1].content)
    assert wire["questions"]["stamp"]["criteria"]["format"] == "YYYY-MM-DDTHH"
    assert wire["questions"]["clock"]["criteria"]["format"] == "HH"
