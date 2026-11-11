from datetime import date, datetime, time
from typing import Literal, assert_type

from modaic import (
    AsyncModaic,
    Box,
    Choice,
    Date,
    Datetime,
    DecisionResponse,
    Modaic,
    Noul,
    Number,
    NumberAnswer,
    NumberCriteria,
    NumberQuestion,
    Question,
    Score,
    Signature,
    Span,
    SpanAnswer,
    Time,
    TokenUsage,
)
from modaic._resources import AsyncModelDecisions, ModelDecisions
from modaic.types import Answer, JsonValue


class Review(Signature):
    country: Literal["fr", "us"] | None = Question()
    count: int = Question()


client = Modaic(api_key="test")
result = client.decisions.create(model="mo-latest", state={}, questions=Review)
assert_type(result, Review)
assert_type(result.country, Literal["fr", "us"] | None)
assert_type(result.count, int)


async def check_async(client: AsyncModaic) -> None:
    result = await client.decisions.create(model="mo-latest", state={}, questions=Review)
    assert_type(result, Review)


def check_bound(client: ModelDecisions) -> None:
    assert_type(client.create(state={}, questions=Review), Review)


async def check_async_bound(client: AsyncModelDecisions) -> None:
    assert_type(await client.create(state={}, questions=Review), Review)


class Native(Signature):
    day: date = Question(format="YYYY-MM-DD")
    stamp: datetime | None = Question(format="YYYY-MM-DDTHH:MM:SS")
    clock: time = Question(format="HH:MM")
    summary: str = Question()
    quote: SpanAnswer = Question()


def native_types(client: Modaic) -> None:
    result = client.decisions.create(model="mo-2.0", state={}, questions=Native)
    assert_type(result.day, date)
    assert_type(result.stamp, datetime | None)
    assert_type(result.clock, time)
    assert_type(result.summary, str)
    assert_type(result.quote, SpanAnswer)
    assert_type(result.quote.span.start, int)


def unified_types(client: Modaic) -> None:
    result = client.decisions.create(
        model="mo-2.0", state="Paris", questions={"evidence": Span(instructions="Find")}
    )
    span = result.answers["evidence"]
    assert isinstance(span, Span)
    assert_type(span.span.text, str)
    assert_type(span.spans[0].confidence, float)
    assert_type(span.span.start, int)
    nullable = client.decisions.create(
        model="mo-2.0", state="Paris", questions={"evidence": Span(instructions="Find").nullable()}
    )
    assert_type(nullable, DecisionResponse)


class PrimitiveReview(Signature):
    country: Choice[Literal["fr", "us"]] | None = Question(criteria=["France", "United States"])
    relevant: Noul = Question()
    published: Date = Question()
    opening: Time = Question()
    event: Datetime = Question()


def primitive_forwarding(client: Modaic) -> None:
    result = client.decisions.create(model="mo-2.0", state={}, questions=PrimitiveReview)
    assert_type(result, PrimitiveReview)
    assert_type(result.country, Choice[Literal["fr", "us"]] | None)
    if result.country is not None:
        assert_type(result.country.choice, Literal["fr", "us"])
    assert_type(result.relevant.noul, float)
    assert_type(result.published, Date)
    assert_type(result.opening, Time)
    assert_type(result.event, Datetime)
    assert_type(result.published.datetime, str)
    bare: Choice = Choice(criteria={"fr": "France"})
    assert_type(bare.choice, str)


class ImageReview(Signature):
    region: Box = Question(instructions="Find the object")


def box_types(client: Modaic) -> None:
    result = client.decisions.create(model="mo-2.0", state={}, questions=ImageReview)
    assert_type(result.region, Box)
    assert_type(result.region.box.top, float)
    assert_type(result.region.confidence, float | None)


def signature_envelope_types(client: Modaic) -> None:
    result = client.decisions.create(model="mo-2.0", state={}, questions=PrimitiveReview)
    assert_type(result.model, str)
    assert_type(result.usage, TokenUsage)
    assert_type(result.answers["relevant"], Answer | None)
    assert_type(result.relevant, Noul)
    assert_type(result.relevant.noul, float)


def score_map_types(answer: Score) -> None:
    assert_type(answer.legend, dict[int, JsonValue])
    assert_type(answer.probabilities, dict[int, float])


class NumericReview(Signature):
    amount: Number = Question(instructions="How much?")


def number_contract_types(client: Modaic) -> None:
    criteria: NumberCriteria = {"min": "0", "step": "0.1"}
    question: NumberQuestion = {"type": "number", "instructions": "How much?", "criteria": criteria}
    client.decisions.create(model="mo-2.0", state={}, questions={"amount": question})
    result = client.decisions.create(model="mo-2.0", state={}, questions=NumericReview)
    assert_type(result.amount.number, float)
    assert_type(result.amount.confidence, float)
    assert_type(result.numbers, dict[str, NumberAnswer])
