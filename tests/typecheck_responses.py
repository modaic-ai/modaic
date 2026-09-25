"""Static regression checks: run mypy on this file, never execute it."""

from typing import assert_type

from pydantic import BaseModel

from modaic import AsyncModaic, Choice, DecisionResponse, Modaic, Noul, NoulAnswer, Question, Score


class BillingResponse(DecisionResponse):
    billing: NoulAnswer


class StandaloneAnswers(BaseModel):
    billing: NoulAnswer


class StandaloneBilling(BaseModel):
    """A response model that does not extend DecisionResponse."""

    answers: StandaloneAnswers


def sync_responses(client: Modaic) -> None:
    questions: dict[str, Question] = {
        "billing": Noul(instructions="Billing?"),
        "category": Choice(criteria={"billing": "Payment"}),
        "severity": Score(criteria=["Routine", "Urgent"]),
    }
    result = client.decisions.create(
        model="typesafe/jev-latest",
        state="Charged twice",
        questions=questions,
        response_model=BillingResponse,
    )
    assert_type(result, BillingResponse)
    assert_type(result.billing, NoulAnswer)
    model = client.models.create(workspace="acme", slug="billing", questions=questions)
    assert_type(model.decisions.create(state=None, response_model=BillingResponse), BillingResponse)
    assert_type(client.decisions.create(model="typesafe/jev-latest", state=None), DecisionResponse)
    assert_type(model.decisions.create(state=None), DecisionResponse)
    standalone = client.decisions.create(
        model="typesafe/jev-latest", state=None, response_model=StandaloneBilling
    )
    assert_type(standalone, StandaloneBilling)
    assert_type(standalone.answers.billing, NoulAnswer)
    assert_type(
        model.decisions.create(state=None, response_model=StandaloneBilling), StandaloneBilling
    )


async def async_responses(client: AsyncModaic) -> None:
    result = await client.decisions.create(
        model="typesafe/jev-latest", state="Charged twice", response_model=BillingResponse
    )
    assert_type(result, BillingResponse)
    assert_type(result.billing, NoulAnswer)
    model = await client.models.get(workspace="acme", model="billing")
    assert_type(
        await model.decisions.create(state=None, response_model=BillingResponse), BillingResponse
    )
    assert_type(
        await client.decisions.create(model="typesafe/jev-latest", state=None), DecisionResponse
    )
    assert_type(await model.decisions.create(state=None), DecisionResponse)
