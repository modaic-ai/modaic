import json
from typing import Any

import httpx
import pytest
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import Datetime, Number, Question, Signature, Text


@pytest.mark.parametrize("signature", [False, True])
@pytest.mark.parametrize("bound", [False, True])
async def test_number_contract_and_filtered_views(
    api: Harness, signature: bool, bound: bool
) -> None:
    class Review(Signature):
        amount: Number = Question(instructions="How much?", min=0, step=0.1)
        summary: Text = Question(instructions="Summarize")
        day: Datetime = Question(instructions="When?", format="YYYY-MM-DD")

    questions: Any = (
        Review
        if signature
        else {
            "amount": Number(instructions="How much?", criteria={"min": "0", "step": "0.1"}),
            "summary": Text(instructions="Summarize"),
            "day": Datetime(instructions="When?", format="YYYY-MM-DD"),
        }
    )
    payload = {
        "model": "mo-2.0",
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "answers": {
            "amount": {
                "type": "number",
                "number": 12.4,
                "confidence": 0.91,
                "probabilities": {"12.4": 1},
                "variance": 0.16,
            },
            "summary": {"type": "text", "text": "A parcel", "confidence": 0.8},
            "day": {
                "type": "datetime",
                "datetime": "2026-11-08",
                "probabilities": {"2026-11-08": 1},
                "confidence": 0.9,
            },
        },
    }
    resource = (await api.model()).decisions if bound else api.client.decisions
    api.handler = lambda _: httpx.Response(200, json=payload)
    result = await call(
        resource.create,
        state="A parcel costs 12.4",
        questions=questions,
        **({} if bound else {"model": "mo-2.0"}),
    )
    amount = result.numbers["amount"]
    assert isinstance(amount, Number)
    assert amount.number == 12.4 and amount.confidence == 0.91 and amount.variance == 0.16
    assert set(result.numbers) == {"amount"}
    assert set(result.texts) == {"summary"}
    assert set(result.datetimes) == {"day"}
    assert amount is result.answers["amount"]
    assert result.model_dump()["answers"] == payload["answers"]
    if signature:
        assert result.amount is amount
    assert json.loads(api.requests[-1].content)["questions"]["amount"]["criteria"]["step"] == "0.1"
    amount.number = 7
    assert amount.model_dump()["number"] == 7
    assert amount.to_question()["type"] == "number"
