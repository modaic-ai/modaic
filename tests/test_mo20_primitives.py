from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from modaic import (
    AsyncModaic,
    Box,
    BoxAnswer,
    Datetime,
    DatetimeAnswer,
    Modaic,
    Number,
    NumberAnswer,
    Span,
    SpanAnswer,
    Text,
    TextAnswer,
)


def handler(request: httpx.Request) -> httpx.Response:
    import json

    body = json.loads(request.content)
    assert body["model"] == "mo-2.0"
    assert body["assets"]["photo"]["data_base64"] == "test-bytes"
    assert body["questions"]["owner"] == {"type": "span", "instructions": "Who?"}
    assert body["questions"]["weight"] == {
        "type": "number",
        "instructions": "What is the weight?",
        "criteria": {"min": "0", "max": "25", "step": "0.1"},
    }
    assert body["questions"]["inspection"] == {
        "type": "datetime",
        "instructions": "When was it inspected?",
        "criteria": {"format": "YYYY-MM-DD"},
    }
    return httpx.Response(
        200,
        json={
            "model": "mo-2.0",
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "answers": {
                "owner": {
                    "type": "span",
                    "start": 3,
                    "end": 8,
                    "span": "Alice",
                    "confidence": 0.9,
                },
                "location": {
                    "type": "box",
                    "box": {"top": 0, "bottom": 1, "left": 0, "right": 1},
                    "confidence": 0.8,
                },
                "weight": {
                    "type": "number",
                    "number": 12.4,
                    "probabilities": {"12.4": 1},
                    "variance": 0.16,
                    "confidence": 0.91,
                },
                "inspection": {
                    "type": "datetime",
                    "datetime": "2024-02-29",
                    "probabilities": {"2024-02-29": 0.75, "2024-03-01": 0.25},
                    "confidence": 0.82,
                },
                "summary": {
                    "type": "text",
                    "text": "Alice owns the photo",
                    "probabilities": {
                        "Alice owns the photo": 0.75,
                        "The photo belongs to Alice": 0.25,
                    },
                    "confidence": 0.94,
                },
            },
        },
    )


def parameters() -> dict[str, Any]:
    return dict(
        model="mo-2.0",
        state="😀 Alice photo",
        assets={"photo": {"type": "image", "format": "png", "data_base64": "test-bytes"}},
        questions={
            "owner": Span(instructions="Who?"),
            "location": Box(instructions="Locate photo."),
            "weight": Number(
                instructions="What is the weight?",
                criteria={"min": "0", "max": "25", "step": "0.1"},
            ),
            "inspection": Datetime(
                instructions="When was it inspected?",
                criteria={"format": "YYYY-MM-DD"},
            ),
            "summary": Text(instructions="Summarize ownership."),
        },
    )


def test_sync_primitives() -> None:
    with Modaic(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ) as client:
        result = client.decisions.create(**parameters())
    assert isinstance(result.answers["owner"], SpanAnswer)
    assert isinstance(result.answers["location"], BoxAnswer)
    assert isinstance(result.answers["weight"], NumberAnswer)
    assert isinstance(result.answers["inspection"], DatetimeAnswer)
    assert isinstance(result.answers["summary"], TextAnswer)
    assert result.answers["owner"].start == 3
    assert result.answers["owner"].span == "Alice"
    assert result.numbers["weight"].number == 12.4
    assert result.numbers["weight"].confidence == 0.91
    assert result.datetimes["inspection"].datetime == "2024-02-29"
    assert result.texts["summary"].text == "Alice owns the photo"
    assert result.texts["summary"].probabilities["Alice owns the photo"] == 0.75


@pytest.mark.asyncio
async def test_async_primitives() -> None:
    async with AsyncModaic(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        result = await client.decisions.create(**parameters())
    assert isinstance(result.answers["owner"], SpanAnswer)
    assert isinstance(result.answers["weight"], NumberAnswer)


def test_criteria_rejected() -> None:
    for cls in (Span, Box):
        with pytest.raises(ValidationError):
            cls(instructions="Find it", criteria={})


def test_structured_constructor_validation() -> None:
    for cls in (Number, Text, Datetime):
        with pytest.raises(ValidationError):
            cls(instructions=" ")
    with pytest.raises(ValidationError):
        Datetime(instructions="When?", criteria={"format": "MM-DD-YYYY"})
    with pytest.raises(ValidationError):
        Datetime(instructions="When?", criteria={})


def test_cli_decision_request(monkeypatch: Any, capsys: Any, tmp_path: Any) -> None:
    import json

    from modaic import cli

    request = parameters()
    request["questions"] = {
        name: question.model_dump() for name, question in request["questions"].items()
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(
        cli,
        "_client",
        lambda args: Modaic(
            api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    assert cli.main(["--json", "decisions", "create", "--request", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["answers"]["owner"]["span"] == "Alice"
