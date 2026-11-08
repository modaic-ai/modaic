from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from modaic import (
    AsyncModaic,
    Box,
    BoxAnswer,
    Modaic,
    Question,
    Signature,
    Span,
    SpanAnswer,
)


def handler(request: httpx.Request) -> httpx.Response:
    import json

    body = json.loads(request.content)
    assert body["model"] == "mo-2.0"
    assert body["assets"]["photo"]["data_base64"] == "test-bytes"
    assert body["questions"]["owner"] == {"type": "extraction", "instructions": "Who?"}
    return httpx.Response(
        200,
        json={
            "model": "mo-2.0",
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "answers": {
                "owner": {
                    "type": "extraction",
                    "extraction": {"start": 3, "end": 8, "text": "Alice"},
                    "confidence": 0.9,
                },
                "location": {
                    "type": "box",
                    "box": {"top": 0, "bottom": 1, "left": 0, "right": 1},
                    "confidence": 0.8,
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
        },
    )


def test_sync_primitives() -> None:
    with Modaic(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ) as client:
        result = client.decisions.create(**parameters())
    assert isinstance(result.answers["owner"], SpanAnswer)
    assert isinstance(result.answers["location"], BoxAnswer)
    assert result.answers["owner"].start == 3


def test_saved_model_span_projection_before_custom_validation() -> None:
    from pydantic import model_validator

    from modaic import DecisionResponse

    class Review(DecisionResponse):
        owner: Span

        @model_validator(mode="before")
        @classmethod
        def check_projection(cls, value: Any) -> Any:
            answer = value["answers"]["owner"]
            assert answer["type"] == "span"
            assert answer["span"] == "Alice"
            assert "extraction" not in answer
            return value

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "acme/review",
                "usage": {"input_tokens": 5, "output_tokens": 0},
                "answers": {
                    "owner": {
                        "type": "extraction",
                        "extraction": {
                            "start": 3,
                            "end": 8,
                            "text": "Alice",
                        },
                        "confidence": 0.9,
                    }
                },
            },
        )

    with Modaic(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))
    ) as client:
        result = client.decisions.create(
            model="acme/review",
            state="😀 Alice",
            response_model=Review,
        )
    assert isinstance(result.owner, Span)
    assert result.owner is result.answers["owner"]
    assert result.owner.span == "Alice"


@pytest.mark.asyncio
async def test_async_primitives() -> None:
    async with AsyncModaic(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        result = await client.decisions.create(**parameters())
    assert isinstance(result.answers["owner"], SpanAnswer)


def test_criteria_rejected() -> None:
    for cls in (Span, Box):
        with pytest.raises(ValidationError):
            cls(instructions="Find it", criteria={})


@pytest.mark.parametrize("probability", [0.4, 0.5])
@pytest.mark.parametrize("signature", [False, True])
@pytest.mark.asyncio
async def test_nullable_extraction(probability: float, signature: bool) -> None:
    import json

    class Review(Signature):
        owner: Span | None = Question(instructions="Who?")

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["questions"]["owner"] == {"type": "extraction", "instructions": "Who?"}
        gate = body["questions"]["__modaic_answerable_owner"]
        assert gate["instructions"]["question"]["type"] == "extraction"
        return httpx.Response(
            200,
            json={
                "model": "mo-2.0",
                "usage": {"input_tokens": 5, "output_tokens": 0},
                "answers": {
                    "owner": {
                        "type": "extraction",
                        "extraction": {
                            "start": 3,
                            "end": 8,
                            "text": "Alice",
                        },
                        "confidence": 0.9,
                    },
                    "__modaic_answerable_owner": {"type": "noul", "noul": probability},
                },
            },
        )

    async with AsyncModaic(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ) as client:
        result = await client.decisions.create(
            model="mo-2.0",
            state="😀 Alice",
            questions=Review if signature else {"owner": Span(instructions="Who?").nullable()},
        )
    owner = result.answers["owner"]
    assert list(result.answers) == ["owner"]
    if signature:
        assert isinstance(result, Review)
        assert result.owner is owner
    if probability < 0.5:
        assert owner is None
    else:
        assert isinstance(owner, Span)
        assert owner.span == "Alice"
        assert owner.name == "owner"
        assert owner.to_question() == {"type": "span", "instructions": "Who?"}
        owner.span = "Bob"
        assert owner.model_dump()["span"] == "Bob"


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
