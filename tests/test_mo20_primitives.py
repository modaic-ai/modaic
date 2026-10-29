from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from modaic import AsyncModaic, Box, BoxAnswer, Extraction, ExtractionAnswer, Modaic


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
            "owner": Extraction(instructions="Who?"),
            "location": Box(instructions="Locate photo."),
        },
    )


def test_sync_primitives() -> None:
    with Modaic(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ) as client:
        result = client.decisions.create(**parameters())
    assert isinstance(result.answers["owner"], ExtractionAnswer)
    assert isinstance(result.answers["location"], BoxAnswer)
    assert result.answers["owner"].extraction.start == 3


@pytest.mark.asyncio
async def test_async_primitives() -> None:
    async with AsyncModaic(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        result = await client.decisions.create(**parameters())
    assert isinstance(result.answers["owner"], ExtractionAnswer)


def test_criteria_rejected() -> None:
    for cls in (Extraction, Box):
        with pytest.raises(ValidationError):
            cls(instructions="Find it", criteria={})


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
    assert json.loads(capsys.readouterr().out)["answers"]["owner"]["extraction"]["text"] == "Alice"
