import json
from typing import Any

import httpx
import pytest
from test_regressions import Harness, call
from test_regressions import api as api

from modaic import Box, BoxAnswer, ModaicConnectionError, Question, Signature

COORDINATES = {"top": 0.1, "bottom": 0.8, "left": 0.2, "right": 0.9}


def response(box: Any = None) -> dict[str, Any]:
    return {
        "model": "mo-2.0",
        "answers": {
            "region": {"type": "box", "box": COORDINATES if box is None else box, "confidence": 0.9}
        },
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.mark.parametrize("signature", [False, True])
async def test_box(api: Harness, signature: bool) -> None:
    class Image(Signature):
        region: Box = Question(instructions="Find the object")

    api.handler = lambda _: httpx.Response(200, json=response())
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        questions=Image if signature else {"region": Box(instructions="Find the object")},
    )
    region = result.region if signature else result.answers["region"]
    assert type(region) is Box and isinstance(region, BoxAnswer)
    assert region.box.top == 0.1 and region.box.right == 0.9
    assert region.name == "region" and region.instructions == "Find the object"
    assert region.model_dump() == response()["answers"]["region"]
    assert json.loads(api.requests[-1].content)["questions"]["region"] == {
        "type": "box",
        "instructions": "Find the object",
    }


async def test_box_saved_model_and_nullable(api: Harness) -> None:
    payload = response()
    api.handler = lambda _: httpx.Response(200, json=payload)
    saved = await call(api.client.decisions.create, model="acme/image", state={})
    assert type(saved.answers["region"]) is Box
    payload["answers"]["__modaic_answerable_region"] = {"type": "noul", "noul": 0.1}
    result = await call(
        api.client.decisions.create,
        model="mo-2.0",
        state={},
        questions={"region": Box(instructions="Find").nullable()},
    )
    assert result.answers["region"] is None


@pytest.mark.parametrize(
    "coordinates",
    [
        {**COORDINATES, "top": -0.1},
        {**COORDINATES, "right": 1.1},
        {**COORDINATES, "top": 0.9},
        {**COORDINATES, "left": 0.95},
    ],
)
async def test_invalid_box(api: Harness, coordinates: dict[str, float]) -> None:
    api.handler = lambda _: httpx.Response(200, json=response(coordinates))
    with pytest.raises(ModaicConnectionError):
        await call(api.client.decisions.create, model="acme/image", state={})
