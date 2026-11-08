from __future__ import annotations

import json

import httpx
import pytest

from modaic import AsyncModaic, Modaic

JOB_ID = "9e3d5d17-a8db-4e61-b3c6-c8834f8faf12"


def test_protege_lifecycle() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status = "running" if request.method == "POST" else "cancelled"
        return httpx.Response(
            201 if request.method == "POST" else 200, json={"id": JOB_ID, "status": status}
        )

    with Modaic(
        api_key="test-key",
        http_client=httpx.Client(
            base_url="https://example.test/api/v1", transport=httpx.MockTransport(handler)
        ),
    ) as client:
        created = client.protege.create(
            prompt="Improve the model", repo="acme/support", eval_only_dataset="heldout"
        )
        cancelled = client.protege.cancel(created.id)

    assert created.status == "running"
    assert cancelled.status == "cancelled"
    assert json.loads(requests[0].content) == {
        "prompt": "Improve the model",
        "repo": "acme/support",
        "eval_only_dataset": "heldout",
    }
    assert requests[1].url.path == f"/api/v1/protege/{JOB_ID}"


@pytest.mark.asyncio
async def test_async_protege_get() -> None:
    async with AsyncModaic(
        api_key="test-key",
        http_client=httpx.AsyncClient(
            base_url="https://example.test/api/v1",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"id": JOB_ID, "status": "succeeded"})
            ),
        ),
    ) as client:
        assert (await client.protege.get(JOB_ID)).status == "succeeded"
