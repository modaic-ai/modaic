from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from test_client import alignment_json

from modaic import AsyncModaic, Modaic


def transport_request(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST"
    assert request.url.path == "/api/v1/instruction-optimizations"
    assert request.headers["idempotency-key"] == "dataset-optimization-1"
    assert json.loads(request.content) == {
        "model": "mo-latest",
        "destination": "acme/triage",
        "training_file": "train",
        "validation_file": "validation",
        "eval_only_dataset": "heldout",
        "branch": "main",
        "reflection": {"seed": 0},
    }
    return httpx.Response(
        202,
        json={
            **alignment_json(),
            "training_file": "train",
            "datasetCommitSha": "a" * 40,
            "datasetSha256": "b" * 64,
        },
    )


PARAMS: dict[str, Any] = {
    "model": "mo-latest",
    "destination": "acme/triage",
    "training_file": "train",
    "validation_file": "validation",
    "eval_only_dataset": "heldout",
    "idempotency_key": "dataset-optimization-1",
}


def test_sync_optimization_dataset_contract() -> None:
    with Modaic(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(transport_request))
    ) as client:
        job = client.instruction_optimizations.create(**PARAMS)
        assert job.training_file == "train"
        assert job.dataset_commit_sha == "a" * 40
        assert job.dataset_sha256 == "b" * 64


@pytest.mark.asyncio
async def test_async_optimization_dataset_contract() -> None:
    async with AsyncModaic(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport_request)),
    ) as client:
        job = await client.instruction_optimizations.create(**PARAMS)
        assert job.training_file == "train"
        assert job.dataset_commit_sha == "a" * 40
