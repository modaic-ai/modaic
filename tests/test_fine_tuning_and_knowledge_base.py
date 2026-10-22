from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from modaic import AsyncModaic, Modaic

JOB_ID = "9e3d5d17-a8db-4e61-b3c6-c8834f8faf12"
UPLOAD_ID = "41fcc86c-7f49-4daa-a8b1-577db0c8fe76"
MODEL_ID = "018f5f04-a793-7b21-a1f6-bc6b56789012"
SHA = "a" * 64
COMMIT = "b" * 40


def fine_tune_json() -> dict[str, object]:
    return {
        "id": JOB_ID,
        "repositoryId": MODEL_ID,
        "datasetId": "51fcc86c-7f49-4daa-a8b1-577db0c8fe76",
        "training_file": "hard-labels",
        "datasetCommitSha": COMMIT,
        "datasetSha256": SHA,
        "status": "done",
        "branch": "main",
        "sourceCommitSha": COMMIT,
        "resultCommitSha": "c" * 40,
        "training_type": "lora",
        "preset": "mo131_lora_head_v1",
        "progress": {"step": 4},
        "result": {
            "metrics": {"loss": 0.1},
            "evaluation": None,
            "servingQualification": None,
            "publication": {"commitSha": "c" * 40, "branch": "main"},
        },
        "error": None,
        "createdAt": "2026-10-20T00:00:00Z",
        "updatedAt": "2026-10-20T00:01:00Z",
        "startedAt": "2026-10-20T00:00:01Z",
        "finishedAt": "2026-10-20T00:01:00Z",
    }


def upload_json(*, uploaded: bool = False) -> dict[str, object]:
    return {
        "id": UPLOAD_ID,
        "name": "hard-labels",
        "description": None,
        "kind": "dataset",
        "sha256": SHA,
        "sizeBytes": 4,
        "repos": [],
        "status": "pending",
        "createdAt": "2026-10-20T00:00:00Z",
        "expiresAt": "2026-10-21T00:00:00Z",
        "partSizeBytes": 4,
        "parts": [
            {
                "partNumber": 1,
                "sha256": hashlib.sha256(b"data").hexdigest(),
                "sizeBytes": 4,
                "uploaded": uploaded,
            }
        ]
        if uploaded
        else [],
        "result": None,
        "error": None,
    }


def test_fine_tune_contract_and_full_training_controls() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json=fine_tune_json())
        if request.url.path.endswith("/logs"):
            return httpx.Response(
                200, json={"logs": ["trained"], "available": True, "running": False}
            )
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.url.path.endswith(JOB_ID):
            return httpx.Response(200, json=fine_tune_json())
        return httpx.Response(200, json={"fineTunes": [fine_tune_json()]})

    transport = httpx.MockTransport(handler)
    with Modaic(
        api_key="test-key",
        http_client=httpx.Client(base_url="https://example.test/api/v1", transport=transport),
    ) as client:
        created = client.fine_tunes.create(
            model="mo-1.3.1",
            destination="acme/color-model",
            training_file="hard-labels",
            training_type="lora",
            controls={
                "n_epochs": 2,
                "n_evals": 3,
                "batch_size": "max",
                "lr_scheduler": {"lr_scheduler_type": "cosine"},
                "early_stopping_enabled": True,
                "early_stopping_patience": 1,
            },
            validation_file="validation",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            idempotency_key="fine-tune-123",
        )
        assert created.result_commit_sha == "c" * 40
        assert created.result is not None
        assert created.result.publication is not None
        assert created.result.publication.commit_sha == "c" * 40
        listed = client.fine_tunes.list(workspace="acme", model_id=MODEL_ID)
        assert listed.fine_tunes[0].id == JOB_ID
        assert client.fine_tunes.get(JOB_ID).status == "done"
        assert client.fine_tunes.logs(JOB_ID).logs == ["trained"]
        client.fine_tunes.cancel(JOB_ID)

    create = requests[0]
    assert create.headers["idempotency-key"] == "fine-tune-123"
    assert json.loads(create.content) == {
        "model": "mo-1.3.1",
        "destination": "acme/color-model",
        "training_file": "hard-labels",
        "training_type": "lora",
        "n_epochs": 2,
        "n_evals": 3,
        "batch_size": "max",
        "lr_scheduler": {"lr_scheduler_type": "cosine"},
        "early_stopping_enabled": True,
        "early_stopping_patience": 1,
        "validation_file": "validation",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }
    assert dict(requests[1].url.params) == {
        "workspace": "acme",
        "modelId": MODEL_ID,
        "limit": "20",
    }


def test_fine_tune_list_omits_unspecified_model_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"fineTunes": []})

    with Modaic(
        api_key="test-key",
        http_client=httpx.Client(
            base_url="https://example.test/api/v1",
            transport=httpx.MockTransport(handler),
        ),
    ) as client:
        assert client.fine_tunes.list(workspace="acme").fine_tunes == []

    assert dict(requests[0].url.params) == {"workspace": "acme", "limit": "20"}


def test_knowledge_base_upload_and_download_do_not_leak_api_token() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "transfer.test" and request.method == "PUT":
            assert request.headers.get("authorization") == "Bearer part-grant"
            assert request.content == b"data"
            return httpx.Response(200, json={"receipt": "part-receipt"})
        if request.url.host == "download.test":
            assert request.headers.get("authorization") is None
            return httpx.Response(200, content=b"resource-bytes")
        path = request.url.path
        if path.endswith("/parts"):
            return httpx.Response(
                201,
                json={
                    "partNumber": 1,
                    "uploadUrl": "https://transfer.test/upload",
                    "method": "PUT",
                    "headers": {"Authorization": "Bearer part-grant"},
                    "expiresIn": 300,
                },
            )
        if path.endswith("/parts/1"):
            assert json.loads(request.content) == {"receipt": "part-receipt"}
            return httpx.Response(200, json=upload_json(uploaded=True))
        if path.endswith("/content"):
            return httpx.Response(307, headers={"Location": "https://download.test/object"})
        if path.endswith("/resources/hard-labels"):
            return httpx.Response(
                200,
                json={
                    "id": "51fcc86c-7f49-4daa-a8b1-577db0c8fe76",
                    "name": "hard-labels",
                    "description": None,
                    "kind": "dataset",
                    "path": "resources/uploaded-datasets/hard-labels.jsonl",
                    "sha256": SHA,
                    "sizeBytes": 4,
                    "repos": [],
                    "commitSha": COMMIT,
                },
            )
        return httpx.Response(201, json=upload_json())

    transport = httpx.MockTransport(handler)
    with Modaic(
        api_key="api-secret",
        http_client=httpx.Client(base_url="https://example.test/api/v1", transport=transport),
    ) as client:
        session = client.knowledge_base.uploads.create(
            "acme", kind="dataset", sha256=SHA, size_bytes=4, name="hard-labels"
        )
        uploaded = client.knowledge_base.uploads.upload_part(
            "acme", session.id, part_number=1, data=b"data"
        )
        assert uploaded.parts[0].uploaded is True
        assert client.knowledge_base.resources.get("acme", "hard-labels").kind == "dataset"
        assert client.knowledge_base.resources.download("acme", "hard-labels") == b"resource-bytes"

    api_requests = [request for request in requests if request.url.host == "example.test"]
    assert api_requests
    assert all(
        request.headers.get("authorization") == "Bearer api-secret" for request in api_requests
    )


@pytest.mark.asyncio
async def test_async_resources_match_sync_surface() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/fine-tunes":
            if request.method == "POST":
                return httpx.Response(202, json=fine_tune_json())
            return httpx.Response(200, json={"fineTunes": [fine_tune_json()]})
        return httpx.Response(201, json=upload_json())

    async with AsyncModaic(
        api_key="test-key",
        http_client=httpx.AsyncClient(
            base_url="https://example.test/api/v1", transport=httpx.MockTransport(handler)
        ),
    ) as client:
        job = await client.fine_tunes.create(
            model="acme/color-model",
            training_file="hard-labels",
            training_type="head_only",
        )
        assert job.training_type == "lora"
        session = await client.knowledge_base.uploads.create(
            "acme", kind="dataset", sha256=SHA, size_bytes=4, name="hard-labels"
        )
        assert session.name == "hard-labels"

    assert json.loads(requests[0].content) == {
        "model": "acme/color-model",
        "training_file": "hard-labels",
        "training_type": "head_only",
    }


def test_upload_file_streams_parts_completes_and_waits(tmp_path: Path) -> None:
    payload = b"abcdefghij"
    source = tmp_path / "training.jsonl"
    source.write_bytes(payload)
    requests: list[httpx.Request] = []
    transfers: list[bytes] = []
    progress: list[tuple[int, int]] = []

    def session(status: str) -> dict[str, object]:
        return {
            **upload_json(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "sizeBytes": len(payload),
            "partSizeBytes": 4,
            "status": status,
        }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "transfer.test":
            assert request.headers.get("authorization") == "Bearer part-grant"
            transfers.append(request.content)
            return httpx.Response(200, json={"receipt": f"receipt-{len(transfers)}"})
        path = request.url.path
        if path.endswith("/parts"):
            part_number = json.loads(request.content)["partNumber"]
            return httpx.Response(
                201,
                json={
                    "partNumber": part_number,
                    "uploadUrl": f"https://transfer.test/{part_number}",
                    "method": "PUT",
                    "headers": {"Authorization": "Bearer part-grant"},
                    "expiresIn": 300,
                },
            )
        if "/parts/" in path:
            return httpx.Response(200, json=session("pending"))
        if path.endswith("/complete"):
            return httpx.Response(202, json=session("processing"))
        if request.method == "GET":
            return httpx.Response(200, json=session("completed"))
        return httpx.Response(201, json=session("pending"))

    with Modaic(
        api_key="api-secret",
        http_client=httpx.Client(
            base_url="https://example.test/api/v1", transport=httpx.MockTransport(handler)
        ),
    ) as client:
        result = client.knowledge_base.uploads.upload_file(
            "acme",
            source,
            kind="dataset",
            name="hard-labels",
            poll_interval=0,
            progress=lambda uploaded, total: progress.append((uploaded, total)),
        )

    assert result.status == "completed"
    assert transfers == [b"abcd", b"efgh", b"ij"]
    assert progress == [(0, 10), (4, 10), (8, 10), (10, 10)]
    create_body = json.loads(requests[0].content)
    assert create_body["sha256"] == hashlib.sha256(payload).hexdigest()
    assert create_body["sizeBytes"] == 10


@pytest.mark.asyncio
async def test_async_upload_file_cancels_an_interrupted_transfer(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"abcdefgh")
    cancelled = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal cancelled
        if request.url.host == "transfer.test":
            return httpx.Response(503, json={"error": "unavailable"})
        if request.method == "DELETE":
            cancelled = True
            return httpx.Response(200, json=upload_json())
        if request.url.path.endswith("/parts"):
            return httpx.Response(
                201,
                json={
                    "partNumber": 1,
                    "uploadUrl": "https://transfer.test/1",
                    "method": "PUT",
                    "headers": {"Authorization": "Bearer part-grant"},
                    "expiresIn": 300,
                },
            )
        return httpx.Response(
            201,
            json={**upload_json(), "sizeBytes": 8, "partSizeBytes": 4},
        )

    async with AsyncModaic(
        api_key="api-secret",
        http_client=httpx.AsyncClient(
            base_url="https://example.test/api/v1", transport=httpx.MockTransport(handler)
        ),
    ) as client:
        with pytest.raises(Exception, match="HTTP 503"):
            await client.knowledge_base.uploads.upload_file(
                "acme", source, kind="artifact", name="artifact"
            )

    assert cancelled is True
