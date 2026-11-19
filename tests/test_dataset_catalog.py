from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from modaic import AsyncModaic, Dataset, DatasetPage, Modaic, cli

COMMIT = "a" * 40
ITEM = {
    "id": "51fcc86c-7f49-4daa-a8b1-577db0c8fe76",
    "name": "train",
    "description": None,
    "kind": "dataset",
    "path": "resources/uploaded-datasets/train.jsonl",
    "sha256": "b" * 64,
    "sizeBytes": 12,
    "repos": [],
    "commitSha": COMMIT,
}


def response(request: httpx.Request) -> httpx.Response:
    assert request.headers["authorization"] == "Bearer test-key"
    path = request.url.path
    assert "/resources" not in path
    if path.endswith("/datasets"):
        assert request.url.params["commitSha"] == COMMIT
        assert request.url.params["cursor"] == "next"
        return httpx.Response(
            200, json={"datasets": [ITEM], "commitSha": COMMIT, "nextCursor": None}
        )
    if request.method == "PATCH":
        assert json.loads(request.content) == {"description": None, "repos": []}
    if path.endswith("/content"):
        return httpx.Response(200, content=b"dataset bytes")
    return httpx.Response(200, json=ITEM)


def test_sync_dataset_catalog_contract() -> None:
    with Modaic(
        api_key="test-key",
        http_client=httpx.Client(
            base_url="https://example.test/api/v1", transport=httpx.MockTransport(response)
        ),
    ) as client:
        page = client.datasets.list(workspace="acme", commit_sha=COMMIT, cursor="next")
        assert isinstance(page, DatasetPage)
        assert isinstance(page.datasets[0], Dataset)
        assert client.datasets.get("acme", "train", commit_sha=COMMIT).name == "train"
        assert (
            client.datasets.update("acme", "train", description=None, repos=[]).description is None
        )
        assert client.datasets.download("acme", "train") == b"dataset bytes"
        assert not hasattr(client, "resources")


@pytest.mark.asyncio
async def test_async_dataset_catalog_contract() -> None:
    async with AsyncModaic(
        api_key="test-key",
        http_client=httpx.AsyncClient(
            base_url="https://example.test/api/v1", transport=httpx.MockTransport(response)
        ),
    ) as client:
        page = await client.datasets.list(workspace="acme", commit_sha=COMMIT, cursor="next")
        assert isinstance(page.datasets[0], Dataset)
        assert (await client.datasets.get("acme", "train")).name == "train"
        assert (
            await client.datasets.update("acme", "train", description=None, repos=[])
        ).description is None
        assert await client.datasets.download("acme", "train") == b"dataset bytes"
        assert not hasattr(client, "resources")


@pytest.mark.parametrize("action", ["list", "get", "update", "download", "upload"])
def test_cli_datasets_dispatch(action: str, monkeypatch: Any, tmp_path: Any) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.datasets.download.return_value = b"dataset bytes"
    monkeypatch.setattr(cli, "_client", lambda args: client)
    monkeypatch.setattr(cli, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_success", lambda *args, **kwargs: None)
    args = ["datasets", action, "acme"]
    if action == "upload":
        args += [str(tmp_path / "train.jsonl"), "--kind", "dataset", "--name", "train"]
    elif action != "list":
        args += ["train"]
    if action == "download":
        args += ["--output", str(tmp_path / "download.jsonl")]
    if action == "update":
        args += ["--description", "updated"]
    assert cli.main(args) == 0
    if action == "upload":
        client.uploads.upload_file.assert_called_once()
    else:
        getattr(client.datasets, action).assert_called_once()


def test_cli_rejects_retired_resources_command() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["resources", "list", "acme"])
    assert error.value.code == 2


DELETION = {
    "id": ITEM["id"],
    "name": "train",
    "status": "queued",
    "createdAt": "2026-11-18T00:00:00Z",
    "completedAt": None,
}


def test_sync_permanent_deletion_polls_until_completed() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            assert request.url.path.endswith("/datasets/train")
            return httpx.Response(202, json=DELETION)
        assert request.url.path.endswith("/dataset-deletions/" + str(ITEM["id"]))
        return httpx.Response(
            200, json={**DELETION, "status": "completed", "completedAt": "2026-11-18T00:10:01Z"}
        )

    with Modaic(
        api_key="test-key",
        http_client=httpx.Client(
            base_url="https://example.test/api/v1", transport=httpx.MockTransport(handler)
        ),
    ) as client:
        deletion = client.datasets.delete("acme", "train")
        result = client.datasets.wait_for_deletion("acme", deletion.id)
        assert result.status == "completed"
        assert result.completed_at is not None
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_async_permanent_deletion_and_timeout() -> None:
    from modaic import ModaicTimeoutError

    state = "queued"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            202 if request.method == "DELETE" else 200, json={**DELETION, "status": state}
        )

    async with AsyncModaic(
        api_key="test-key",
        http_client=httpx.AsyncClient(
            base_url="https://example.test/api/v1", transport=httpx.MockTransport(handler)
        ),
    ) as client:
        deletion = await client.datasets.delete("acme", "train")
        with pytest.raises(ModaicTimeoutError):
            await client.datasets.wait_for_deletion("acme", deletion.id, timeout=0)
        state = "completed"
        assert (await client.datasets.wait_for_deletion("acme", deletion.id)).status == "completed"


@pytest.mark.parametrize("no_wait", [False, True])
def test_cli_permanent_delete_waits_by_default(no_wait: bool, monkeypatch: Any) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    monkeypatch.setattr(cli, "_client", lambda args: client)
    monkeypatch.setattr(cli, "_emit", lambda *args, **kwargs: None)
    args = ["datasets", "delete", "acme", "train"] + (["--no-wait"] if no_wait else [])
    assert cli.main(args) == 0
    client.datasets.delete.assert_called_once_with("acme", "train")
    assert client.datasets.wait_for_deletion.call_count == (0 if no_wait else 1)
