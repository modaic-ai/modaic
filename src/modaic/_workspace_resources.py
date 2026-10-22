from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from ._transport import AsyncTransport, SyncTransport
from .errors import ModaicConnectionError, ModaicTimeoutError
from .types import (
    PreparedUploadPart,
    Resource,
    ResourceKind,
    ResourcePage,
    Upload,
)

_HASH_BUFFER_BYTES = 8 * 1024 * 1024
_UPLOAD_TERMINAL = {"completed", "failed", "cancelled", "expired"}
UploadProgress = Callable[[int, int], None]


def _file_sha256_and_size(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_HASH_BUFFER_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _query(**values: str | int | None) -> dict[str, str | int]:
    return {key: value for key, value in values.items() if value is not None}


def _workspace_path(workspace: str, suffix: str) -> str:
    return f"/workspaces/{quote(workspace, safe='')}{suffix}"


class Uploads:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(
        self,
        workspace: str,
        *,
        name: str,
        kind: ResourceKind,
        sha256: str,
        size_bytes: int,
        description: str | None = None,
        repos: Sequence[str] | None = None,
    ) -> Upload:
        body: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "sha256": sha256,
            "sizeBytes": size_bytes,
            "repos": list(repos or []),
        }
        if description is not None:
            body["description"] = description
        return cast(
            Upload,
            self._transport.request(
                "POST", _workspace_path(workspace, "/uploads"), json=body, model=Upload
            ),
        )

    def get(self, workspace: str, upload_id: str) -> Upload:
        return cast(
            Upload,
            self._transport.request(
                "GET",
                _workspace_path(workspace, f"/uploads/{quote(upload_id, safe='')}"),
                model=Upload,
            ),
        )

    def prepare_part(
        self,
        workspace: str,
        upload_id: str,
        *,
        part_number: int,
        sha256: str,
        size_bytes: int,
    ) -> PreparedUploadPart:
        return cast(
            PreparedUploadPart,
            self._transport.request(
                "POST",
                _workspace_path(workspace, f"/uploads/{quote(upload_id, safe='')}/parts"),
                json={"partNumber": part_number, "sha256": sha256, "sizeBytes": size_bytes},
                model=PreparedUploadPart,
            ),
        )

    def confirm_part(
        self,
        workspace: str,
        upload_id: str,
        *,
        part_number: int,
        receipt: str,
    ) -> Upload:
        return cast(
            Upload,
            self._transport.request(
                "PUT",
                _workspace_path(
                    workspace, f"/uploads/{quote(upload_id, safe='')}/parts/{part_number}"
                ),
                json={"receipt": receipt},
                model=Upload,
            ),
        )

    def upload_part(
        self,
        workspace: str,
        upload_id: str,
        *,
        part_number: int,
        data: bytes | bytearray | memoryview,
    ) -> Upload:
        content = bytes(data)
        prepared = self.prepare_part(
            workspace,
            upload_id,
            part_number=part_number,
            sha256=sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        receipt = self._transport.upload(
            prepared.upload_url, headers=prepared.headers, content=content
        )
        return self.confirm_part(
            workspace,
            upload_id,
            part_number=part_number,
            receipt=receipt,
        )

    def wait(
        self,
        workspace: str,
        upload_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> Upload:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.get(workspace, upload_id)
            if result.status in _UPLOAD_TERMINAL:
                return result
            if deadline is not None and time.monotonic() + poll_interval >= deadline:
                raise ModaicTimeoutError("Timed out waiting for resource validation.")
            time.sleep(poll_interval)

    def upload_file(
        self,
        workspace: str,
        path: str | PathLike[str],
        *,
        kind: ResourceKind,
        name: str,
        description: str | None = None,
        repos: Sequence[str] | None = None,
        wait_for_validation: bool = True,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        progress: UploadProgress | None = None,
    ) -> Upload:
        """Upload a local file without materializing it in memory."""
        source = Path(path)
        digest, size = _file_sha256_and_size(source)
        upload = self.create(
            workspace,
            kind=kind,
            sha256=digest,
            size_bytes=size,
            name=name,
            description=description,
            repos=repos,
        )
        uploaded = 0
        try:
            if progress is not None:
                progress(0, size)
            with source.open("rb") as stream:
                part_number = 1
                while uploaded < size:
                    chunk = stream.read(min(upload.part_size_bytes, size - uploaded))
                    if not chunk:
                        raise ModaicConnectionError(
                            "Local file changed while it was being uploaded."
                        )
                    self.upload_part(
                        workspace,
                        upload.id,
                        part_number=part_number,
                        data=chunk,
                    )
                    uploaded += len(chunk)
                    if progress is not None:
                        progress(uploaded, size)
                    part_number += 1
        except Exception:
            with suppress(Exception):
                self.cancel(workspace, upload.id)
            raise

        result = self.complete(workspace, upload.id)
        if not wait_for_validation or result.status in _UPLOAD_TERMINAL:
            return result
        return self.wait(
            workspace,
            upload.id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def complete(self, workspace: str, upload_id: str) -> Upload:
        return cast(
            Upload,
            self._transport.request(
                "POST",
                _workspace_path(workspace, f"/uploads/{quote(upload_id, safe='')}/complete"),
                model=Upload,
            ),
        )

    def cancel(self, workspace: str, upload_id: str) -> Upload:
        return cast(
            Upload,
            self._transport.request(
                "DELETE",
                _workspace_path(workspace, f"/uploads/{quote(upload_id, safe='')}"),
                model=Upload,
            ),
        )


class Resources:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def list(
        self,
        *,
        workspace: str,
        kind: ResourceKind | None = None,
        glob: str | None = None,
        commit_sha: str | None = None,
        cursor: str | None = None,
    ) -> ResourcePage:
        return cast(
            ResourcePage,
            self._transport.request(
                "GET",
                _workspace_path(workspace, "/resources"),
                params=_query(kind=kind, glob=glob, commitSha=commit_sha, cursor=cursor),
                model=ResourcePage,
            ),
        )

    def get(
        self, workspace: str, name: str, *, commit_sha: str | None = None
    ) -> Resource:
        return cast(
            Resource,
            self._transport.request(
                "GET",
                _workspace_path(workspace, f"/resources/{quote(name, safe='')}"),
                params=_query(commitSha=commit_sha),
                model=Resource,
            ),
        )

    def download(self, workspace: str, name: str, *, commit_sha: str | None = None) -> bytes:
        return self._transport.download(
            _workspace_path(workspace, f"/resources/{quote(name, safe='')}/content"),
            params=_query(commitSha=commit_sha),
        )

    def update(
        self,
        workspace: str,
        name: str,
        *,
        description: str | None | object = ...,
        repos: Sequence[str] | None = None,
    ) -> Resource:
        body: dict[str, Any] = {}
        if description is not ...:
            body["description"] = description
        if repos is not None:
            body["repos"] = list(repos)
        if not body:
            raise ValueError("Provide description and/or repos.")
        return cast(
            Resource,
            self._transport.request(
                "PATCH",
                _workspace_path(workspace, f"/resources/{quote(name, safe='')}"),
                json=body,
                model=Resource,
            ),
        )


class AsyncUploads:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self,
        workspace: str,
        *,
        name: str,
        kind: ResourceKind,
        sha256: str,
        size_bytes: int,
        description: str | None = None,
        repos: Sequence[str] | None = None,
    ) -> Upload:
        body: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "sha256": sha256,
            "sizeBytes": size_bytes,
            "repos": list(repos or []),
        }
        if description is not None:
            body["description"] = description
        return cast(
            Upload,
            await self._transport.request(
                "POST", _workspace_path(workspace, "/uploads"), json=body, model=Upload
            ),
        )

    async def get(self, workspace: str, upload_id: str) -> Upload:
        return cast(
            Upload,
            await self._transport.request(
                "GET",
                _workspace_path(workspace, f"/uploads/{quote(upload_id, safe='')}"),
                model=Upload,
            ),
        )

    async def prepare_part(
        self,
        workspace: str,
        upload_id: str,
        *,
        part_number: int,
        sha256: str,
        size_bytes: int,
    ) -> PreparedUploadPart:
        return cast(
            PreparedUploadPart,
            await self._transport.request(
                "POST",
                _workspace_path(workspace, f"/uploads/{quote(upload_id, safe='')}/parts"),
                json={"partNumber": part_number, "sha256": sha256, "sizeBytes": size_bytes},
                model=PreparedUploadPart,
            ),
        )

    async def confirm_part(
        self,
        workspace: str,
        upload_id: str,
        *,
        part_number: int,
        receipt: str,
    ) -> Upload:
        return cast(
            Upload,
            await self._transport.request(
                "PUT",
                _workspace_path(
                    workspace, f"/uploads/{quote(upload_id, safe='')}/parts/{part_number}"
                ),
                json={"receipt": receipt},
                model=Upload,
            ),
        )

    async def upload_part(
        self,
        workspace: str,
        upload_id: str,
        *,
        part_number: int,
        data: bytes | bytearray | memoryview,
    ) -> Upload:
        content = bytes(data)
        prepared = await self.prepare_part(
            workspace,
            upload_id,
            part_number=part_number,
            sha256=sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        receipt = await self._transport.upload(
            prepared.upload_url, headers=prepared.headers, content=content
        )
        return await self.confirm_part(
            workspace,
            upload_id,
            part_number=part_number,
            receipt=receipt,
        )

    async def wait(
        self,
        workspace: str,
        upload_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> Upload:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = await self.get(workspace, upload_id)
            if result.status in _UPLOAD_TERMINAL:
                return result
            if deadline is not None and time.monotonic() + poll_interval >= deadline:
                raise ModaicTimeoutError("Timed out waiting for resource validation.")
            await asyncio.sleep(poll_interval)

    async def upload_file(
        self,
        workspace: str,
        path: str | PathLike[str],
        *,
        kind: ResourceKind,
        name: str,
        description: str | None = None,
        repos: Sequence[str] | None = None,
        wait_for_validation: bool = True,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        progress: UploadProgress | None = None,
    ) -> Upload:
        """Upload a local file without blocking the event loop on file reads."""
        source = Path(path)
        digest, size = await asyncio.to_thread(_file_sha256_and_size, source)
        upload = await self.create(
            workspace,
            kind=kind,
            sha256=digest,
            size_bytes=size,
            name=name,
            description=description,
            repos=repos,
        )
        uploaded = 0
        stream = None
        try:
            if progress is not None:
                progress(0, size)
            stream = await asyncio.to_thread(source.open, "rb")
            part_number = 1
            while uploaded < size:
                chunk = await asyncio.to_thread(
                    stream.read, min(upload.part_size_bytes, size - uploaded)
                )
                if not chunk:
                    raise ModaicConnectionError("Local file changed while it was being uploaded.")
                await self.upload_part(
                    workspace,
                    upload.id,
                    part_number=part_number,
                    data=chunk,
                )
                uploaded += len(chunk)
                if progress is not None:
                    progress(uploaded, size)
                part_number += 1
        except Exception:
            with suppress(Exception):
                await self.cancel(workspace, upload.id)
            raise
        finally:
            if stream is not None:
                await asyncio.to_thread(stream.close)

        result = await self.complete(workspace, upload.id)
        if not wait_for_validation or result.status in _UPLOAD_TERMINAL:
            return result
        return await self.wait(
            workspace,
            upload.id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    async def complete(self, workspace: str, upload_id: str) -> Upload:
        return cast(
            Upload,
            await self._transport.request(
                "POST",
                _workspace_path(workspace, f"/uploads/{quote(upload_id, safe='')}/complete"),
                model=Upload,
            ),
        )

    async def cancel(self, workspace: str, upload_id: str) -> Upload:
        return cast(
            Upload,
            await self._transport.request(
                "DELETE",
                _workspace_path(workspace, f"/uploads/{quote(upload_id, safe='')}"),
                model=Upload,
            ),
        )


class AsyncResources:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def list(
        self,
        *,
        workspace: str,
        kind: ResourceKind | None = None,
        glob: str | None = None,
        commit_sha: str | None = None,
        cursor: str | None = None,
    ) -> ResourcePage:
        return cast(
            ResourcePage,
            await self._transport.request(
                "GET",
                _workspace_path(workspace, "/resources"),
                params=_query(kind=kind, glob=glob, commitSha=commit_sha, cursor=cursor),
                model=ResourcePage,
            ),
        )

    async def get(
        self, workspace: str, name: str, *, commit_sha: str | None = None
    ) -> Resource:
        return cast(
            Resource,
            await self._transport.request(
                "GET",
                _workspace_path(workspace, f"/resources/{quote(name, safe='')}"),
                params=_query(commitSha=commit_sha),
                model=Resource,
            ),
        )

    async def download(self, workspace: str, name: str, *, commit_sha: str | None = None) -> bytes:
        return await self._transport.download(
            _workspace_path(workspace, f"/resources/{quote(name, safe='')}/content"),
            params=_query(commitSha=commit_sha),
        )

    async def update(
        self,
        workspace: str,
        name: str,
        *,
        description: str | None | object = ...,
        repos: Sequence[str] | None = None,
    ) -> Resource:
        body: dict[str, Any] = {}
        if description is not ...:
            body["description"] = description
        if repos is not None:
            body["repos"] = list(repos)
        if not body:
            raise ValueError("Provide description and/or repos.")
        return cast(
            Resource,
            await self._transport.request(
                "PATCH",
                _workspace_path(workspace, f"/resources/{quote(name, safe='')}"),
                json=body,
                model=Resource,
            ),
        )

