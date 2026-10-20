from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import Any, cast

from ._transport import AsyncTransport, SyncTransport
from .errors import ModaicConnectionError, ModaicTimeoutError
from .types import (
    KnowledgeBaseResource,
    KnowledgeBaseResourcePage,
    KnowledgeBaseUpload,
    PreparedUploadPart,
    ResourceKind,
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


class KnowledgeBaseUploads:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(
        self,
        *,
        kind: ResourceKind,
        sha256: str,
        size_bytes: int,
        name: str | None = None,
        workspace: str | None = None,
    ) -> KnowledgeBaseUpload:
        body: dict[str, Any] = {"kind": kind, "sha256": sha256, "sizeBytes": size_bytes}
        if name is not None:
            body["name"] = name
        if workspace is not None:
            body["workspace"] = workspace
        return cast(
            KnowledgeBaseUpload,
            self._transport.request(
                "POST", "/knowledge-base/uploads", json=body, model=KnowledgeBaseUpload
            ),
        )

    def get(self, upload_id: str, *, workspace: str | None = None) -> KnowledgeBaseUpload:
        return cast(
            KnowledgeBaseUpload,
            self._transport.request(
                "GET",
                f"/knowledge-base/uploads/{upload_id}",
                params=_query(workspace=workspace),
                model=KnowledgeBaseUpload,
            ),
        )

    def prepare_part(
        self,
        upload_id: str,
        *,
        part_number: int,
        sha256: str,
        size_bytes: int,
        workspace: str | None = None,
    ) -> PreparedUploadPart:
        return cast(
            PreparedUploadPart,
            self._transport.request(
                "POST",
                f"/knowledge-base/uploads/{upload_id}/parts",
                params=_query(workspace=workspace),
                json={"partNumber": part_number, "sha256": sha256, "sizeBytes": size_bytes},
                model=PreparedUploadPart,
            ),
        )

    def confirm_part(
        self,
        upload_id: str,
        *,
        part_number: int,
        receipt: str,
        workspace: str | None = None,
    ) -> KnowledgeBaseUpload:
        return cast(
            KnowledgeBaseUpload,
            self._transport.request(
                "PUT",
                f"/knowledge-base/uploads/{upload_id}/parts/{part_number}",
                params=_query(workspace=workspace),
                json={"receipt": receipt},
                model=KnowledgeBaseUpload,
            ),
        )

    def upload_part(
        self,
        upload_id: str,
        *,
        part_number: int,
        data: bytes | bytearray | memoryview,
        workspace: str | None = None,
    ) -> KnowledgeBaseUpload:
        content = bytes(data)
        prepared = self.prepare_part(
            upload_id,
            part_number=part_number,
            sha256=sha256(content).hexdigest(),
            size_bytes=len(content),
            workspace=workspace,
        )
        receipt = self._transport.upload(
            prepared.upload_url, headers=prepared.headers, content=content
        )
        return self.confirm_part(
            upload_id,
            part_number=part_number,
            receipt=receipt,
            workspace=workspace,
        )

    def wait(
        self,
        upload_id: str,
        *,
        workspace: str | None = None,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> KnowledgeBaseUpload:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.get(upload_id, workspace=workspace)
            if result.status in _UPLOAD_TERMINAL:
                return result
            if deadline is not None and time.monotonic() + poll_interval >= deadline:
                raise ModaicTimeoutError("Timed out waiting for knowledge-base validation.")
            time.sleep(poll_interval)

    def upload_file(
        self,
        path: str | PathLike[str],
        *,
        kind: ResourceKind,
        name: str | None = None,
        workspace: str | None = None,
        wait_for_validation: bool = True,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        progress: UploadProgress | None = None,
    ) -> KnowledgeBaseUpload:
        """Upload a local file without materializing it in memory."""
        source = Path(path)
        digest, size = _file_sha256_and_size(source)
        upload = self.create(
            kind=kind,
            sha256=digest,
            size_bytes=size,
            name=name,
            workspace=workspace,
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
                        upload.id,
                        part_number=part_number,
                        data=chunk,
                        workspace=workspace,
                    )
                    uploaded += len(chunk)
                    if progress is not None:
                        progress(uploaded, size)
                    part_number += 1
        except Exception:
            with suppress(Exception):
                self.cancel(upload.id, workspace=workspace)
            raise

        result = self.complete(upload.id, workspace=workspace)
        if not wait_for_validation or result.status in _UPLOAD_TERMINAL:
            return result
        return self.wait(
            upload.id,
            workspace=workspace,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def complete(self, upload_id: str, *, workspace: str | None = None) -> KnowledgeBaseUpload:
        return cast(
            KnowledgeBaseUpload,
            self._transport.request(
                "POST",
                f"/knowledge-base/uploads/{upload_id}/complete",
                params=_query(workspace=workspace),
                model=KnowledgeBaseUpload,
            ),
        )

    def cancel(self, upload_id: str, *, workspace: str | None = None) -> KnowledgeBaseUpload:
        return cast(
            KnowledgeBaseUpload,
            self._transport.request(
                "DELETE",
                f"/knowledge-base/uploads/{upload_id}",
                params=_query(workspace=workspace),
                model=KnowledgeBaseUpload,
            ),
        )


class KnowledgeBaseResources:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def list(
        self,
        *,
        workspace: str | None = None,
        kind: ResourceKind | None = None,
        commit_sha: str | None = None,
        cursor: str | None = None,
    ) -> KnowledgeBaseResourcePage:
        return cast(
            KnowledgeBaseResourcePage,
            self._transport.request(
                "GET",
                "/knowledge-base/resources",
                params=_query(workspace=workspace, kind=kind, commitSha=commit_sha, cursor=cursor),
                model=KnowledgeBaseResourcePage,
            ),
        )

    def get(
        self, name: str, *, workspace: str | None = None, commit_sha: str | None = None
    ) -> KnowledgeBaseResource:
        return cast(
            KnowledgeBaseResource,
            self._transport.request(
                "GET",
                f"/knowledge-base/resources/{name}",
                params=_query(workspace=workspace, commitSha=commit_sha),
                model=KnowledgeBaseResource,
            ),
        )

    def download(
        self, name: str, *, workspace: str | None = None, commit_sha: str | None = None
    ) -> bytes:
        return self._transport.download(
            f"/knowledge-base/resources/{name}/content",
            params=_query(workspace=workspace, commitSha=commit_sha),
        )


class KnowledgeBase:
    def __init__(self, transport: SyncTransport) -> None:
        self.uploads = KnowledgeBaseUploads(transport)
        self.resources = KnowledgeBaseResources(transport)


class AsyncKnowledgeBaseUploads:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self,
        *,
        kind: ResourceKind,
        sha256: str,
        size_bytes: int,
        name: str | None = None,
        workspace: str | None = None,
    ) -> KnowledgeBaseUpload:
        body: dict[str, Any] = {"kind": kind, "sha256": sha256, "sizeBytes": size_bytes}
        if name is not None:
            body["name"] = name
        if workspace is not None:
            body["workspace"] = workspace
        return cast(
            KnowledgeBaseUpload,
            await self._transport.request(
                "POST", "/knowledge-base/uploads", json=body, model=KnowledgeBaseUpload
            ),
        )

    async def get(self, upload_id: str, *, workspace: str | None = None) -> KnowledgeBaseUpload:
        return cast(
            KnowledgeBaseUpload,
            await self._transport.request(
                "GET",
                f"/knowledge-base/uploads/{upload_id}",
                params=_query(workspace=workspace),
                model=KnowledgeBaseUpload,
            ),
        )

    async def prepare_part(
        self,
        upload_id: str,
        *,
        part_number: int,
        sha256: str,
        size_bytes: int,
        workspace: str | None = None,
    ) -> PreparedUploadPart:
        return cast(
            PreparedUploadPart,
            await self._transport.request(
                "POST",
                f"/knowledge-base/uploads/{upload_id}/parts",
                params=_query(workspace=workspace),
                json={"partNumber": part_number, "sha256": sha256, "sizeBytes": size_bytes},
                model=PreparedUploadPart,
            ),
        )

    async def confirm_part(
        self,
        upload_id: str,
        *,
        part_number: int,
        receipt: str,
        workspace: str | None = None,
    ) -> KnowledgeBaseUpload:
        return cast(
            KnowledgeBaseUpload,
            await self._transport.request(
                "PUT",
                f"/knowledge-base/uploads/{upload_id}/parts/{part_number}",
                params=_query(workspace=workspace),
                json={"receipt": receipt},
                model=KnowledgeBaseUpload,
            ),
        )

    async def upload_part(
        self,
        upload_id: str,
        *,
        part_number: int,
        data: bytes | bytearray | memoryview,
        workspace: str | None = None,
    ) -> KnowledgeBaseUpload:
        content = bytes(data)
        prepared = await self.prepare_part(
            upload_id,
            part_number=part_number,
            sha256=sha256(content).hexdigest(),
            size_bytes=len(content),
            workspace=workspace,
        )
        receipt = await self._transport.upload(
            prepared.upload_url, headers=prepared.headers, content=content
        )
        return await self.confirm_part(
            upload_id,
            part_number=part_number,
            receipt=receipt,
            workspace=workspace,
        )

    async def wait(
        self,
        upload_id: str,
        *,
        workspace: str | None = None,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> KnowledgeBaseUpload:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = await self.get(upload_id, workspace=workspace)
            if result.status in _UPLOAD_TERMINAL:
                return result
            if deadline is not None and time.monotonic() + poll_interval >= deadline:
                raise ModaicTimeoutError("Timed out waiting for knowledge-base validation.")
            await asyncio.sleep(poll_interval)

    async def upload_file(
        self,
        path: str | PathLike[str],
        *,
        kind: ResourceKind,
        name: str | None = None,
        workspace: str | None = None,
        wait_for_validation: bool = True,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        progress: UploadProgress | None = None,
    ) -> KnowledgeBaseUpload:
        """Upload a local file without blocking the event loop on file reads."""
        source = Path(path)
        digest, size = await asyncio.to_thread(_file_sha256_and_size, source)
        upload = await self.create(
            kind=kind,
            sha256=digest,
            size_bytes=size,
            name=name,
            workspace=workspace,
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
                    upload.id,
                    part_number=part_number,
                    data=chunk,
                    workspace=workspace,
                )
                uploaded += len(chunk)
                if progress is not None:
                    progress(uploaded, size)
                part_number += 1
        except Exception:
            with suppress(Exception):
                await self.cancel(upload.id, workspace=workspace)
            raise
        finally:
            if stream is not None:
                await asyncio.to_thread(stream.close)

        result = await self.complete(upload.id, workspace=workspace)
        if not wait_for_validation or result.status in _UPLOAD_TERMINAL:
            return result
        return await self.wait(
            upload.id,
            workspace=workspace,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    async def complete(
        self, upload_id: str, *, workspace: str | None = None
    ) -> KnowledgeBaseUpload:
        return cast(
            KnowledgeBaseUpload,
            await self._transport.request(
                "POST",
                f"/knowledge-base/uploads/{upload_id}/complete",
                params=_query(workspace=workspace),
                model=KnowledgeBaseUpload,
            ),
        )

    async def cancel(self, upload_id: str, *, workspace: str | None = None) -> KnowledgeBaseUpload:
        return cast(
            KnowledgeBaseUpload,
            await self._transport.request(
                "DELETE",
                f"/knowledge-base/uploads/{upload_id}",
                params=_query(workspace=workspace),
                model=KnowledgeBaseUpload,
            ),
        )


class AsyncKnowledgeBaseResources:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def list(
        self,
        *,
        workspace: str | None = None,
        kind: ResourceKind | None = None,
        commit_sha: str | None = None,
        cursor: str | None = None,
    ) -> KnowledgeBaseResourcePage:
        return cast(
            KnowledgeBaseResourcePage,
            await self._transport.request(
                "GET",
                "/knowledge-base/resources",
                params=_query(workspace=workspace, kind=kind, commitSha=commit_sha, cursor=cursor),
                model=KnowledgeBaseResourcePage,
            ),
        )

    async def get(
        self, name: str, *, workspace: str | None = None, commit_sha: str | None = None
    ) -> KnowledgeBaseResource:
        return cast(
            KnowledgeBaseResource,
            await self._transport.request(
                "GET",
                f"/knowledge-base/resources/{name}",
                params=_query(workspace=workspace, commitSha=commit_sha),
                model=KnowledgeBaseResource,
            ),
        )

    async def download(
        self, name: str, *, workspace: str | None = None, commit_sha: str | None = None
    ) -> bytes:
        return await self._transport.download(
            f"/knowledge-base/resources/{name}/content",
            params=_query(workspace=workspace, commitSha=commit_sha),
        )


class AsyncKnowledgeBase:
    def __init__(self, transport: AsyncTransport) -> None:
        self.uploads = AsyncKnowledgeBaseUploads(transport)
        self.resources = AsyncKnowledgeBaseResources(transport)
