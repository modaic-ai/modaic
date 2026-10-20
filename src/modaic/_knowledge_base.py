from __future__ import annotations

from hashlib import sha256
from typing import Any, cast

from ._transport import AsyncTransport, SyncTransport
from .types import (
    KnowledgeBaseResource,
    KnowledgeBaseResourcePage,
    KnowledgeBaseUpload,
    PreparedUploadPart,
    ResourceKind,
)


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
