from __future__ import annotations

import httpx

from ._resources import (
    Alignments,
    AsyncAlignments,
    AsyncBatchDecisions,
    AsyncDecisions,
    AsyncExamples,
    AsyncModels,
    BatchDecisions,
    Decisions,
    Examples,
    Models,
)
from ._transport import AsyncTransport, SyncTransport


class Modaic:
    """Synchronous Modaic API client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._transport = SyncTransport(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            client=http_client,
        )
        self.decisions = Decisions(self._transport)
        self.models = Models(self._transport)
        self.examples = Examples(self._transport)
        self.batch_decisions = BatchDecisions(self._transport)
        self.alignments = Alignments(self._transport)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> Modaic:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncModaic:
    """Asynchronous Modaic API client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._transport = AsyncTransport(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            client=http_client,
        )
        self.decisions = AsyncDecisions(self._transport)
        self.models = AsyncModels(self._transport)
        self.examples = AsyncExamples(self._transport)
        self.batch_decisions = AsyncBatchDecisions(self._transport)
        self.alignments = AsyncAlignments(self._transport)

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> AsyncModaic:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


ModaicClient = Modaic
AsyncModaicClient = AsyncModaic
