from __future__ import annotations

import httpx

from ._datasets import AsyncDatasets, AsyncUploads, Datasets, Uploads
from ._fine_tuning import AsyncFineTunes, FineTunes
from ._protege import AsyncProtege, Protege
from ._resources import (
    Alignments,
    AsyncAlignments,
    AsyncBatchDecisions,
    AsyncDecisions,
    AsyncExamples,
    AsyncIdentity,
    AsyncModels,
    BatchDecisions,
    Decisions,
    Examples,
    Identity,
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
        self.predictions = Decisions(self._transport)
        self.decisions = self.predictions
        self.identity = Identity(self._transport)
        self.models = Models(self._transport)
        self.examples = Examples(self._transport)
        self.batch_predictions = BatchDecisions(self._transport)
        self.batch_decisions = self.batch_predictions
        self.instruction_optimizations = Alignments(self._transport)
        self.fine_tunes = FineTunes(self._transport)
        self.uploads = Uploads(self._transport)
        self.datasets = Datasets(self._transport)
        self.protege = Protege(self._transport)

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
        self.predictions = AsyncDecisions(self._transport)
        self.decisions = self.predictions
        self.identity = AsyncIdentity(self._transport)
        self.models = AsyncModels(self._transport)
        self.examples = AsyncExamples(self._transport)
        self.batch_predictions = AsyncBatchDecisions(self._transport)
        self.batch_decisions = self.batch_predictions
        self.instruction_optimizations = AsyncAlignments(self._transport)
        self.fine_tunes = AsyncFineTunes(self._transport)
        self.uploads = AsyncUploads(self._transport)
        self.datasets = AsyncDatasets(self._transport)
        self.protege = AsyncProtege(self._transport)

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> AsyncModaic:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


ModaicClient = Modaic
AsyncModaicClient = AsyncModaic
