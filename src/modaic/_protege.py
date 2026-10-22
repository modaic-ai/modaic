from __future__ import annotations

import asyncio
import time
from typing import cast

from ._transport import AsyncTransport, SyncTransport
from .errors import ModaicTimeoutError
from .types import ProtegeJob

_TERMINAL = {"succeeded", "failed", "cancelled"}


class Protege:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, *, prompt: str, repo: str, eval_only_dataset: str | None = None) -> ProtegeJob:
        body = {"prompt": prompt, "repo": repo}
        if eval_only_dataset is not None:
            body["eval_only_dataset"] = eval_only_dataset
        return cast(
            ProtegeJob, self._transport.request("POST", "/protege", json=body, model=ProtegeJob)
        )

    def get(self, job_id: str) -> ProtegeJob:
        return cast(
            ProtegeJob, self._transport.request("GET", f"/protege/{job_id}", model=ProtegeJob)
        )

    def cancel(self, job_id: str) -> ProtegeJob:
        return cast(
            ProtegeJob, self._transport.request("DELETE", f"/protege/{job_id}", model=ProtegeJob)
        )

    def wait(
        self, job_id: str, *, poll_interval: float = 2.0, timeout: float = 3600.0
    ) -> ProtegeJob:
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job.status in _TERMINAL:
                return job
            if time.monotonic() >= deadline:
                raise ModaicTimeoutError(f"Timed out waiting for Protégé job {job_id}.")
            time.sleep(poll_interval)


class AsyncProtege:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self, *, prompt: str, repo: str, eval_only_dataset: str | None = None
    ) -> ProtegeJob:
        body = {"prompt": prompt, "repo": repo}
        if eval_only_dataset is not None:
            body["eval_only_dataset"] = eval_only_dataset
        return cast(
            ProtegeJob,
            await self._transport.request("POST", "/protege", json=body, model=ProtegeJob),
        )

    async def get(self, job_id: str) -> ProtegeJob:
        return cast(
            ProtegeJob,
            await self._transport.request("GET", f"/protege/{job_id}", model=ProtegeJob),
        )

    async def cancel(self, job_id: str) -> ProtegeJob:
        return cast(
            ProtegeJob,
            await self._transport.request("DELETE", f"/protege/{job_id}", model=ProtegeJob),
        )

    async def wait(
        self, job_id: str, *, poll_interval: float = 2.0, timeout: float = 3600.0
    ) -> ProtegeJob:
        deadline = time.monotonic() + timeout
        while True:
            job = await self.get(job_id)
            if job.status in _TERMINAL:
                return job
            if time.monotonic() >= deadline:
                raise ModaicTimeoutError(f"Timed out waiting for Protégé job {job_id}.")
            await asyncio.sleep(poll_interval)
