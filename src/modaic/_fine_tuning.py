from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any, Literal, cast

from ._transport import AsyncTransport, SyncTransport
from .errors import ModaicTimeoutError
from .types import (
    Choice,
    FineTune,
    FineTuneList,
    FineTuneLogs,
    JsonValue,
    Noul,
    Question,
    Score,
    TrainingControls,
    TrainingType,
)

_TERMINAL = {"done", "failed", "cancelled"}


def _questions(questions: Mapping[str, Question | Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {"type": question.type, **question.model_dump(mode="json", exclude_unset=True)}
        if isinstance(question, (Noul, Choice, Score))
        else dict(question)
        for name, question in questions.items()
    }


def _body(
    *,
    model: str,
    training_file: str,
    task_id: str,
    training_type: TrainingType,
    destination: str | None,
    questions: Mapping[str, Question | Mapping[str, Any]] | None,
    branch: str | None,
    source_commit_sha: str | None,
    validation_file: str | None,
    suffix: str | None,
    from_checkpoint: str | None,
    controls: TrainingControls | None,
    lora_r: int | None,
    lora_alpha: float | None,
    lora_dropout: float | None,
    wandb_api_key: str | None,
    wandb_base_url: str | None,
    wandb_project_name: str | None,
    wandb_name: str | None,
    wandb_entity: str | None,
    train_vision: Literal[False] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "training_file": training_file,
        "taskId": task_id,
        "training_type": training_type,
    }
    if controls is not None:
        body.update(controls)
    optional: tuple[tuple[str, JsonValue | Mapping[str, Any]], ...] = tuple(
        (key, value)
        for key, value in (
            ("destination", destination),
            ("branch", branch),
            ("sourceCommitSha", source_commit_sha),
            ("validation_file", validation_file),
            ("suffix", suffix),
            ("from_checkpoint", from_checkpoint),
            ("lora_r", lora_r),
            ("lora_alpha", lora_alpha),
            ("lora_dropout", lora_dropout),
            ("wandb_api_key", wandb_api_key),
            ("wandb_base_url", wandb_base_url),
            ("wandb_project_name", wandb_project_name),
            ("wandb_name", wandb_name),
            ("wandb_entity", wandb_entity),
        )
        if value is not None
    )
    body.update(optional)
    if questions is not None:
        body["questions"] = _questions(questions)
    if train_vision is not None:
        body["multimodal_params"] = {"train_vision": train_vision}
    return body


class FineTunes:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(
        self,
        *,
        model: str,
        training_file: str,
        task_id: str,
        training_type: TrainingType = "head_only",
        destination: str | None = None,
        questions: Mapping[str, Question | Mapping[str, Any]] | None = None,
        branch: str | None = None,
        source_commit_sha: str | None = None,
        validation_file: str | None = None,
        suffix: str | None = None,
        from_checkpoint: str | None = None,
        controls: TrainingControls | None = None,
        lora_r: int | None = None,
        lora_alpha: float | None = None,
        lora_dropout: float | None = None,
        wandb_api_key: str | None = None,
        wandb_base_url: str | None = None,
        wandb_project_name: str | None = None,
        wandb_name: str | None = None,
        wandb_entity: str | None = None,
        train_vision: Literal[False] | None = None,
        idempotency_key: str | None = None,
    ) -> FineTune:
        return cast(
            FineTune,
            self._transport.request(
                "POST",
                "/fine-tunes",
                json=_body(
                    model=model,
                    training_file=training_file,
                    task_id=task_id,
                    training_type=training_type,
                    destination=destination,
                    questions=questions,
                    branch=branch,
                    source_commit_sha=source_commit_sha,
                    validation_file=validation_file,
                    suffix=suffix,
                    from_checkpoint=from_checkpoint,
                    controls=controls,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    wandb_api_key=wandb_api_key,
                    wandb_base_url=wandb_base_url,
                    wandb_project_name=wandb_project_name,
                    wandb_name=wandb_name,
                    wandb_entity=wandb_entity,
                    train_vision=train_vision,
                ),
                headers={"idempotency-key": idempotency_key} if idempotency_key else None,
                model=FineTune,
            ),
        )

    def list(self, *, workspace: str, model_id: str | None = None, limit: int = 20) -> FineTuneList:
        return cast(
            FineTuneList,
            self._transport.request(
                "GET",
                "/fine-tunes",
                params={"workspace": workspace, "modelId": model_id, "limit": limit},
                model=FineTuneList,
            ),
        )

    def get(self, job_id: str) -> FineTune:
        return cast(
            FineTune,
            self._transport.request("GET", f"/fine-tunes/{job_id}", model=FineTune),
        )

    def logs(self, job_id: str) -> FineTuneLogs:
        return cast(
            FineTuneLogs,
            self._transport.request("GET", f"/fine-tunes/{job_id}/logs", model=FineTuneLogs),
        )

    def cancel(self, job_id: str) -> None:
        self._transport.request("DELETE", f"/fine-tunes/{job_id}")

    def wait(
        self, job_id: str, *, poll_interval: float = 2.0, timeout: float | None = None
    ) -> FineTune:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.get(job_id)
            if result.status in _TERMINAL:
                return result
            if deadline is not None and time.monotonic() + poll_interval >= deadline:
                raise ModaicTimeoutError("Timed out waiting for fine-tune job.")
            time.sleep(poll_interval)


class AsyncFineTunes:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self,
        *,
        model: str,
        training_file: str,
        task_id: str,
        training_type: TrainingType = "head_only",
        destination: str | None = None,
        questions: Mapping[str, Question | Mapping[str, Any]] | None = None,
        branch: str | None = None,
        source_commit_sha: str | None = None,
        validation_file: str | None = None,
        suffix: str | None = None,
        from_checkpoint: str | None = None,
        controls: TrainingControls | None = None,
        lora_r: int | None = None,
        lora_alpha: float | None = None,
        lora_dropout: float | None = None,
        wandb_api_key: str | None = None,
        wandb_base_url: str | None = None,
        wandb_project_name: str | None = None,
        wandb_name: str | None = None,
        wandb_entity: str | None = None,
        train_vision: Literal[False] | None = None,
        idempotency_key: str | None = None,
    ) -> FineTune:
        return cast(
            FineTune,
            await self._transport.request(
                "POST",
                "/fine-tunes",
                json=_body(
                    model=model,
                    training_file=training_file,
                    task_id=task_id,
                    training_type=training_type,
                    destination=destination,
                    questions=questions,
                    branch=branch,
                    source_commit_sha=source_commit_sha,
                    validation_file=validation_file,
                    suffix=suffix,
                    from_checkpoint=from_checkpoint,
                    controls=controls,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    wandb_api_key=wandb_api_key,
                    wandb_base_url=wandb_base_url,
                    wandb_project_name=wandb_project_name,
                    wandb_name=wandb_name,
                    wandb_entity=wandb_entity,
                    train_vision=train_vision,
                ),
                headers={"idempotency-key": idempotency_key} if idempotency_key else None,
                model=FineTune,
            ),
        )

    async def list(
        self, *, workspace: str, model_id: str | None = None, limit: int = 20
    ) -> FineTuneList:
        return cast(
            FineTuneList,
            await self._transport.request(
                "GET",
                "/fine-tunes",
                params={"workspace": workspace, "modelId": model_id, "limit": limit},
                model=FineTuneList,
            ),
        )

    async def get(self, job_id: str) -> FineTune:
        return cast(
            FineTune,
            await self._transport.request("GET", f"/fine-tunes/{job_id}", model=FineTune),
        )

    async def logs(self, job_id: str) -> FineTuneLogs:
        return cast(
            FineTuneLogs,
            await self._transport.request("GET", f"/fine-tunes/{job_id}/logs", model=FineTuneLogs),
        )

    async def cancel(self, job_id: str) -> None:
        await self._transport.request("DELETE", f"/fine-tunes/{job_id}")

    async def wait(
        self, job_id: str, *, poll_interval: float = 2.0, timeout: float | None = None
    ) -> FineTune:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = await self.get(job_id)
            if result.status in _TERMINAL:
                return result
            if deadline is not None and time.monotonic() + poll_interval >= deadline:
                raise ModaicTimeoutError("Timed out waiting for fine-tune job.")
            await asyncio.sleep(poll_interval)
