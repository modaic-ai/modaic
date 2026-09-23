from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast
from urllib.parse import quote

from ._progress import JobProgress
from ._transport import AsyncTransport, SyncTransport
from .errors import ModaicTimeoutError
from .types import (
    Alignment,
    AlignmentList,
    AlignmentLogs,
    BatchDecision,
    BatchDecisionList,
    CreatedModel,
    DecisionList,
    DecisionResponse,
    Example,
    ExampleIngestResponse,
    ExampleInput,
    ExamplePage,
    InlineExample,
    JsonValue,
    Model,
    ModelList,
    Question,
)


class _Unset:
    pass


UNSET = _Unset()
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _segment(value: str) -> str:
    return quote(value, safe="")


def _headers(idempotency_key: str | None) -> dict[str, str] | None:
    return {"idempotency-key": idempotency_key} if idempotency_key else None


def _decision_payload(
    *,
    state: JsonValue,
    model: str,
    questions: Mapping[str, Question | Mapping[str, Any]] | _Unset,
    revision: str | _Unset,
    example_id: str | _Unset,
    capture: bool | _Unset,
) -> dict[str, Any]:
    body: dict[str, Any] = {"state": state, "model": model}
    if not isinstance(questions, _Unset):
        body["questions"] = questions
    if not isinstance(revision, _Unset):
        body["revision"] = revision
    if not isinstance(example_id, _Unset):
        body["example_id"] = example_id
    if not isinstance(capture, _Unset):
        body["capture"] = capture
    return body


def _example_payload(example: ExampleInput | Mapping[str, Any]) -> dict[str, Any]:
    body = dict(example)
    annotation = body.get("annotation")
    if isinstance(annotation, Mapping):
        converted = dict(annotation)
        if "ground_truth" in converted:
            converted["groundTruth"] = converted.pop("ground_truth")
        if "ground_reasoning" in converted:
            converted["groundReasoning"] = converted.pop("ground_reasoning")
        body["annotation"] = converted
    return body


def _validate_selection(
    example_ids: Sequence[str] | _Unset,
    examples: Sequence[InlineExample | Mapping[str, Any]] | _Unset,
    scope: str | _Unset,
) -> None:
    supplied = sum(not isinstance(value, _Unset) for value in (example_ids, examples, scope))
    if supplied != 1:
        raise ValueError("Provide exactly one of example_ids, examples, or scope='all'.")


class Decisions:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(
        self,
        *,
        state: JsonValue,
        model: str,
        questions: Mapping[str, Question | Mapping[str, Any]] | _Unset = UNSET,
        revision: str | _Unset = UNSET,
        example_id: str | _Unset = UNSET,
        capture: bool | _Unset = UNSET,
        idempotency_key: str | None = None,
    ) -> DecisionResponse:
        return cast(
            DecisionResponse,
            self._transport.request(
                "POST",
                "/decision",
                json=_decision_payload(
                    state=state,
                    model=model,
                    questions=questions,
                    revision=revision,
                    example_id=example_id,
                    capture=capture,
                ),
                headers=_headers(idempotency_key),
                model=DecisionResponse,
            ),
        )


class AsyncDecisions:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self,
        *,
        state: JsonValue,
        model: str,
        questions: Mapping[str, Question | Mapping[str, Any]] | _Unset = UNSET,
        revision: str | _Unset = UNSET,
        example_id: str | _Unset = UNSET,
        capture: bool | _Unset = UNSET,
        idempotency_key: str | None = None,
    ) -> DecisionResponse:
        return cast(
            DecisionResponse,
            await self._transport.request(
                "POST",
                "/decision",
                json=_decision_payload(
                    state=state,
                    model=model,
                    questions=questions,
                    revision=revision,
                    example_id=example_id,
                    capture=capture,
                ),
                headers=_headers(idempotency_key),
                model=DecisionResponse,
            ),
        )


class ModelDecisions:
    def __init__(self, decisions: Decisions, model: str) -> None:
        self._decisions = decisions
        self._model = model

    def create(
        self,
        *,
        state: JsonValue,
        questions: Mapping[str, Question | Mapping[str, Any]] | _Unset = UNSET,
        revision: str | _Unset = UNSET,
        example_id: str | _Unset = UNSET,
        capture: bool | _Unset = UNSET,
        idempotency_key: str | None = None,
    ) -> DecisionResponse:
        return self._decisions.create(
            state=state,
            model=self._model,
            questions=questions,
            revision=revision,
            example_id=example_id,
            capture=capture,
            idempotency_key=idempotency_key,
        )


class AsyncModelDecisions:
    def __init__(self, decisions: AsyncDecisions, model: str) -> None:
        self._decisions = decisions
        self._model = model

    async def create(
        self,
        *,
        state: JsonValue,
        questions: Mapping[str, Question | Mapping[str, Any]] | _Unset = UNSET,
        revision: str | _Unset = UNSET,
        example_id: str | _Unset = UNSET,
        capture: bool | _Unset = UNSET,
        idempotency_key: str | None = None,
    ) -> DecisionResponse:
        return await self._decisions.create(
            state=state,
            model=self._model,
            questions=questions,
            revision=revision,
            example_id=example_id,
            capture=capture,
            idempotency_key=idempotency_key,
        )


class Models:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def list(self) -> ModelList:
        return cast(ModelList, self._transport.request("GET", "/models", model=ModelList))

    def create(
        self,
        *,
        workspace: str,
        slug: str,
        description: str | _Unset = UNSET,
        default_branch: str | _Unset = UNSET,
        model: str | _Unset = UNSET,
        questions: Mapping[str, Question | Mapping[str, Any]] | _Unset = UNSET,
    ) -> CreatedModel[ModelDecisions, ModelExamples, ModelJobs]:
        body: dict[str, Any] = {"workspace": workspace, "slug": slug}
        if not isinstance(description, _Unset):
            body["description"] = description
        if not isinstance(default_branch, _Unset):
            body["defaultBranch"] = default_branch
        if not isinstance(model, _Unset):
            body["model"] = model
        if not isinstance(questions, _Unset):
            body["questions"] = questions
        result = cast(
            CreatedModel[ModelDecisions, ModelExamples, ModelJobs],
            self._transport.request("POST", "/models", json=body, model=CreatedModel),
        )
        result._bind_decisions(
            ModelDecisions(Decisions(self._transport), f"{result.workspace}/{result.slug}")
        )
        result._bind_resources(
            ModelExamples(Examples(self._transport), result.id),
            ModelJobs(self._transport, result.id),
        )
        return result

    def get(self, *, workspace: str, model: str) -> Model[ModelDecisions, ModelExamples, ModelJobs]:
        path = f"/entities/{_segment(workspace)}/models/{_segment(model)}"
        result = cast(
            Model[ModelDecisions, ModelExamples, ModelJobs],
            self._transport.request("GET", path, model=Model),
        )
        result._bind_decisions(
            ModelDecisions(Decisions(self._transport), f"{result.workspace.slug}/{result.slug}")
        )
        result._bind_resources(
            ModelExamples(Examples(self._transport), result.id),
            ModelJobs(self._transport, result.id),
        )
        return result

    def update(
        self,
        model_id: str,
        *,
        branch: str | None = None,
        description: str | None | _Unset = UNSET,
        model: str | _Unset = UNSET,
        questions: Mapping[str, Question | Mapping[str, Any]] | _Unset = UNSET,
        message: str | _Unset = UNSET,
    ) -> Model[ModelDecisions, ModelExamples, ModelJobs]:
        body: dict[str, Any] = {}
        for key, value in (
            ("description", description),
            ("model", model),
            ("questions", questions),
            ("message", message),
        ):
            if not isinstance(value, _Unset):
                body[key] = value
        params = {"branch": branch} if branch is not None else None
        result = cast(
            Model[ModelDecisions, ModelExamples, ModelJobs],
            self._transport.request(
                "PATCH", f"/models/{_segment(model_id)}", params=params, json=body, model=Model
            ),
        )
        result._bind_decisions(
            ModelDecisions(Decisions(self._transport), f"{result.workspace.slug}/{result.slug}")
        )
        result._bind_resources(
            ModelExamples(Examples(self._transport), result.id),
            ModelJobs(self._transport, result.id),
        )
        return result

    def delete(self, model_id: str) -> None:
        self._transport.request("DELETE", f"/models/{_segment(model_id)}")


class AsyncModels:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def list(self) -> ModelList:
        return cast(ModelList, await self._transport.request("GET", "/models", model=ModelList))

    async def create(
        self,
        *,
        workspace: str,
        slug: str,
        description: str | _Unset = UNSET,
        default_branch: str | _Unset = UNSET,
        model: str | _Unset = UNSET,
        questions: Mapping[str, Question | Mapping[str, Any]] | _Unset = UNSET,
    ) -> CreatedModel[AsyncModelDecisions, AsyncModelExamples, AsyncModelJobs]:
        body: dict[str, Any] = {"workspace": workspace, "slug": slug}
        if not isinstance(description, _Unset):
            body["description"] = description
        if not isinstance(default_branch, _Unset):
            body["defaultBranch"] = default_branch
        if not isinstance(model, _Unset):
            body["model"] = model
        if not isinstance(questions, _Unset):
            body["questions"] = questions
        result = cast(
            CreatedModel[AsyncModelDecisions, AsyncModelExamples, AsyncModelJobs],
            await self._transport.request("POST", "/models", json=body, model=CreatedModel),
        )
        result._bind_decisions(
            AsyncModelDecisions(
                AsyncDecisions(self._transport), f"{result.workspace}/{result.slug}"
            )
        )
        result._bind_resources(
            AsyncModelExamples(AsyncExamples(self._transport), result.id),
            AsyncModelJobs(self._transport, result.id),
        )
        return result

    async def get(
        self, *, workspace: str, model: str
    ) -> Model[AsyncModelDecisions, AsyncModelExamples, AsyncModelJobs]:
        path = f"/entities/{_segment(workspace)}/models/{_segment(model)}"
        result = cast(
            Model[AsyncModelDecisions, AsyncModelExamples, AsyncModelJobs],
            await self._transport.request("GET", path, model=Model),
        )
        result._bind_decisions(
            AsyncModelDecisions(
                AsyncDecisions(self._transport), f"{result.workspace.slug}/{result.slug}"
            )
        )
        result._bind_resources(
            AsyncModelExamples(AsyncExamples(self._transport), result.id),
            AsyncModelJobs(self._transport, result.id),
        )
        return result

    async def update(
        self,
        model_id: str,
        *,
        branch: str | None = None,
        description: str | None | _Unset = UNSET,
        model: str | _Unset = UNSET,
        questions: Mapping[str, Question | Mapping[str, Any]] | _Unset = UNSET,
        message: str | _Unset = UNSET,
    ) -> Model[AsyncModelDecisions, AsyncModelExamples, AsyncModelJobs]:
        body: dict[str, Any] = {}
        for key, value in (
            ("description", description),
            ("model", model),
            ("questions", questions),
            ("message", message),
        ):
            if not isinstance(value, _Unset):
                body[key] = value
        params = {"branch": branch} if branch is not None else None
        result = cast(
            Model[AsyncModelDecisions, AsyncModelExamples, AsyncModelJobs],
            await self._transport.request(
                "PATCH", f"/models/{_segment(model_id)}", params=params, json=body, model=Model
            ),
        )
        result._bind_decisions(
            AsyncModelDecisions(
                AsyncDecisions(self._transport), f"{result.workspace.slug}/{result.slug}"
            )
        )
        result._bind_resources(
            AsyncModelExamples(AsyncExamples(self._transport), result.id),
            AsyncModelJobs(self._transport, result.id),
        )
        return result

    async def delete(self, model_id: str) -> None:
        await self._transport.request("DELETE", f"/models/{_segment(model_id)}")


class Examples:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def ingest(
        self, model_id: str, *, examples: Sequence[ExampleInput | Mapping[str, Any]]
    ) -> ExampleIngestResponse:
        body = {"examples": [_example_payload(example) for example in examples]}
        return cast(
            ExampleIngestResponse,
            self._transport.request(
                "POST",
                f"/models/{_segment(model_id)}/examples",
                json=body,
                model=ExampleIngestResponse,
            ),
        )

    def list(self, model_id: str, *, page: int = 1, page_size: int = 30) -> ExamplePage:
        return cast(
            ExamplePage,
            self._transport.request(
                "GET",
                f"/models/{_segment(model_id)}/examples",
                params={"page": page, "pageSize": page_size},
                model=ExamplePage,
            ),
        )

    def get(self, model_id: str, example_id: str) -> Example:
        path = f"/models/{_segment(model_id)}/examples/{_segment(example_id)}"
        return cast(Example, self._transport.request("GET", path, model=Example))

    def annotate(
        self,
        model_id: str,
        example_id: str,
        *,
        ground_truth: Mapping[str, JsonValue] | _Unset = UNSET,
        ground_reasoning: str | _Unset = UNSET,
    ) -> Example:
        body: dict[str, Any] = {}
        if not isinstance(ground_truth, _Unset):
            body["groundTruth"] = ground_truth
        if not isinstance(ground_reasoning, _Unset):
            body["groundReasoning"] = ground_reasoning
        path = f"/models/{_segment(model_id)}/examples/{_segment(example_id)}/annotation"
        return cast(Example, self._transport.request("PATCH", path, json=body, model=Example))

    def list_decisions(self, model_id: str, example_id: str) -> DecisionList:
        path = f"/models/{_segment(model_id)}/examples/{_segment(example_id)}/decisions"
        return cast(DecisionList, self._transport.request("GET", path, model=DecisionList))


class AsyncExamples:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def ingest(
        self, model_id: str, *, examples: Sequence[ExampleInput | Mapping[str, Any]]
    ) -> ExampleIngestResponse:
        body = {"examples": [_example_payload(example) for example in examples]}
        return cast(
            ExampleIngestResponse,
            await self._transport.request(
                "POST",
                f"/models/{_segment(model_id)}/examples",
                json=body,
                model=ExampleIngestResponse,
            ),
        )

    async def list(self, model_id: str, *, page: int = 1, page_size: int = 30) -> ExamplePage:
        return cast(
            ExamplePage,
            await self._transport.request(
                "GET",
                f"/models/{_segment(model_id)}/examples",
                params={"page": page, "pageSize": page_size},
                model=ExamplePage,
            ),
        )

    async def get(self, model_id: str, example_id: str) -> Example:
        path = f"/models/{_segment(model_id)}/examples/{_segment(example_id)}"
        return cast(Example, await self._transport.request("GET", path, model=Example))

    async def annotate(
        self,
        model_id: str,
        example_id: str,
        *,
        ground_truth: Mapping[str, JsonValue] | _Unset = UNSET,
        ground_reasoning: str | _Unset = UNSET,
    ) -> Example:
        body: dict[str, Any] = {}
        if not isinstance(ground_truth, _Unset):
            body["groundTruth"] = ground_truth
        if not isinstance(ground_reasoning, _Unset):
            body["groundReasoning"] = ground_reasoning
        path = f"/models/{_segment(model_id)}/examples/{_segment(example_id)}/annotation"
        return cast(Example, await self._transport.request("PATCH", path, json=body, model=Example))

    async def list_decisions(self, model_id: str, example_id: str) -> DecisionList:
        path = f"/models/{_segment(model_id)}/examples/{_segment(example_id)}/decisions"
        return cast(DecisionList, await self._transport.request("GET", path, model=DecisionList))


class ModelExamples:
    def __init__(self, examples: Examples, model_id: str) -> None:
        self._examples = examples
        self._model_id = model_id

    def ingest(
        self, *, examples: Sequence[ExampleInput | Mapping[str, Any]]
    ) -> ExampleIngestResponse:
        return self._examples.ingest(self._model_id, examples=examples)

    def list(self, *, page: int = 1, page_size: int = 30) -> ExamplePage:
        return self._examples.list(self._model_id, page=page, page_size=page_size)

    def get(self, example_id: str) -> Example:
        return self._examples.get(self._model_id, example_id)

    def annotate(
        self,
        example_id: str,
        *,
        ground_truth: Mapping[str, JsonValue] | _Unset = UNSET,
        ground_reasoning: str | _Unset = UNSET,
    ) -> Example:
        return self._examples.annotate(
            self._model_id,
            example_id,
            ground_truth=ground_truth,
            ground_reasoning=ground_reasoning,
        )

    def list_decisions(self, example_id: str) -> DecisionList:
        return self._examples.list_decisions(self._model_id, example_id)


class AsyncModelExamples:
    def __init__(self, examples: AsyncExamples, model_id: str) -> None:
        self._examples = examples
        self._model_id = model_id

    async def ingest(
        self, *, examples: Sequence[ExampleInput | Mapping[str, Any]]
    ) -> ExampleIngestResponse:
        return await self._examples.ingest(self._model_id, examples=examples)

    async def list(self, *, page: int = 1, page_size: int = 30) -> ExamplePage:
        return await self._examples.list(self._model_id, page=page, page_size=page_size)

    async def get(self, example_id: str) -> Example:
        return await self._examples.get(self._model_id, example_id)

    async def annotate(
        self,
        example_id: str,
        *,
        ground_truth: Mapping[str, JsonValue] | _Unset = UNSET,
        ground_reasoning: str | _Unset = UNSET,
    ) -> Example:
        return await self._examples.annotate(
            self._model_id,
            example_id,
            ground_truth=ground_truth,
            ground_reasoning=ground_reasoning,
        )

    async def list_decisions(self, example_id: str) -> DecisionList:
        return await self._examples.list_decisions(self._model_id, example_id)


class ModelBatchDecisions:
    def __init__(self, jobs: BatchDecisions, model_id: str) -> None:
        self._jobs = jobs
        self._model_id = model_id

    def create(
        self,
        *,
        idempotency_key: str,
        branch: str = "main",
        source_commit_sha: str | _Unset = UNSET,
        example_ids: Sequence[str] | _Unset = UNSET,
        examples: Sequence[InlineExample | Mapping[str, Any]] | _Unset = UNSET,
        scope: Literal["all"] | _Unset = UNSET,
    ) -> BatchDecision:
        return self._jobs.create(
            self._model_id,
            idempotency_key=idempotency_key,
            branch=branch,
            source_commit_sha=source_commit_sha,
            example_ids=example_ids,
            examples=examples,
            scope=scope,
        )

    def list(self, *, limit: int = 20) -> BatchDecisionList:
        return self._jobs.list(self._model_id, limit=limit)


class AsyncModelBatchDecisions:
    def __init__(self, jobs: AsyncBatchDecisions, model_id: str) -> None:
        self._jobs = jobs
        self._model_id = model_id

    async def create(
        self,
        *,
        idempotency_key: str,
        branch: str = "main",
        source_commit_sha: str | _Unset = UNSET,
        example_ids: Sequence[str] | _Unset = UNSET,
        examples: Sequence[InlineExample | Mapping[str, Any]] | _Unset = UNSET,
        scope: Literal["all"] | _Unset = UNSET,
    ) -> BatchDecision:
        return await self._jobs.create(
            self._model_id,
            idempotency_key=idempotency_key,
            branch=branch,
            source_commit_sha=source_commit_sha,
            example_ids=example_ids,
            examples=examples,
            scope=scope,
        )

    async def list(self, *, limit: int = 20) -> BatchDecisionList:
        return await self._jobs.list(self._model_id, limit=limit)


class ModelAlignments:
    def __init__(self, jobs: Alignments, model_id: str) -> None:
        self._jobs = jobs
        self._model_id = model_id

    def create(
        self,
        *,
        branch: str,
        source_commit_sha: str,
        max_metric_calls: int,
        idempotency_key: str,
        reflection_model: str | _Unset = UNSET,
        reflection_minibatch_size: int | _Unset = UNSET,
        seed: int = 0,
    ) -> Alignment:
        return self._jobs.create(
            self._model_id,
            branch=branch,
            source_commit_sha=source_commit_sha,
            max_metric_calls=max_metric_calls,
            idempotency_key=idempotency_key,
            reflection_model=reflection_model,
            reflection_minibatch_size=reflection_minibatch_size,
            seed=seed,
        )

    def list(self, *, limit: int = 20) -> AlignmentList:
        return self._jobs.list(self._model_id, limit=limit)


class AsyncModelAlignments:
    def __init__(self, jobs: AsyncAlignments, model_id: str) -> None:
        self._jobs = jobs
        self._model_id = model_id

    async def create(
        self,
        *,
        branch: str,
        source_commit_sha: str,
        max_metric_calls: int,
        idempotency_key: str,
        reflection_model: str | _Unset = UNSET,
        reflection_minibatch_size: int | _Unset = UNSET,
        seed: int = 0,
    ) -> Alignment:
        return await self._jobs.create(
            self._model_id,
            branch=branch,
            source_commit_sha=source_commit_sha,
            max_metric_calls=max_metric_calls,
            idempotency_key=idempotency_key,
            reflection_model=reflection_model,
            reflection_minibatch_size=reflection_minibatch_size,
            seed=seed,
        )

    async def list(self, *, limit: int = 20) -> AlignmentList:
        return await self._jobs.list(self._model_id, limit=limit)


class ModelJobs:
    def __init__(self, transport: SyncTransport, model_id: str) -> None:
        self.alignments = ModelAlignments(Alignments(transport), model_id)
        self.batch_decisions = ModelBatchDecisions(BatchDecisions(transport), model_id)


class AsyncModelJobs:
    def __init__(self, transport: AsyncTransport, model_id: str) -> None:
        self.alignments = AsyncModelAlignments(AsyncAlignments(transport), model_id)
        self.batch_decisions = AsyncModelBatchDecisions(AsyncBatchDecisions(transport), model_id)


class BatchDecisions:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(
        self,
        model_id: str,
        *,
        idempotency_key: str,
        branch: str = "main",
        source_commit_sha: str | _Unset = UNSET,
        example_ids: Sequence[str] | _Unset = UNSET,
        examples: Sequence[InlineExample | Mapping[str, Any]] | _Unset = UNSET,
        scope: Literal["all"] | _Unset = UNSET,
    ) -> BatchDecision:
        _validate_selection(example_ids, examples, scope)
        body: dict[str, Any] = {"branch": branch}
        if not isinstance(source_commit_sha, _Unset):
            body["sourceCommitSha"] = source_commit_sha
        if not isinstance(example_ids, _Unset):
            body["exampleIds"] = list(example_ids)
        if not isinstance(examples, _Unset):
            body["examples"] = list(examples)
        if not isinstance(scope, _Unset):
            body["scope"] = scope
        return cast(
            BatchDecision,
            self._transport.request(
                "POST",
                f"/models/{_segment(model_id)}/batch-decisions",
                json=body,
                headers=_headers(idempotency_key),
                model=BatchDecision,
            ),
        )

    def list(self, model_id: str, *, limit: int = 20) -> BatchDecisionList:
        return cast(
            BatchDecisionList,
            self._transport.request(
                "GET",
                f"/models/{_segment(model_id)}/batch-decisions",
                params={"limit": limit},
                model=BatchDecisionList,
            ),
        )

    def get(self, batch_decision_id: str) -> BatchDecision:
        return cast(
            BatchDecision,
            self._transport.request(
                "GET", f"/batch-decisions/{_segment(batch_decision_id)}", model=BatchDecision
            ),
        )

    def cancel(self, batch_decision_id: str) -> None:
        self._transport.request("DELETE", f"/batch-decisions/{_segment(batch_decision_id)}")

    def wait(
        self,
        batch_decision_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
        progress: bool = False,
    ) -> BatchDecision:
        deadline = time.monotonic() + timeout
        with JobProgress(progress) as display:
            while True:
                job = self.get(batch_decision_id)
                display.update(job)
                if job.status in TERMINAL_STATUSES:
                    return job
                if time.monotonic() >= deadline:
                    raise ModaicTimeoutError(
                        f"Timed out waiting for batch decision {batch_decision_id}."
                    )
                time.sleep(poll_interval)


class AsyncBatchDecisions:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self,
        model_id: str,
        *,
        idempotency_key: str,
        branch: str = "main",
        source_commit_sha: str | _Unset = UNSET,
        example_ids: Sequence[str] | _Unset = UNSET,
        examples: Sequence[InlineExample | Mapping[str, Any]] | _Unset = UNSET,
        scope: Literal["all"] | _Unset = UNSET,
    ) -> BatchDecision:
        _validate_selection(example_ids, examples, scope)
        body: dict[str, Any] = {"branch": branch}
        if not isinstance(source_commit_sha, _Unset):
            body["sourceCommitSha"] = source_commit_sha
        if not isinstance(example_ids, _Unset):
            body["exampleIds"] = list(example_ids)
        if not isinstance(examples, _Unset):
            body["examples"] = list(examples)
        if not isinstance(scope, _Unset):
            body["scope"] = scope
        return cast(
            BatchDecision,
            await self._transport.request(
                "POST",
                f"/models/{_segment(model_id)}/batch-decisions",
                json=body,
                headers=_headers(idempotency_key),
                model=BatchDecision,
            ),
        )

    async def list(self, model_id: str, *, limit: int = 20) -> BatchDecisionList:
        return cast(
            BatchDecisionList,
            await self._transport.request(
                "GET",
                f"/models/{_segment(model_id)}/batch-decisions",
                params={"limit": limit},
                model=BatchDecisionList,
            ),
        )

    async def get(self, batch_decision_id: str) -> BatchDecision:
        return cast(
            BatchDecision,
            await self._transport.request(
                "GET", f"/batch-decisions/{_segment(batch_decision_id)}", model=BatchDecision
            ),
        )

    async def cancel(self, batch_decision_id: str) -> None:
        await self._transport.request("DELETE", f"/batch-decisions/{_segment(batch_decision_id)}")

    async def wait(
        self,
        batch_decision_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
        progress: bool = False,
    ) -> BatchDecision:
        deadline = time.monotonic() + timeout
        with JobProgress(progress) as display:
            while True:
                job = await self.get(batch_decision_id)
                display.update(job)
                if job.status in TERMINAL_STATUSES:
                    return job
                if time.monotonic() >= deadline:
                    raise ModaicTimeoutError(
                        f"Timed out waiting for batch decision {batch_decision_id}."
                    )
                await asyncio.sleep(poll_interval)


class Alignments:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(
        self,
        model_id: str,
        *,
        branch: str,
        source_commit_sha: str,
        max_metric_calls: int,
        idempotency_key: str,
        reflection_model: str | _Unset = UNSET,
        reflection_minibatch_size: int | _Unset = UNSET,
        seed: int = 0,
    ) -> Alignment:
        reflection: dict[str, Any] = {"seed": seed}
        if not isinstance(reflection_model, _Unset):
            reflection["model"] = reflection_model
        if not isinstance(reflection_minibatch_size, _Unset):
            reflection["minibatchSize"] = reflection_minibatch_size
        body = {
            "branch": branch,
            "sourceCommitSha": source_commit_sha,
            "budget": {"maxMetricCalls": max_metric_calls},
            "reflection": reflection,
        }
        return cast(
            Alignment,
            self._transport.request(
                "POST",
                f"/models/{_segment(model_id)}/alignments",
                json=body,
                headers=_headers(idempotency_key),
                model=Alignment,
            ),
        )

    def list(self, model_id: str, *, limit: int = 20) -> AlignmentList:
        return cast(
            AlignmentList,
            self._transport.request(
                "GET",
                f"/models/{_segment(model_id)}/alignments",
                params={"limit": limit},
                model=AlignmentList,
            ),
        )

    def get(self, alignment_id: str) -> Alignment:
        return cast(
            Alignment,
            self._transport.request(
                "GET", f"/alignments/{_segment(alignment_id)}", model=Alignment
            ),
        )

    def logs(self, alignment_id: str) -> AlignmentLogs:
        return cast(
            AlignmentLogs,
            self._transport.request(
                "GET", f"/alignments/{_segment(alignment_id)}/logs", model=AlignmentLogs
            ),
        )

    def cancel(self, alignment_id: str) -> None:
        self._transport.request("DELETE", f"/alignments/{_segment(alignment_id)}")

    def wait(
        self,
        alignment_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float = 900.0,
        progress: bool = False,
    ) -> Alignment:
        deadline = time.monotonic() + timeout
        with JobProgress(progress) as display:
            while True:
                alignment = self.get(alignment_id)
                display.update(alignment)
                if alignment.status in TERMINAL_STATUSES:
                    return alignment
                if time.monotonic() >= deadline:
                    raise ModaicTimeoutError(f"Timed out waiting for alignment {alignment_id}.")
                time.sleep(poll_interval)


class AsyncAlignments:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self,
        model_id: str,
        *,
        branch: str,
        source_commit_sha: str,
        max_metric_calls: int,
        idempotency_key: str,
        reflection_model: str | _Unset = UNSET,
        reflection_minibatch_size: int | _Unset = UNSET,
        seed: int = 0,
    ) -> Alignment:
        reflection: dict[str, Any] = {"seed": seed}
        if not isinstance(reflection_model, _Unset):
            reflection["model"] = reflection_model
        if not isinstance(reflection_minibatch_size, _Unset):
            reflection["minibatchSize"] = reflection_minibatch_size
        body = {
            "branch": branch,
            "sourceCommitSha": source_commit_sha,
            "budget": {"maxMetricCalls": max_metric_calls},
            "reflection": reflection,
        }
        return cast(
            Alignment,
            await self._transport.request(
                "POST",
                f"/models/{_segment(model_id)}/alignments",
                json=body,
                headers=_headers(idempotency_key),
                model=Alignment,
            ),
        )

    async def list(self, model_id: str, *, limit: int = 20) -> AlignmentList:
        return cast(
            AlignmentList,
            await self._transport.request(
                "GET",
                f"/models/{_segment(model_id)}/alignments",
                params={"limit": limit},
                model=AlignmentList,
            ),
        )

    async def get(self, alignment_id: str) -> Alignment:
        return cast(
            Alignment,
            await self._transport.request(
                "GET", f"/alignments/{_segment(alignment_id)}", model=Alignment
            ),
        )

    async def logs(self, alignment_id: str) -> AlignmentLogs:
        return cast(
            AlignmentLogs,
            await self._transport.request(
                "GET", f"/alignments/{_segment(alignment_id)}/logs", model=AlignmentLogs
            ),
        )

    async def cancel(self, alignment_id: str) -> None:
        await self._transport.request("DELETE", f"/alignments/{_segment(alignment_id)}")

    async def wait(
        self,
        alignment_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float = 900.0,
        progress: bool = False,
    ) -> Alignment:
        deadline = time.monotonic() + timeout
        with JobProgress(progress) as display:
            while True:
                alignment = await self.get(alignment_id)
                display.update(alignment)
                if alignment.status in TERMINAL_STATUSES:
                    return alignment
                if time.monotonic() >= deadline:
                    raise ModaicTimeoutError(f"Timed out waiting for alignment {alignment_id}.")
                await asyncio.sleep(poll_interval)
