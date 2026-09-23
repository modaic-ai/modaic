from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, NotRequired, Required, TypedDict, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from pydantic import JsonValue as PydanticJsonValue

JsonValue = PydanticJsonValue


class NoulCriteria(TypedDict, total=False):
    true: JsonValue
    false: JsonValue


class NoulQuestion(TypedDict, total=False):
    type: Required[Literal["noul"]]
    instructions: JsonValue
    criteria: NoulCriteria | None


class ChoiceQuestion(TypedDict, total=False):
    type: Required[Literal["choice"]]
    instructions: JsonValue
    criteria: Required[dict[str, JsonValue]]


class ScoreQuestion(TypedDict, total=False):
    type: Required[Literal["score"]]
    instructions: JsonValue
    criteria: Required[list[JsonValue]]


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: JsonValue = None


class Noul(_Question):
    type: Literal["noul"] = "noul"
    criteria: dict[Literal["true", "false"], JsonValue] | None = None


class Choice(_Question):
    type: Literal["choice"] = "choice"
    criteria: dict[str, JsonValue] = Field(min_length=1)


class Score(_Question):
    type: Literal["score"] = "score"
    criteria: list[JsonValue] = Field(min_length=1)


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion | Noul | Choice | Score


class ExampleAnnotationInput(TypedDict):
    ground_truth: dict[str, JsonValue]
    ground_reasoning: NotRequired[str]


class ExampleInput(TypedDict):
    state: JsonValue
    id: NotRequired[str]
    annotation: NotRequired[ExampleAnnotationInput]


class InlineExample(TypedDict):
    state: JsonValue
    id: NotRequired[str]


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class TokenUsage(APIModel):
    input_tokens: int
    output_tokens: int


class NoulAnswer(APIModel):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(APIModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(APIModel):
    type: Literal["score"]
    score: float
    legend: dict[str, JsonValue]
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class DecisionResponse(APIModel):
    model: str
    answers: dict[str, Answer]
    usage: TokenUsage
    example_id: str | None = None
    decision_id: str | None = None
    checkpoint: int | None = None
    revision: str | None = None
    captured: bool | None = None
    request_id: str | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _named_answers(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not isinstance(value.get("answers"), dict):
            return value
        data = dict(value)
        for name, field in cls.model_fields.items():
            if name in DecisionResponse.model_fields:
                continue
            key = field.validation_alias
            if key is None:
                key = field.alias if field.alias is not None else name
            if not isinstance(key, str):
                raise ValueError("Named answers require a string alias")
            if key in DecisionResponse.model_fields:
                raise ValueError(f"Answer alias {key!r} conflicts with response metadata")
            # Named fields must come from answers, not a coincidental top-level key.
            data.pop(name, None)
            data.pop(key, None)
            if key in data["answers"]:
                data[key] = data["answers"][key]
        return data

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {key: value for key, value in self.answers.items() if isinstance(value, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {
            key: value for key, value in self.answers.items() if isinstance(value, ChoiceAnswer)
        }

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {key: value for key, value in self.answers.items() if isinstance(value, ScoreAnswer)}


class Entity(APIModel):
    id: str
    kind: Literal["user", "organization"]
    slug: str
    name: str
    description: str | None = None
    avatar_url: str | None = Field(default=None, alias="avatarUrl")


class ModelSummary(APIModel):
    name: str
    description: str
    type: Literal["base", "repository"]
    repository_id: str | None = None


class ModelList(APIModel):
    models: list[ModelSummary]


class ModelConfiguration(APIModel):
    schema_version: int | None = Field(default=None, alias="schemaVersion")
    model: str | None = None
    checkpoint: int | None = None
    questions: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    capture: dict[str, Any] | None = None


class CommitResult(APIModel):
    commit_sha: str = Field(alias="commitSha")
    branch: str
    previous_sha: str | None = Field(default=None, alias="previousSha")


DecisionClientT = TypeVar("DecisionClientT")
ExampleClientT = TypeVar("ExampleClientT")
JobClientT = TypeVar("JobClientT")


class _ModelMetadata(APIModel, Generic[DecisionClientT, ExampleClientT, JobClientT]):
    _decisions: DecisionClientT | None = PrivateAttr(default=None)
    _examples: ExampleClientT | None = PrivateAttr(default=None)
    _jobs: JobClientT | None = PrivateAttr(default=None)

    @property
    def decisions(self) -> DecisionClientT:
        if self._decisions is None:
            raise RuntimeError("Get or create this model through a Modaic client to run decisions.")
        return self._decisions

    def _bind_decisions(self, decisions: DecisionClientT) -> None:
        self._decisions = decisions

    @property
    def examples(self) -> ExampleClientT:
        if self._examples is None:
            raise RuntimeError(
                "Get or create this model through a Modaic client to access examples."
            )
        return self._examples

    @property
    def jobs(self) -> JobClientT:
        if self._jobs is None:
            raise RuntimeError("Get or create this model through a Modaic client to access jobs.")
        return self._jobs

    def _bind_resources(self, examples: ExampleClientT, jobs: JobClientT) -> None:
        self._examples = examples
        self._jobs = jobs

    id: str
    slug: str
    description: str | None
    default_branch: str = Field(alias="defaultBranch")
    visibility: Literal["private"]
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")


class CreatedModel(
    _ModelMetadata[DecisionClientT, ExampleClientT, JobClientT],
    Generic[DecisionClientT, ExampleClientT, JobClientT],
):
    workspace: str


class Model(
    _ModelMetadata[DecisionClientT, ExampleClientT, JobClientT],
    Generic[DecisionClientT, ExampleClientT, JobClientT],
):
    workspace: Entity
    configuration: ModelConfiguration | None = None
    commit: CommitResult | None = None
    # Set by ``models.update`` when the supplied model and questions already
    # matched the stored configuration: nothing was committed, ``commit``
    # points at the current head, and the checkpoint is unchanged.
    unchanged: bool | None = None


class Annotation(APIModel):
    ground_truth: dict[str, JsonValue] = Field(alias="groundTruth")
    ground_reasoning: str | None = Field(default=None, alias="groundReasoning")
    split: Literal["train", "test"]


class _StoredDecision(APIModel):
    id: str
    commit_sha: str = Field(alias="commitSha")
    model: str
    answers: dict[str, Any]
    request: dict[str, Any]
    confidence: float | None
    version: int
    checkpoint: int
    revision: str
    source: Literal["batch", "live"]
    image_urls: list[str] = Field(alias="imageUrls")
    created_at: datetime = Field(alias="createdAt")


class ExampleDecision(_StoredDecision):
    """Successful decision nested in an example; no exampleId, jobId, or error."""

    response: dict[str, Any]


class DecisionRecord(_StoredDecision):
    """Decision-history entry, including its example and any failure."""

    example_id: str = Field(alias="exampleId")
    response: dict[str, Any] | None
    job_id: str | None = Field(alias="jobId")
    error: str | None


class Example(APIModel):
    id: str
    state: JsonValue
    source: Literal["ingest", "live"]
    image_urls: list[str] = Field(alias="imageUrls")
    annotation: Annotation | None
    latest_decision: ExampleDecision | None = Field(alias="latestDecision")
    decision_count: int = Field(alias="decisionCount")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")


class ExampleIngestResponse(APIModel):
    examples: list[Example]


class ExamplePage(APIModel):
    items: list[Example]
    page: int
    page_size: int = Field(alias="pageSize")
    total: int
    total_pages: int = Field(alias="totalPages")


class DecisionList(APIModel):
    decisions: list[DecisionRecord]


JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class JobError(APIModel):
    code: str
    message: str


class BatchProgress(APIModel):
    total: int
    completed: int
    failed: int


class BatchDecision(APIModel):
    id: str
    repository_id: str = Field(alias="repositoryId")
    status: JobStatus
    phase: Literal["queued", "deciding", "done", "failed", "cancelled"]
    branch: str
    source_commit_sha: str = Field(alias="sourceCommitSha")
    progress: BatchProgress
    result: dict[str, Any] | None
    error: JobError | None
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    started_at: datetime | None = Field(alias="startedAt")
    finished_at: datetime | None = Field(alias="finishedAt")


class BatchDecisionList(APIModel):
    batch_decisions: list[BatchDecision] = Field(alias="batchDecisions")


class AlignmentProgress(APIModel):
    stage: Literal["starting", "optimizing", "scoring", "finishing", "committing", "done"]
    metric_calls: int | None = Field(default=None, alias="metricCalls")
    max_metric_calls: int | None = Field(default=None, alias="maxMetricCalls")
    iteration: int | None = None
    candidates: int | None = None
    initial_score: float | None = Field(default=None, alias="initialScore")
    best_score: float | None = Field(default=None, alias="bestScore")
    elapsed_seconds: float | None = Field(default=None, alias="elapsedSeconds")
    updated_at: str | None = Field(default=None, alias="updatedAt")


class Alignment(APIModel):
    id: str
    repository_id: str = Field(alias="repositoryId")
    status: JobStatus
    phase: Literal["queued", "optimizing", "done", "failed", "cancelled"]
    branch: str
    source_commit_sha: str = Field(alias="sourceCommitSha")
    result_commit_sha: str | None = Field(alias="resultCommitSha")
    result: dict[str, Any] | None
    progress: AlignmentProgress | None
    error: JobError | None
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    started_at: datetime | None = Field(alias="startedAt")
    finished_at: datetime | None = Field(alias="finishedAt")


class AlignmentList(APIModel):
    alignments: list[Alignment]


class AlignmentLogs(APIModel):
    logs: list[str]
    available: bool
    running: bool
