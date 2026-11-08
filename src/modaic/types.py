from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    NotRequired,
    Required,
    Self,
    TypedDict,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    TypeAdapter,
    create_model,
    model_serializer,
    model_validator,
)
from pydantic import JsonValue as PydanticJsonValue
from typing_extensions import TypeVar as DefaultTypeVar

JsonValue = PydanticJsonValue
ReasoningEffort = Literal["low", "medium", "high", "xhigh"]


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


class ExtractionQuestion(TypedDict):
    type: Literal["extraction"]
    instructions: str


class BoxQuestion(TypedDict):
    type: Literal["box"]
    instructions: str


class ScoreQuestion(TypedDict, total=False):
    type: Required[Literal["score"]]
    instructions: JsonValue
    criteria: Required[list[JsonValue]]


DatetimeFormat = Literal[
    "YYYY",
    "YYYY-MM",
    "YYYY-MM-DD",
    "HH",
    "HH:MM",
    "HH:MM:SS",
    "YYYY-MM-DDTHH",
    "YYYY-MM-DDTHH:MM",
    "YYYY-MM-DDTHH:MM:SS",
]
DATETIME_FORMATS = (
    "YYYY",
    "YYYY-MM",
    "YYYY-MM-DD",
    "HH",
    "HH:MM",
    "HH:MM:SS",
    "YYYY-MM-DDTHH",
    "YYYY-MM-DDTHH:MM",
    "YYYY-MM-DDTHH:MM:SS",
)


def _question_field(
    *,
    instructions: Any = None,
    criteria: Any = None,
    nullable: float | None = None,
    min: int | float | None = None,
    max: int | float | None = None,
    step: int | float | None = None,
    type: Literal["score", "text", "span", "datetime"] | None = None,
    format: DatetimeFormat | None = None,
) -> Any:
    """Configure a signature field. Optional annotations default to a 0.5 gate."""
    if nullable is not None and (
        isinstance(nullable, bool) or not math.isfinite(nullable) or not 0 <= nullable <= 1
    ):
        raise ValueError("nullable must be a finite threshold between 0 and 1")
    return Field(
        json_schema_extra={
            "modaic": dict(
                instructions=instructions,
                criteria=criteria,
                nullable=nullable,
                min=min,
                max=max,
                step=step,
                type=type,
                format=format,
            )
        }
    )


class Question:
    """Signature field factory, also a common type for question constructors."""

    def __new__(cls, **kwargs: Any) -> Any:
        if cls is Question:
            return _question_field(**kwargs)
        return super().__new__(cls)


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class _AnswerModel(APIModel):
    name: str | None = Field(default=None, exclude=True)
    instructions: JsonValue = Field(default=None, exclude=True)


class NoulAnswer(_AnswerModel):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(_AnswerModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(_AnswerModel):
    type: Literal["score"]
    score: float
    legend: dict[int, JsonValue]
    probabilities: dict[int, float]
    confidence: float


class TextAnswer(_AnswerModel):
    type: Literal["text"]
    text: str
    confidence: float


class NumberAnswer(_AnswerModel):
    type: Literal["number"]
    value: float
    probabilities: dict[str, float]
    variance: float


class SpanAnswer(_AnswerModel):
    type: Literal["span"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    span: str = Field(min_length=1)
    confidence: float

    @model_validator(mode="after")
    def _ordered(self) -> SpanAnswer:
        if self.start >= self.end:
            raise ValueError("Span start must be smaller than end")
        return self


class ExtractionValue(APIModel):
    start: int
    end: int
    text: str


class ExtractionAnswer(_AnswerModel):
    type: Literal["extraction"]
    extraction: ExtractionValue
    confidence: float


class BoxCoordinates(APIModel):
    top: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)
    left: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.top >= self.bottom or self.left >= self.right:
            raise ValueError("Box requires top < bottom and left < right")
        return self


# Preserve main's nested response type name alongside the unified public name.
BoxValue = BoxCoordinates


class BoxAnswer(_AnswerModel):
    type: Literal["box"]
    box: BoxCoordinates
    confidence: float | None = Field(ge=0, le=1)


class DatetimeAnswer(_AnswerModel):
    type: Literal["datetime"]
    datetime: str
    probabilities: dict[str, float]
    confidence: float


Answer = (
    NoulAnswer
    | ChoiceAnswer
    | ScoreAnswer
    | TextAnswer
    | NumberAnswer
    | SpanAnswer
    | ExtractionAnswer
    | DatetimeAnswer
    | BoxAnswer
)


QuestionT = TypeVar("QuestionT", bound="_Question")


class _Question(Question, BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: JsonValue = None
    name: str | None = Field(default=None, exclude=True)
    _answer_type: ClassVar[type[APIModel]]
    _answered: bool = PrivateAttr(default=False)
    _answer_data: APIModel | None = PrivateAttr(default=None)
    _original_question: dict[str, Any] | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _wire_answer(cls, value: Any, handler: Any) -> Any:
        primary = {
            "noul": "noul",
            "choice": "choice",
            "score": "score",
            "text": "text",
            "number": "value",
            "span": "span",
            "extraction": "extraction",
            "datetime": "datetime",
            "box": "box",
        }
        kind = cls.model_fields["type"].default
        if isinstance(value, dict) and value.get("type") == kind and primary[kind] in value:
            return cls.from_answer(value)
        return handler(value)

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            answer_type = getattr(type(self), "_answer_type", None)
            if (
                private is not None
                and not private.get("_answered", False)
                and answer_type
                and name in answer_type.model_fields
                and name not in ("type", "name", "instructions")
            ):
                raise AttributeError(f"{name} is only available on an answered question")
        return super().__getattribute__(name)

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        data = (
            self._current_answer().model_dump()
            if self._answer_data is not None
            else self.to_question()
        )
        if self.name is not None:
            data = {"name": self.name, **data}
        return iter(data.items())

    def to_question(self) -> dict[str, Any]:
        if self._answered:
            if self._original_question is None:
                raise ValueError("The original question definition is unavailable for this answer")
            return dict(self._original_question)
        result = {"type": self.type}
        for key in ("instructions", "criteria"):
            if key in self.model_fields_set:
                result[key] = getattr(self, key)
        return result

    def _current_answer(self) -> APIModel:
        # Keep response extensions while serializing the current public fields.
        data = dict(self._answer_data.model_extra or {}) if self._answer_data is not None else {}
        for name, field in self._answer_type.model_fields.items():
            if not field.exclude:
                data[name] = getattr(self, name)
        # Like normal Pydantic serialization, this does not revalidate assignments.
        return self._answer_type.model_construct(**data)

    @model_serializer
    def _serialize(self, info: SerializationInfo) -> dict[str, Any]:
        if self._answer_data is not None:
            return self._current_answer().model_dump(
                mode=info.mode,
                include=cast(Any, info.include),
                exclude=cast(Any, info.exclude),
                by_alias=info.by_alias,
                exclude_unset=info.exclude_unset,
                exclude_defaults=info.exclude_defaults,
                exclude_none=info.exclude_none,
                round_trip=info.round_trip,
            )
        return self.to_question()

    @classmethod
    def from_answer(
        cls, value: Any, *, name: str | None = None, question: dict[str, Any] | None = None
    ) -> Self:
        validated = cls._answer_type.model_validate(value)
        definition = dict(question) if question is not None else None
        data = {
            **(definition or {}),
            **validated.model_dump(),
            "name": name,
            "instructions": (definition or {}).get("instructions"),
        }
        result = cls.model_construct(**data)
        result._answered = True
        result._answer_data = validated
        result._original_question = definition
        return result

    def nullable(self, threshold: float = 0.5) -> Nullable[Self]:
        return Nullable(self, threshold)

    def __or__(self, other: object) -> Nullable[Self]:
        if other is None:
            return self.nullable()
        return NotImplemented


class Noul(_Question, NoulAnswer):
    def __bool__(self) -> bool:
        return self.noul > 0.5

    model_config = ConfigDict(extra="forbid")
    _answer_type: ClassVar[type[APIModel]] = NoulAnswer
    noul: float = 0

    type: Literal["noul"] = "noul"
    criteria: dict[Literal["true", "false"], JsonValue] | None = None


_CHOICE_LITERAL_MODELS: dict[tuple[type[Any], tuple[Any, ...]], type[BaseModel]] = {}

ChoiceT = DefaultTypeVar("ChoiceT", bound=str, default=str)


class Choice(_Question, ChoiceAnswer, Generic[ChoiceT]):
    @classmethod
    def __class_getitem__(
        cls, typevar_values: type[Any] | tuple[type[Any], ...]
    ) -> type[BaseModel]:
        args = typevar_values if isinstance(typevar_values, tuple) else (typevar_values,)
        if len(args) == 1 and get_origin(args[0]) is Literal:
            keys = get_args(args[0])
            key = (cls, keys)
            if key not in _CHOICE_LITERAL_MODELS:
                specialized = cast(type[BaseModel], super().__class_getitem__(typevar_values))
                # Pydantic's cache treats reordered Literal unions as equal.
                # Keep their declaration order for positional criteria descriptions.
                if get_args(specialized.model_fields["choice"].annotation) != keys:
                    specialized = create_model(
                        specialized.__name__,
                        __base__=specialized,
                        choice=(args[0], Field(default="")),
                    )
                _CHOICE_LITERAL_MODELS[key] = specialized
            return _CHOICE_LITERAL_MODELS[key]
        return cast(type[BaseModel], super().__class_getitem__(typevar_values))

    model_config = ConfigDict(extra="forbid")
    _answer_type: ClassVar[type[APIModel]] = ChoiceAnswer
    choice: ChoiceT = cast(ChoiceT, "")
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0

    type: Literal["choice"] = "choice"
    criteria: dict[str, JsonValue] = Field(min_length=1)

    @classmethod
    def literal_keys(cls) -> tuple[str, ...] | None:
        annotation = cls.model_fields["choice"].annotation
        if get_origin(annotation) is Literal:
            keys = get_args(annotation)
            if not keys or any(not isinstance(key, str) or not key for key in keys):
                raise ValueError("Choice literals must be nonempty strings")
            return cast(tuple[str, ...], keys)
        if annotation is str or isinstance(annotation, TypeVar):
            return None
        raise TypeError("Choice must be parameterized with str or string Literal values")

    @model_validator(mode="before")
    @classmethod
    def _literal_criteria(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("type") == "choice" and "choice" in value:
            # Wire answers must not acquire request-only criteria during validation.
            return value
        keys = cls.literal_keys()
        if keys is None or not isinstance(value, dict):
            return value
        value = dict(value)
        criteria = value.get("criteria")
        if criteria is None:
            criteria = {key: key for key in keys}
        elif isinstance(criteria, list):
            if len(criteria) != len(keys):
                raise ValueError("Choice criteria must match the number of literal keys")
            criteria = dict(zip(keys, criteria, strict=True))
        elif not isinstance(criteria, dict) or set(criteria) != set(keys):
            raise ValueError("Choice criteria keys must match the literal keys")
        value["criteria"] = criteria
        return value

    @classmethod
    def from_answer(
        cls, value: Any, *, name: str | None = None, question: dict[str, Any] | None = None
    ) -> Self:
        result = super().from_answer(value, name=name, question=question)
        TypeAdapter(cls.model_fields["choice"].annotation).validate_python(result.choice)
        return result


class Score(_Question, ScoreAnswer):
    model_config = ConfigDict(extra="forbid")
    _answer_type: ClassVar[type[APIModel]] = ScoreAnswer
    score: float = 0
    legend: dict[int, JsonValue] = Field(default_factory=dict)
    probabilities: dict[int, float] = Field(default_factory=dict)
    confidence: float = 0

    type: Literal["score"] = "score"
    criteria: list[JsonValue] = Field(min_length=1)


class Text(_Question, TextAnswer):
    model_config = ConfigDict(extra="forbid")
    _answer_type: ClassVar[type[APIModel]] = TextAnswer
    text: str = ""
    confidence: float = 0

    type: Literal["text"] = "text"
    instructions: str = Field(min_length=1)


class Number(_Question, NumberAnswer):
    model_config = ConfigDict(extra="forbid")
    _answer_type: ClassVar[type[APIModel]] = NumberAnswer
    value: float = 0
    probabilities: dict[str, float] = Field(default_factory=dict)
    variance: float = 0

    type: Literal["number"] = "number"
    instructions: str = Field(min_length=1)
    criteria: dict[Literal["min", "max", "step"], str] | None = None


class Span(_Question, SpanAnswer):
    model_config = ConfigDict(extra="forbid")
    _answer_type: ClassVar[type[APIModel]] = SpanAnswer
    span: str = ""
    start: int = 0
    end: int = 0
    confidence: float = 0

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        # The strict SpanAnswer validator runs in from_answer.
        return self

    type: Literal["span"] = "span"
    instructions: str = Field(min_length=1)


class Extraction(_Question, ExtractionAnswer):
    """Extract text using the current Mo 2.0 nested extraction contract."""

    model_config = ConfigDict(extra="forbid")
    _answer_type: ClassVar[type[APIModel]] = ExtractionAnswer
    type: Literal["extraction"] = "extraction"
    instructions: str = Field(min_length=1, pattern=r"\S")
    extraction: ExtractionValue = Field(
        default_factory=lambda: ExtractionValue(start=0, end=0, text="")
    )
    confidence: float = 0

    @classmethod
    def from_answer(
        cls, value: Any, *, name: str | None = None, question: dict[str, Any] | None = None
    ) -> Self:
        result = super().from_answer(value, name=name, question=question)
        result.extraction = cast(ExtractionAnswer, result._answer_data).extraction
        return result


class Box(_Question, BoxAnswer):
    model_config = ConfigDict(extra="forbid")
    _answer_type: ClassVar[type[APIModel]] = BoxAnswer
    type: Literal["box"] = "box"
    instructions: str = Field(min_length=1)
    box: BoxCoordinates = Field(
        default_factory=lambda: BoxCoordinates.model_construct(
            top=0,
            bottom=0,
            left=0,
            right=0,
        )
    )
    confidence: float | None = 0

    @classmethod
    def from_answer(
        cls, value: Any, *, name: str | None = None, question: dict[str, Any] | None = None
    ) -> Self:
        result = super().from_answer(value, name=name, question=question)
        result.box = cast(BoxAnswer, result._answer_data).box
        return result


def _datetime_criteria(pattern: DatetimeFormat) -> dict[Literal["format"], DatetimeFormat]:
    return {"format": pattern}


class Datetime(_Question, DatetimeAnswer):
    model_config = ConfigDict(extra="forbid")
    _answer_type: ClassVar[type[APIModel]] = DatetimeAnswer
    datetime: str = ""
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0

    type: Literal["datetime"] = "datetime"
    instructions: str = Field(min_length=1)
    criteria: dict[Literal["format"], DatetimeFormat] = Field(
        default_factory=lambda: _datetime_criteria("YYYY-MM-DDTHH:MM"),
        min_length=1,
        max_length=1,
    )
    _allowed_formats: ClassVar[tuple[str, ...]] = DATETIME_FORMATS

    def to_question(self) -> dict[str, Any]:
        result = super().to_question()
        if not self._answered:
            result["criteria"] = dict(self.criteria)
        return result

    @model_validator(mode="after")
    def _compatible_format(self) -> Self:
        if self.criteria["format"] not in self._allowed_formats:
            raise ValueError(f"Unsupported format for {type(self).__name__}")
        return self

    @model_validator(mode="before")
    @classmethod
    def _format(cls, value: Any) -> Any:
        if isinstance(value, dict) and "format" in value:
            value = dict(value)
            pattern = value.pop("format")
            if "criteria" in value:
                raise ValueError("Use format or criteria, not both")
            value["criteria"] = {"format": pattern}
        return value


class Date(Datetime):
    criteria: dict[Literal["format"], DatetimeFormat] = Field(
        default_factory=lambda: _datetime_criteria("YYYY-MM-DD"),
        min_length=1,
        max_length=1,
    )
    _allowed_formats: ClassVar[tuple[str, ...]] = ("YYYY", "YYYY-MM", "YYYY-MM-DD")


class Time(Datetime):
    criteria: dict[Literal["format"], DatetimeFormat] = Field(
        default_factory=lambda: _datetime_criteria("HH:MM"),
        min_length=1,
        max_length=1,
    )
    _allowed_formats: ClassVar[tuple[str, ...]] = ("HH", "HH:MM", "HH:MM:SS")


@dataclass(frozen=True)
class Nullable(Generic[QuestionT]):
    question: QuestionT
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.threshold, bool)
            or not math.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
        ):
            raise ValueError("Nullable threshold must be a finite number between 0 and 1")

    def nullable(self, threshold: float = 0.5) -> Nullable[QuestionT]:
        return Nullable(self.question, threshold)


QuestionLike = (
    Question
    | NoulQuestion
    | ChoiceQuestion
    | ScoreQuestion
    | Noul
    | Choice
    | Score
    | Text
    | Number
    | Nullable[Any]
    | Span
    | Extraction
    | ExtractionQuestion
    | BoxQuestion
    | Datetime
    | Box
)


_PRIMITIVES: dict[str, type[_Question]] = {
    "noul": Noul,
    "choice": Choice,
    "score": Score,
    "text": Text,
    "number": Number,
    "span": Span,
    "extraction": Extraction,
    "datetime": Datetime,
    "box": Box,
}


def primitive_type(annotation: Any) -> type[_Question] | None:
    if isinstance(annotation, type) and issubclass(annotation, _Question):
        return annotation
    for primitive in _PRIMITIVES.values():
        if annotation is primitive._answer_type:
            return primitive
    return None


def unified_answer(
    value: Any, *, name: str | None = None, question: dict[str, Any] | None = None
) -> _Question:
    if isinstance(value, _Question):
        return value
    if isinstance(value, APIModel):
        value = value.model_dump()
    if not isinstance(value, dict) or value.get("type") not in _PRIMITIVES:
        raise ValueError("Unknown or malformed answer type")
    if question is not None and value["type"] != question["type"]:
        raise ValueError("Answer type does not match its question")
    return _PRIMITIVES[value["type"]].from_answer(value, name=name, question=question)


def hydrate_response(
    response: Any,
    *,
    questions: dict[str, Any] | None = None,
    question_models: dict[str, type[_Question]] | None = None,
) -> Any:
    questions = questions or {}
    question_models = question_models or {}
    named: dict[str, tuple[str, Any]] = {}
    for name, field in type(response).model_fields.items():
        if name in DecisionResponse.model_fields:
            continue
        key = field.validation_alias
        if key is None:
            key = field.alias if field.alias is not None else name
        value = getattr(response, name)
        if isinstance(key, str) and isinstance(value, _AnswerModel):
            named[key] = (name, value)
    known_answers = {primitive._answer_type for primitive in _PRIMITIVES.values()}
    for key, original in response.answers.items():
        value = named[key][1] if key in named else original
        if value is None:
            continue
        definition = questions.get(key)
        if definition is not None and value.type != definition["type"]:
            raise ValueError("Answer type does not match its question")
        if isinstance(value, _Question):
            answer = value
        elif type(value) not in known_answers:
            # The caller requested a custom Answer subclass. Preserve its validators/type.
            answer = value
        else:
            primitive = question_models.get(key) or primitive_type(type(value))
            if primitive is None:
                raise ValueError("Unknown answer type")
            answer = primitive.from_answer(value.model_dump(), name=key, question=definition)
        answer.name = key
        answer.instructions = definition.get("instructions") if definition else None
        if isinstance(answer, _Question) and definition is not None:
            answer._original_question = dict(definition)
            if "criteria" in definition:
                cast(Any, answer).criteria = definition["criteria"]
        response.answers[key] = answer
        if key in named:
            setattr(response, named[key][0], answer)
    return response


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


class TokenUsage(APIModel):
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None


class ThinkingResult(APIModel):
    max_tokens: int | None
    reasoning_tokens: int
    generated_tokens: int
    finish_reason: Literal["natural", "budget_exhausted", "context_exhausted"]
    generation_ms: float
    scoring_ms: float
    cached_tokens: int | None


AnswerT = DefaultTypeVar("AnswerT", default=Answer)


class DecisionResponse(APIModel, Generic[AnswerT]):
    _defer_unification: ClassVar[bool] = False

    model: str
    answers: dict[str, AnswerT]
    usage: TokenUsage
    thinking: dict[str, ThinkingResult] | None = None
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

    @model_validator(mode="after")
    def _unify_after_validation(self) -> Self:
        if not self._defer_unification:
            hydrate_response(self)
        return self

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


class CurrentUser(APIModel):
    id: str
    email: str | None
    profile: Entity | None


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


class Branch(APIModel):
    name: str
    head_sha: str = Field(alias="headSha")
    created_at: datetime = Field(alias="createdAt")


class BranchList(APIModel):
    branches: list[Branch]


class Commit(APIModel):
    sha: str
    parent_shas: list[str] = Field(alias="parentShas")
    message: str
    author_name: str = Field(alias="authorName")
    author_email: str = Field(alias="authorEmail")
    created_at: datetime = Field(alias="createdAt")


class CommitList(APIModel):
    commits: list[Commit]


class CommitFile(APIModel):
    path: str
    status: Literal[
        "added",
        "modified",
        "deleted",
        "renamed",
        "copied",
        "type_changed",
        "unmerged",
        "unknown",
    ]
    previous_path: str | None = Field(default=None, alias="previousPath")


class CommitStats(APIModel):
    files: int
    additions: int
    deletions: int
    changes: int


class CommitDetails(Commit):
    committer_name: str = Field(alias="committerName")
    committer_email: str = Field(alias="committerEmail")
    verified: bool
    files: list[CommitFile]
    stats: CommitStats


class RepositoryFile(APIModel):
    path: str
    type: Literal["blob", "tree", "symlink", "submodule"]
    mode: str


class RepositoryFileList(APIModel):
    files: list[RepositoryFile]


class RepositoryFileContents(APIModel):
    path: str
    revision: str | None
    contents: str


class Tag(APIModel):
    """An immutable name for a commit, usable as a decision ``revision``."""

    name: str
    commit_sha: str = Field(alias="commitSha")


class TagList(APIModel):
    tags: list[Tag]


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


class ExampleSummary(APIModel):
    total: int
    graded: int
    commit_shas: list[str] = Field(alias="commitShas")


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


ResourceKind = Literal["dataset", "eval-only-dataset", "artifact"]
FineTuneStatus = Literal[
    "queued", "training", "committing", "publishing", "done", "failed", "cancelled"
]
TrainingType = Literal["head_only", "lora"]


class LearningRateSchedulerArgs(TypedDict, total=False):
    min_lr_ratio: float
    num_cycles: float


class LearningRateScheduler(TypedDict, total=False):
    lr_scheduler_type: Literal["constant", "linear", "cosine"]
    lr_scheduler_args: LearningRateSchedulerArgs


class TrainingControls(TypedDict, total=False):
    n_epochs: int
    n_checkpoints: int
    n_evals: int
    batch_size: int | Literal["max"]
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler: LearningRateScheduler
    warmup_ratio: float
    max_grad_norm: float
    weight_decay: float
    random_seed: int | None
    early_stopping_enabled: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_warmup_evals: int | None


class Resource(APIModel):
    id: str
    name: str
    description: str | None
    kind: ResourceKind
    path: str
    sha256: str
    size_bytes: int = Field(alias="sizeBytes")
    repos: list[str]
    commit_sha: str = Field(alias="commitSha")


class ResourcePage(APIModel):
    resources: list[Resource]
    commit_sha: str | None = Field(alias="commitSha")
    next_cursor: str | None = Field(alias="nextCursor")


class UploadPart(APIModel):
    part_number: int = Field(alias="partNumber")
    sha256: str
    size_bytes: int = Field(alias="sizeBytes")
    uploaded: bool


class UploadResult(APIModel):
    name: str
    description: str | None
    kind: ResourceKind
    path: str
    sha256: str
    size_bytes: int = Field(alias="sizeBytes")
    repos: list[str]
    commit_sha: str = Field(alias="commitSha")


class Upload(APIModel):
    id: str
    name: str
    description: str | None
    kind: ResourceKind
    sha256: str
    size_bytes: int = Field(alias="sizeBytes")
    repos: list[str]
    status: Literal["pending", "processing", "completed", "failed", "cancelled", "expired"]
    created_at: datetime = Field(alias="createdAt")
    expires_at: datetime = Field(alias="expiresAt")
    part_size_bytes: int = Field(alias="partSizeBytes")
    parts: list[UploadPart]
    result: UploadResult | None
    error: JobError | None


class PreparedUploadPart(APIModel):
    part_number: int = Field(alias="partNumber")
    upload_url: str = Field(alias="uploadUrl")
    method: Literal["PUT"]
    headers: dict[str, str]
    expires_in: int = Field(alias="expiresIn")


class ProtegeJob(APIModel):
    id: str
    status: Literal["running", "succeeded", "failed", "cancelled"]


class FineTunePublication(APIModel):
    commit_sha: str = Field(alias="commitSha")
    branch: str


class FineTuneResult(APIModel):
    metrics: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    serving_qualification: dict[str, Any] | None = Field(alias="servingQualification")
    publication: FineTunePublication | None


class FineTune(APIModel):
    id: str
    repository_id: str = Field(alias="repositoryId")
    dataset_id: str = Field(alias="datasetId")
    training_file: str
    dataset_commit_sha: str = Field(alias="datasetCommitSha")
    dataset_sha256: str = Field(alias="datasetSha256")
    status: FineTuneStatus
    branch: str
    source_commit_sha: str = Field(alias="sourceCommitSha")
    result_commit_sha: str | None = Field(alias="resultCommitSha")
    training_type: TrainingType
    preset: str
    progress: Any | None
    result: FineTuneResult | None
    error: JobError | None
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    started_at: datetime | None = Field(alias="startedAt")
    finished_at: datetime | None = Field(alias="finishedAt")


class FineTuneList(APIModel):
    fine_tunes: list[FineTune] = Field(alias="fineTunes")


class FineTuneLogs(APIModel):
    logs: list[str]
    available: bool
    running: bool
