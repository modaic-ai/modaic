from datetime import datetime
from typing import Any, Literal, Optional, TypedDict, Union

from pydantic import AliasChoices, BaseModel, Field, model_validator


class Output(BaseModel):
    model_config = {"extra": "allow"}


class PredictedExample(BaseModel):
    id: Optional[str] = None
    alt_id: Optional[str] = None
    arbiter_repo: str
    arbiter_hash: str = ""
    input: Any = None
    ground_truth: Optional[str] = None
    ground_reasoning: str = ""
    messages: Optional[list[dict]] = None
    output: Optional[dict] = None
    serialized_output: Optional[str] = None
    reasoning: Optional[str] = None
    split: Literal["train", "test", "none"] = None
    version: Optional[int] = None
    prediction_timestamp: Optional[datetime] = None
    confidence: Optional[float] = None


class IngestExamplesResponse(BaseModel):
    queued: bool
    example_ids: list[str]


class ExamplesPage(BaseModel):
    items: list[PredictedExample]
    total: int
    page: int
    page_size: int
    total_pages: int


class PredictionAnnotation(TypedDict, total=False):
    # ``ground_truth`` is a dict mapping output field name -> value (v2 format,
    # routed to /api/v2). A plain ``str`` is the deprecated v1 format and routes
    # to the legacy /api/v1 annotate endpoint.
    arbiter_repo: str
    ground_truth: Optional[Union[str, dict]]
    ground_reasoning: Optional[str]


class AnnotateExampleResponse(BaseModel):
    status: str


class FieldSchema(BaseModel):
    name: str
    type: Literal["string", "number", "boolean", "array", "object"]
    # `options` was formerly named `allowed_values`; the old name is still
    # accepted on the wire (and in Python) as a validation alias.
    options: list[Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("options", "allowed_values"),
    )
    # An inclusive integer scale, e.g. [1, 5]. Only valid when type == "number";
    # registers the field as a modaic.Scale on the server.
    range: list[Any] | None = None
    object_schema: dict | None = None
    nullable: bool = False
    description: str | None = None

    @model_validator(mode="after")
    def validate_schema(self) -> "FieldSchema":
        if self.type == "object" and self.options is not None:
            raise ValueError("options must be None if type is 'object'")
        if self.object_schema is not None and self.type != "object":
            raise ValueError("object_schema must be None if type is not 'object'")
        if self.options is not None and self.range is not None:
            raise ValueError("'options' and 'range' cannot both be set")
        if self.range is not None:
            if self.type != "number":
                raise ValueError("'range' can only be used when type is 'number'")
            if len(self.range) != 2:
                raise ValueError("'range' must be a two-element array, e.g. [1, 5]")
            if any(isinstance(v, bool) or not isinstance(v, int) for v in self.range):
                raise ValueError("'range' values must be integers, e.g. [1, 5]")
            lo, hi = self.range
            if lo > hi:
                raise ValueError(f"'range' lower bound ({lo}) must be <= upper bound ({hi})")
        return self


class InitArbiterRequest(BaseModel):
    repo: str
    inputs: list[FieldSchema]
    outputs: list[FieldSchema]
    instructions: str | None = None
    model: str = "qwen3-vl-32b-instruct"
    base_url: str | None = None


class ConfidenceStatusResponse(BaseModel):
    status: str
    prediction_id: str
    score: float | None = None
    error: str | None = None
