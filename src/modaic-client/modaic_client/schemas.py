from datetime import datetime
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, model_validator


class Output(BaseModel):
    model_config = {"extra": "allow"}


class ArbiterPredictionItem(BaseModel):
    arbiter_repo: str
    commit_hash: str
    output: Any
    output_field: str
    reasoning: str
    messages: list[dict]
    prediction_id: str


class ArbiterPredictResponse(BaseModel):
    example_id: str
    predictions: list[ArbiterPredictionItem]


class PredictedExample(BaseModel):
    id: Optional[str] = None
    alt_id: Optional[str] = None
    arbiter_repo: str
    arbiter_hash: str = ""
    input: Any = None
    ground_truth: Optional[str] = None
    ground_reasoning: str = ""
    messages: Optional[list[dict]] = None
    output: Optional[str] = None
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
    arbiter_repo: str
    ground_truth: Optional[str]
    ground_reasoning: Optional[str]


class AnnotateExampleResponse(BaseModel):
    status: str


class FieldSchema(BaseModel):
    name: str
    type: Literal["string", "number", "boolean", "array", "object"]
    allowed_values: list[Any] | None = None
    schema: dict | None = None
    nullable: bool = False
    description: str | None = None

    @model_validator(mode="after")
    def validate_schema(self):
        if self.type == "object" and self.allowed_values is not None:
            raise ValueError("Allowed values must be None if type is 'object'")
        if self.schema is not None and self.type != "object":
            raise ValueError("Schema must be None if type is not 'object'")
        return self


class InitArbiterRequest(BaseModel):
    repo: str
    inputs: list[FieldSchema]
    output: FieldSchema
    instructions: str | None = None


class ConfidenceScoreResponse(BaseModel):
    confidence: float
    embedding: list[float]
