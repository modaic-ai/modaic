from typing import Literal

from pydantic import BaseModel, field_validator


TYPE_MAP = {
    "str": str,
    "string": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}


class PredictField(BaseModel):
    name: str
    type: str = "string"
    description: str | None = None
    options: list[str] | None = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in TYPE_MAP:
            raise ValueError(f"Unknown type '{v}'. Must be one of: {', '.join(TYPE_MAP)}")
        return v

    def resolve_type(self) -> type:
        if self.options is not None:
            return Literal[tuple(self.options)]
        return TYPE_MAP[self.type]


class LMSpec(BaseModel):
    model: str
    model_config = {"extra": "allow"}


class PredictYamlSpec(BaseModel):
    model: str | None = None
    lm: LMSpec | None = None
    instructions: str | None = None
    inputs: list[PredictField] = []
    outputs: list[PredictField] = []
