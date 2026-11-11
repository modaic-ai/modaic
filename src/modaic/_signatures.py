"""Typed question signatures and request-local nullable composition."""

from __future__ import annotations

import math
import types
from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime, time
from decimal import Decimal
from typing import Annotated, Any, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel, create_model, model_validator

from ._datetime import parse_datetime
from .types import (
    DATETIME_FORMATS,
    Answer,
    DatetimeAnswer,
    DecisionResponse,
    Nullable,
    SpanAnswer,
    _Question,
    hydrate_response,
    primitive_type,
)


class Signature(DecisionResponse[Answer | None]):
    @model_validator(mode="before")
    @classmethod
    def _named_answers(cls, value: Any) -> Any:
        # prepare() resolves declared fields, including legacy native annotations,
        # alongside the standard response envelope before validating this model.
        return value


def signature_questions(signature: type[Signature]) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    for name, field in signature.model_fields.items():
        if name in DecisionResponse.model_fields:
            if isinstance(field.json_schema_extra, dict) and "modaic" in field.json_schema_extra:
                raise ValueError(f"Signature question {name!r} conflicts with response metadata")
            continue
        annotation = field.annotation
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        optional = get_origin(annotation) in (
            Union,
            types.UnionType,
        ) and type(None) in get_args(annotation)
        if optional:
            args = tuple(t for t in get_args(annotation) if t is not type(None))
            if len(args) != 1:
                raise TypeError(f"Unsupported union for {name}")
            annotation = args[0]
        extra = field.json_schema_extra
        config: dict[str, Any] = (
            dict(cast(Any, extra).get("modaic", {})) if isinstance(extra, dict) else {}
        )
        instructions = config.get("instructions")
        if instructions is None:
            instructions = name
        criteria = config.get("criteria")
        pattern = config.get("format")
        if pattern is None and annotation is not None:
            pattern = {date: "YYYY-MM-DD", time: "HH:MM", datetime: "YYYY-MM-DDTHH:MM"}.get(
                annotation
            )
        primitive = primitive_type(annotation)
        if (
            pattern is not None
            and annotation not in (str, date, datetime, time)
            and primitive is None
        ):
            raise TypeError(f"format is only supported for date/time or string fields: {name}")
        if primitive is not None:
            definition = {"instructions": instructions}
            if criteria is not None:
                definition["criteria"] = criteria
            if pattern is not None:
                definition["format"] = pattern
            if primitive.model_fields["type"].default == "number":
                primitive_bounds = {
                    k: format(Decimal(str(config[k])), "f")
                    for k in ("min", "max")
                    if config.get(k) is not None
                }
                if primitive_bounds:
                    definition["criteria"] = primitive_bounds
            question = primitive.model_validate(definition).to_question()
        elif get_origin(annotation) is Literal:
            keys = get_args(annotation)
            if not keys or any(not isinstance(key, str) or not key for key in keys):
                raise TypeError("Choice literals must be nonempty strings")
            if criteria is None:
                criteria = {key: key for key in keys}
            elif isinstance(criteria, list):
                if len(criteria) != len(keys):
                    raise ValueError("Choice criteria must match the number of literal keys")
                criteria = dict(zip(keys, criteria, strict=True))
            elif not isinstance(criteria, dict) or set(criteria) != set(keys):
                raise ValueError("Choice criteria keys must match the literal keys")
            question = {"type": "choice", "instructions": instructions, "criteria": criteria}
        elif annotation is bool:
            question = {"type": "noul", "instructions": instructions}
            if criteria is not None:
                question["criteria"] = criteria
        elif annotation in (int, float):
            if config.get("type") == "score" or (annotation is int and isinstance(criteria, list)):
                if annotation is not int or not isinstance(criteria, list) or not criteria:
                    raise ValueError("Score fields require int and nonempty criteria")
                question = {"type": "score", "instructions": instructions, "criteria": criteria}
            else:
                bounds = {k: config[k] for k in ("min", "max") if config.get(k) is not None}
                if annotation is int and any(
                    isinstance(v, bool) or not math.isfinite(v) or int(v) != v
                    for v in bounds.values()
                ):
                    raise ValueError("Integer number fields require integral bounds")
                if any(isinstance(v, bool) or not math.isfinite(v) for v in bounds.values()):
                    raise ValueError("Number bounds must be finite")
                if "min" in bounds and "max" in bounds and bounds["min"] >= bounds["max"]:
                    raise ValueError("Number requires min smaller than max")
                question = {"type": "number", "instructions": instructions}
                if bounds:
                    question["criteria"] = {
                        k: format(Decimal(str(v)), "f") for k, v in bounds.items()
                    }
        elif annotation in (date, datetime, time) or (
            annotation is str and (pattern is not None or config.get("type") == "datetime")
        ):
            if pattern not in DATETIME_FORMATS:
                raise ValueError(f"{name} requires format set to a supported ISO datetime format")
            if criteria is not None:
                raise ValueError("Datetime fields use format, not criteria")
            if annotation is date and pattern != "YYYY-MM-DD":
                raise ValueError("date fields require YYYY-MM-DD; use str for partial dates")
            if annotation is datetime and pattern not in (
                "YYYY-MM-DDTHH",
                "YYYY-MM-DDTHH:MM",
                "YYYY-MM-DDTHH:MM:SS",
            ):
                raise ValueError("datetime fields require a date and time format")
            if annotation is time and pattern not in ("HH", "HH:MM", "HH:MM:SS"):
                raise ValueError("time fields require a time-only format")
            question = {
                "type": "datetime",
                "instructions": instructions,
                "criteria": {"format": pattern},
            }
        elif annotation is SpanAnswer or (annotation is str and config.get("type") == "span"):
            if criteria is not None:
                raise ValueError("Span questions do not accept criteria")
            question = {"type": "span", "instructions": instructions}
        elif annotation is str:
            question = {"type": "text", "instructions": instructions}
        else:
            raise TypeError(f"Unsupported signature field {name}: {annotation!r}")
        threshold = config.get("nullable")
        if threshold is not None or optional:
            question = {"question": question, "threshold": 0.5 if threshold is None else threshold}
        questions[name] = question
    if not questions:
        raise ValueError("A signature requires at least one question")
    return questions


def prepare(questions: Any, response_model: type[BaseModel]) -> tuple[Any, type[BaseModel]]:
    signature = (
        questions if isinstance(questions, type) and issubclass(questions, Signature) else None
    )
    if signature:
        if response_model is not DecisionResponse:
            raise TypeError("A signature cannot be combined with response_model")
        questions = signature_questions(signature)
    if not isinstance(questions, Mapping):
        return questions, response_model
    wire: dict[str, Any] = {}
    gates: dict[str, tuple[str, float]] = {}
    originals: dict[str, Any] = {}
    question_models: dict[str, type[_Question]] = {}
    question: Any
    for name, value in questions.items():
        if isinstance(value, Nullable):
            question, threshold = value.question, value.threshold
        elif signature and isinstance(value, dict) and "question" in value:
            question, threshold = value["question"], value["threshold"]
        else:
            question, threshold = value, None
        if isinstance(question, _Question):
            question_models[name] = type(question)
            question = question.to_question()
        question = deepcopy(question)
        originals[name] = question
        wire[name] = question
        if threshold is not None:
            gate = f"__modaic_answerable_{name}"
            if gate in questions or gate in wire:
                raise ValueError(f"Nullable gate ID conflicts with question {gate!r}")
            gates[name] = (gate, threshold)
            wire[gate] = {
                "type": "noul",
                "instructions": {
                    "task": (
                        "Does the supplied state contain enough information "
                        "to answer this question?"
                    ),
                    "question": question,
                },
            }
    if not gates and not signature and not issubclass(response_model, DecisionResponse):
        return questions, response_model
    base = signature or response_model

    @model_validator(mode="before")
    def resolve(cls: Any, data: Any) -> Any:
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise ValueError("Expected a decision response with answers")
        answers = dict(data["answers"])
        for name, (gate, threshold) in gates.items():
            result = answers.pop(gate, None)
            if not isinstance(result, dict) or result.get("type") != "noul":
                raise ValueError(f"Missing or malformed answerability gate for {name}")
            probability = result.get("noul")
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError(f"Invalid answerability probability for {name}")
            if name not in answers:
                raise ValueError(f"Missing answer for {name}")
            if probability < threshold:
                answers[name] = None
        if not signature:
            return {**data, "answers": answers}
        values: dict[str, Any] = {}
        for name, question in originals.items():
            if name not in answers:
                raise ValueError(f"Missing answer for {name}")
            answer = answers[name]
            if answer is None:
                values[name] = None
                continue
            kind = question["type"]
            if not isinstance(answer, dict) or answer.get("type") != kind:
                raise ValueError(f"Unexpected answer type for {name}")
            annotation = signature.model_fields[name].annotation
            if get_origin(annotation) in (Union, types.UnionType):
                annotation = next(t for t in get_args(annotation) if t is not type(None))
            primitive = primitive_type(annotation)
            if primitive is not None:
                values[name] = answer
                continue
            key = {
                "choice": "choice",
                "noul": "noul",
                "number": "number",
                "text": "text",
                "score": "score",
                "datetime": "datetime",
                "span": "span",
            }[kind]
            if key not in answer:
                raise ValueError(f"Missing answer value for {name}")
            value = answer[key]
            if kind == "datetime":
                DatetimeAnswer.model_validate(answer)
                value = parse_datetime(value, question["criteria"]["format"])
                annotation = signature.model_fields[name].annotation
                if get_origin(annotation) in (Union, types.UnionType):
                    annotation = next(t for t in get_args(annotation) if t is not type(None))
                if annotation is date:
                    value = date.fromisoformat(value)
                elif annotation is datetime:
                    value = datetime.fromisoformat(value)
                elif annotation is time:
                    value = time.fromisoformat(value)
            elif kind == "span":
                span_answer = SpanAnswer.model_validate(answer)
                annotation = signature.model_fields[name].annotation
                if get_origin(annotation) in (Union, types.UnionType):
                    annotation = next(t for t in get_args(annotation) if t is not type(None))
                value = span_answer if annotation is SpanAnswer else span_answer.span.text
            elif kind == "noul":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError("Invalid Noul probability")
                value = value >= 0.5
            elif kind in ("score", "number") and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid numeric answer for {name}")
            if kind == "score":
                # Signatures expose integer scores; round to nearest, ties to even.
                value = round(value)
            values[name] = value
        return {**data, "answers": answers, **values}

    @model_validator(mode="after")
    def hydrate(self: Any) -> Any:
        if isinstance(self, DecisionResponse):
            return hydrate_response(self, questions=originals, question_models=question_models)
        return self

    fields: dict[str, Any] = {}
    if signature:
        for name, field in signature.model_fields.items():
            if name in gates:
                nullable_field = deepcopy(field)
                annotation = cast(Any, field.annotation) | None
                fields[name] = (annotation, nullable_field)
    elif gates and issubclass(base, DecisionResponse):
        fields["answers"] = (dict[str, Answer | None], ...)
    parsed = create_model(
        f"{base.__name__}Resolved",
        __base__=base,
        __validators__={
            "_resolve_nullable": cast(Any, resolve),
            "_hydrate_validated": cast(Any, hydrate),
        },
        **fields,
    )
    if issubclass(parsed, DecisionResponse):
        parsed._defer_unification = True
    return wire, parsed
