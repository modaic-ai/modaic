# ruff: noqa: T201
import copy
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

import dspy
from dspy import Signature
from pydantic import TypeAdapter

from modaic.serializers import DSPyTypeSchemaGenerator

from .modalities import canonicalize_modalities, merge_required_modalities, reconcile_explicit_modalities

if TYPE_CHECKING:
    from .predict import Predict


# Known-model metadata overrides. This is not an allowlist: models absent from
# the mapping are still valid arbiters and use the server's shared-probe fallback.
ARBITER_PROBES = {
    # "qwen3-32b": {"probe_model": "modaic/qwen3-32b-probe", "size": "medium"},
    # "qwen3-vl-32b-instruct": {"probe_model": "modaic/qwen3-32b-probe", "size": "medium"},
    # "qwen3.5-4b": {"probe_model": "modaic/qwen3.5-4b-probe", "size": "small", "supports_reasoning": True},
    "llama-3.1-8b": {"model": "llama-3.1-8b", "size": "small"},
    "llama-3.1-8b-instruct": {
        "model": "llama-3.1-8b",
        "size": "small",
    },
    "gpt-oss-120b": {
        "model": "gpt-oss-120b",
        "size": "medium",
        "supports_reasoning": True,
    },
    "gpt-5.5": {
        "model": "gpt-5.5",
        "size": "medium",
        "supports_reasoning": True,
    },
    "claude-opus-4-8": {
        "model": "claude-opus-4-8",
        "size": "medium",
        "supports_reasoning": True,
    },
    "glm-5.2": {"model": "glm-5.2", "size": "small", "supports_reasoning": True},
    "glm-5.3": {"model": "glm-5.3", "size": "small", "supports_reasoning": True},
    "glm-5.3-flash": {
        "model": "glm-5.3-flash",
        "size": "small",
        "supports_reasoning": True,
    },
    "qwen3.8-2.4t-a95b": {
        "model": "qwen3.8-2.4t-a95b",
        "size": "small",
        "supports_reasoning": True,
    },
    "qwen3.8-27b": {
        "model": "qwen3.8-27b",
        "size": "small",
        "supports_reasoning": True,
    },
    "qwen3.8-flash": {
        "model": "qwen3.8-flash",
        "size": "small",
        "supports_reasoning": True,
    },
    "qwen3.8-max-0902": {
        "model": "qwen3.8-max-0902",
        "size": "small",
        "supports_reasoning": True,
    },
    "kimi-k3": {"model": "kimi-k3", "size": "small", "supports_reasoning": True},
    "deepseek-v4-pro": {
        "model": "deepseek-v4-pro",
        "size": "small",
        "supports_reasoning": True,
    },
    "deepseek-v4-flash": {
        "model": "deepseek-v4-flash",
        "size": "small",
        "supports_reasoning": True,
    },
}
_ARBITER_REASONING_DESCRIPTION = (
    "Your reasoning for your answer. Include any uncertainties about your answer or ambiguity in the task."
)


def _annotations_are_structurally_equivalent(runtime_annotation: Any, published_annotation: Any) -> bool:
    if runtime_annotation == published_annotation:
        return True
    try:
        runtime_schema = TypeAdapter(runtime_annotation).json_schema(schema_generator=DSPyTypeSchemaGenerator)
        published_schema = TypeAdapter(published_annotation).json_schema(schema_generator=DSPyTypeSchemaGenerator)
    except Exception:  # Pydantic can reject arbitrary user-defined annotations.
        return False
    return runtime_schema == published_schema


def normalize_model_name(model: str) -> str:
    return model.lower().split("/")[-1].replace(":", "-")


def arbiter_metadata_for_model(model: str) -> dict[str, object]:
    """Build probe metadata for any provider model.

    Known models can override serving size and reasoning capabilities. Unknown
    models use the small shared-probe tier and are resolved by the server's
    probe fallback instead of being rejected client-side.
    """
    normalized = normalize_model_name(model)
    return {"model": normalized, "size": "small", **ARBITER_PROBES.get(normalized, {})}


def is_reasoning_model(model: str) -> bool:
    normalized = normalize_model_name(model)
    probe = ARBITER_PROBES.get(normalized)
    return bool(probe and probe.get("supports_reasoning", False))


@lru_cache(maxsize=None)
def register_reasoning_model(model: str) -> None:
    if is_reasoning_model(model):
        import litellm

        existing = litellm.model_cost.get(model, {})
        existing["supports_reasoning"] = True
        litellm.register_model({model: existing})


def _insert_reasoning_field(
    signature: type[Signature],
    index: int,
    source_field: Any = None,
) -> type[Signature]:
    output_fields = [(name, field) for name, field in signature.output_fields.items() if name != "reasoning"]
    if index < 0:
        index += len(output_fields) + 1
    if index < 0 or index > len(output_fields):
        raise ValueError(
            "Cannot make Predict an Arbiter because its runtime and published signatures have a reasoning "
            f"structural mismatch: output index {index} cannot be represented in {signature.__name__}."
        )

    if source_field is None:
        field = dspy.OutputField(desc=_ARBITER_REASONING_DESCRIPTION)
    else:
        field = copy.deepcopy(source_field)
    output_fields.insert(index, ("reasoning", field))

    fields = {}
    for name, existing_field in [*signature.input_fields.items(), *output_fields]:
        field_annotation = dspy.Reasoning if name == "reasoning" else existing_field.annotation
        fields[name] = (field_annotation, copy.deepcopy(existing_field))
    return dspy.make_signature(
        fields,
        instructions=signature.instructions,
        signature_name=signature.__name__,
    )


def _align_reasoning_fields(
    runtime_signature: type[Signature],
    published_signature: type[Signature],
) -> tuple[type[Signature], type[Signature]]:
    """Align reasoning positions without replacing either signature's instructions."""
    for signature in (runtime_signature, published_signature):
        if "reasoning" in signature.input_fields:
            raise ValueError(
                "Cannot make Predict an Arbiter because its runtime and published signatures have a reasoning "
                "structural mismatch: 'reasoning' must be an output field."
            )

    runtime_inputs = list(runtime_signature.input_fields)
    published_inputs = list(published_signature.input_fields)
    if runtime_inputs != published_inputs:
        raise ValueError(
            "Cannot make Predict an Arbiter because its runtime and published signatures have a reasoning "
            "structural mismatch: input fields must have the same names and order "
            f"(runtime={runtime_inputs}, published={published_inputs})."
        )
    for name in runtime_inputs:
        runtime_annotation = runtime_signature.input_fields[name].annotation
        published_annotation = published_signature.input_fields[name].annotation
        if not _annotations_are_structurally_equivalent(runtime_annotation, published_annotation):
            raise ValueError(
                "Cannot make Predict an Arbiter because its runtime and published signatures have a reasoning "
                f"structural mismatch: input field {name!r} has different types "
                f"(runtime={runtime_annotation!r}, published={published_annotation!r})."
            )

    runtime_outputs = [name for name in runtime_signature.output_fields if name != "reasoning"]
    published_outputs = [name for name in published_signature.output_fields if name != "reasoning"]
    if runtime_outputs != published_outputs:
        raise ValueError(
            "Cannot make Predict an Arbiter because its runtime and published signatures have a reasoning "
            "structural mismatch: non-reasoning output fields must have the same names and order "
            f"(runtime={runtime_outputs}, published={published_outputs})."
        )
    for name in runtime_outputs:
        runtime_annotation = runtime_signature.output_fields[name].annotation
        published_annotation = published_signature.output_fields[name].annotation
        if not _annotations_are_structurally_equivalent(runtime_annotation, published_annotation):
            raise ValueError(
                "Cannot make Predict an Arbiter because its runtime and published signatures have a reasoning "
                f"structural mismatch: output field {name!r} has different types "
                f"(runtime={runtime_annotation!r}, published={published_annotation!r})."
            )

    runtime_reasoning = runtime_signature.output_fields.get("reasoning")
    published_reasoning = published_signature.output_fields.get("reasoning")

    for reasoning_field in (runtime_reasoning, published_reasoning):
        if reasoning_field and reasoning_field.annotation not in (dspy.Reasoning, str):
            raise ValueError("'reasoning' field must be a 'dspy.Reasoning' to make modaic.Predict an Arbiter")

    # Older serialized signatures may represent reasoning as ``str``. Rebuild
    # those fields before aligning positions so both artifacts round-trip with
    # DSPy's reasoning semantics instead of silently preserving the legacy type.
    if runtime_reasoning and runtime_reasoning.annotation is not dspy.Reasoning:
        runtime_index = list(runtime_signature.output_fields).index("reasoning")
        runtime_signature = _insert_reasoning_field(runtime_signature, runtime_index, runtime_reasoning)
        runtime_reasoning = runtime_signature.output_fields["reasoning"]
    if published_reasoning and published_reasoning.annotation is not dspy.Reasoning:
        published_index = list(published_signature.output_fields).index("reasoning")
        published_signature = _insert_reasoning_field(published_signature, published_index, published_reasoning)
        published_reasoning = published_signature.output_fields["reasoning"]

    if runtime_reasoning:
        reasoning_index = list(runtime_signature.output_fields).index("reasoning")
        if published_reasoning:
            published_index = list(published_signature.output_fields).index("reasoning")
            if published_index != reasoning_index:
                published_signature = _insert_reasoning_field(
                    published_signature,
                    reasoning_index,
                    published_reasoning,
                )
        else:
            published_signature = _insert_reasoning_field(
                published_signature,
                reasoning_index,
                runtime_reasoning,
            )
    elif published_reasoning:
        reasoning_index = list(published_signature.output_fields).index("reasoning")
        runtime_signature = _insert_reasoning_field(
            runtime_signature,
            reasoning_index,
            published_reasoning,
        )
    else:
        runtime_signature = _insert_reasoning_field(runtime_signature, -2)
        reasoning_index = list(runtime_signature.output_fields).index("reasoning")
        published_signature = _insert_reasoning_field(published_signature, reasoning_index)

    return runtime_signature, published_signature


def make_arbiter(
    predict: "Predict",
    *,
    _warning_stacklevel: int = 4,
) -> "Predict":
    predict = copy.deepcopy(predict)
    if predict.lm is None:
        raise ValueError(
            "You must set an LM to make a modaic.Predict an arbiter. See available LMs https://docs.modaic.dev/guides/basic_usage/create_an_arbiter"
        )
    register_reasoning_model(predict.lm.model)
    existing_metadata = dict(predict.metadata or {})
    published_signature = predict.config.signature
    previous_provenance = predict.config.modality_provenance or {}
    try:
        inferred_modalities = merge_required_modalities(
            published_signature,
            _warning_stacklevel=_warning_stacklevel,
        )
        explicit_modalities = reconcile_explicit_modalities(
            existing_metadata.get("modalities"),
            previous_inferred=previous_provenance.get("inferred"),
            previous_explicit=previous_provenance.get("explicit"),
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            "Invalid modalities in existing arbiter metadata (including README/Hub metadata) "
            f"or Predict config provenance: {exc}"
        ) from exc
    modalities = canonicalize_modalities(inferred_modalities, explicit_modalities)
    new_metadata = {
        **existing_metadata,
        "is_arbiter": True,
        **arbiter_metadata_for_model(predict.lm.model),
        "modalities": modalities,
    }
    runtime_signature = predict.signature
    runtime_signature, published_signature = _align_reasoning_fields(runtime_signature, published_signature)

    predict.metadata = new_metadata
    predict.config.modality_provenance = {
        "inferred": inferred_modalities,
        "explicit": explicit_modalities,
    }
    predict.signature = runtime_signature
    predict.config.signature = published_signature

    return predict


if __name__ == "__main__":

    class _LMStub:
        def __init__(self, model: str):
            self.model = model

    class _PredictStub:
        def __init__(self, signature: Signature, lm: Optional["_LMStub"] = None):
            self.signature = signature
            self.config = SimpleNamespace(signature=signature, modality_provenance=None)
            self.lm = lm
            self.metadata = {}

    class NoReasoningSignature(dspy.Signature):
        """Arbiter output without a reasoning field."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class ReasoningIntSignature(dspy.Signature):
        """Arbiter output with a non-string reasoning field."""

        question: str = dspy.InputField()
        reasoning: int = dspy.OutputField()
        answer: str = dspy.OutputField()

    class ReasoningUnannotatedSignature(dspy.Signature):
        """Arbiter output with an unannotated reasoning field."""

        question: str = dspy.InputField()
        reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    class ReasoningStrSignature(dspy.Signature):
        """Arbiter output with a string reasoning field."""

        question: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    supported_model = f"provider/{next(iter(ARBITER_PROBES.keys()))}"

    print("no reasoning field")
    no_reasoning_predict = _PredictStub(NoReasoningSignature, lm=_LMStub(supported_model))
    print("reasoning field:", no_reasoning_predict.signature.output_fields.get("reasoning"))
    try:
        make_arbiter(no_reasoning_predict)
        print("make_arbiter passed")
    except Exception as exc:
        print(f"make_arbiter raised {type(exc).__name__}: {exc}")
    print()

    print("reasoning field annotated as int")
    reasoning_int_predict = _PredictStub(ReasoningIntSignature, lm=_LMStub(supported_model))
    print("reasoning annotation:", reasoning_int_predict.signature.output_fields["reasoning"].annotation)
    try:
        make_arbiter(reasoning_int_predict)
        print("make_arbiter passed")
    except Exception as exc:
        print(f"make_arbiter raised {type(exc).__name__}: {exc}")
    print()

    print("reasoning field unannotated")
    reasoning_unannotated_predict = _PredictStub(ReasoningUnannotatedSignature, lm=_LMStub(supported_model))
    print("reasoning annotation:", reasoning_unannotated_predict.signature.output_fields["reasoning"].annotation)
    try:
        make_arbiter(reasoning_unannotated_predict)
        print("make_arbiter passed")
    except Exception as exc:
        print(f"make_arbiter raised {type(exc).__name__}: {exc}")
    print()

    print("reasoning field annotated as str")
    reasoning_str_predict = _PredictStub(ReasoningStrSignature, lm=_LMStub(supported_model))
    print("reasoning annotation:", reasoning_str_predict.signature.output_fields["reasoning"].annotation)
    try:
        make_arbiter(reasoning_str_predict)
        print("make_arbiter passed")
    except Exception as exc:
        print(f"make_arbiter raised {type(exc).__name__}: {exc}")
