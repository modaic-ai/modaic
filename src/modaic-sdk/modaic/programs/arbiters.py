import copy
from functools import lru_cache
from typing import TYPE_CHECKING

import dspy
from dspy import Signature

if TYPE_CHECKING:
    from .predict import Predict


ARBITER_PROBES = {
    "qwen3-32b": {"probe_model": "modaic/qwen3-32b-probe", "size": "medium"},
    "qwen3-vl-32b-instruct": {"probe_model": "modaic/qwen3-32b-probe", "size": "medium"},
    "qwen3.5-4b": {"probe_model": "modaic/qwen3.5-4b-probe", "size": "small", "supports_reasoning": True},
    "gpt-oss-120b": {
        "probe_model": "modaic/gpt-oss-120b-probe",
        "size": "large",
        "supports_reasoning": True,
        "a": 3.537087,
        "b": -1.616884,
    },
}


def normalize_model_name(model: str) -> str:
    return model.lower().split("/")[-1].replace(":", "-")


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


def make_arbiter(predict: "Predict") -> "Predict":
    predict = copy.deepcopy(predict)
    if predict.lm is None:
        raise ValueError(
            "You must set an LM to make a modaic.Predict an arbiter. See available LMs https://docs.modaic.dev/guides/basic_usage/create_an_arbiter"
        )
    normalized_model_name = normalize_model_name(predict.lm.model)
    if predict.lm is not None and normalized_model_name not in ARBITER_PROBES:
        raise ValueError(
            f"Arbiters are not supported for model {predict.lm.model}, see https://docs.modaic.dev/guides/basic_usage/create_an_arbiter"
        )
    signature = predict.signature
    if (reas_field := signature.output_fields.get("reasoning")) and (
        reas_field.annotation is not dspy.Reasoning and reas_field.annotation is not str
    ):
        raise ValueError("'reasoning' field must be a 'dspy.Reasoning' to make modaic.Predict an Arbiter")
    elif reas_field:
        return predict

    new_signature = signature.insert(
        -2,
        "reasoning",
        dspy.OutputField(
            desc="Your reasoning for your answer. Inlude any uncertainties about your answer or ambiguity in the task."
        ),
        dspy.Reasoning,
    )
    predict.signature = new_signature
    predict.config.signature = new_signature
    predict.metadata |= {"is_arbiter": True, **ARBITER_PROBES[normalized_model_name]}

    return predict


if __name__ == "__main__":

    class _LMStub:
        def __init__(self, model: str):
            self.model = model

    class _PredictStub:
        def __init__(self, signature: Signature, lm=None):
            self.signature = signature
            self.lm = lm

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
