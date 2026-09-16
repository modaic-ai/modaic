from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import dspy
import pytest
from modaic import PrecompiledProgram, Predict, PredictConfig
from modaic.module_utils import add_metadata_to_readme, load_metadata_from_readme
from modaic.programs import modalities as modalities_module
from modaic.programs.arbiters import make_arbiter
from modaic.programs.modalities import (
    canonicalize_modalities,
    infer_required_modalities,
    merge_required_modalities,
    reconcile_explicit_modalities,
)
from pydantic import BaseModel
from pydantic.errors import PydanticInvalidForJsonSchema


class TextSignature(dspy.Signature):
    prompt: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ImageSignature(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    answer: str = dspy.OutputField()


class ImageReasoningSignature(dspy.Signature):
    """Base image-judging instructions."""

    image: dspy.Image = dspy.InputField()
    reasoning: dspy.Reasoning = dspy.OutputField()
    answer: str = dspy.OutputField()


class ImageInstructionSignature(dspy.Signature):
    """Base image instructions."""

    image: dspy.Image = dspy.InputField()
    answer: str = dspy.OutputField()


class MultiOutputImageSignature(dspy.Signature):
    """Base multi-output image instructions."""

    image: dspy.Image = dspy.InputField()
    summary: str = dspy.OutputField(desc="Short summary")
    answer: str = dspy.OutputField(desc="Final answer")


class ReorderedMultiOutputImageSignature(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    answer: str = dspy.OutputField(desc="Final answer")
    summary: str = dspy.OutputField(desc="Short summary")


class AlternateInputMultiOutputImageSignature(dspy.Signature):
    prompt: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="Short summary")
    answer: str = dspy.OutputField(desc="Final answer")


class AlternateInputTypeMultiOutputImageSignature(dspy.Signature):
    image: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="Short summary")
    answer: str = dspy.OutputField(desc="Final answer")


class AlternateOutputTypeMultiOutputImageSignature(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    summary: int = dspy.OutputField(desc="Short summary")
    answer: str = dspy.OutputField(desc="Final answer")


class Screenshot(dspy.Image):
    caption: str


class StructuredInput(BaseModel):
    """A documented nested request model."""

    prompt: str
    priority: int = 0


class StructuredSignature(dspy.Signature):
    request: StructuredInput = dspy.InputField()
    answer: str = dspy.OutputField()


class ImageSubclassSignature(dspy.Signature):
    screenshot: Screenshot = dspy.InputField()
    answer: str = dspy.OutputField()


class ReusedImageSubclassSignature(dspy.Signature):
    before: Screenshot = dspy.InputField()
    after: Screenshot = dspy.InputField()
    answer: str = dspy.OutputField()


class ContainerImageSubclassSignature(dspy.Signature):
    maybe: Optional[Screenshot] = dspy.InputField(default=None)
    gallery: list[Screenshot] = dspy.InputField()
    answer: str = dspy.OutputField()


class AudioClip(dspy.Audio):
    language: str


class AudioSubclassSignature(dspy.Signature):
    clip: AudioClip = dspy.InputField()
    answer: str = dspy.OutputField()


class OptionalImageSignature(dspy.Signature):
    image: Optional[dspy.Image] = dspy.InputField(default=None)
    answer: str = dspy.OutputField()


class ContainerImageSignature(dspy.Signature):
    images: dict[str, list[Optional[dspy.Image]]] = dspy.InputField()
    answer: str = dspy.OutputField()


class NestedPayload(BaseModel):
    caption: str
    images: Optional[list[dspy.Image]] = None


class NestedImageSignature(dspy.Signature):
    payload: NestedPayload = dspy.InputField()
    answer: str = dspy.OutputField()


class PayloadWithTypeField(BaseModel):
    type: str
    image: Optional[dspy.Image] = None


class TypeFieldImageSignature(dspy.Signature):
    payload: PayloadWithTypeField = dspy.InputField()
    answer: str = dspy.OutputField()


class DefaultTypeValueSignature(dspy.Signature):
    payload: dict = dspy.InputField(default={"type": ["not", "schema"]})
    answer: str = dspy.OutputField()


class RecursivePayload(BaseModel):
    image: Optional[dspy.Image] = None
    child: Optional[RecursivePayload] = None


class RecursiveImageSignature(dspy.Signature):
    payload: RecursivePayload = dspy.InputField()
    answer: str = dspy.OutputField()


class RecursiveMediaPayload(BaseModel):
    screenshot: Optional[Screenshot] = None
    child: Optional[RecursiveMediaPayload] = None


class RecursiveImageSubclassSignature(dspy.Signature):
    payload: RecursiveMediaPayload = dspy.InputField()
    answer: str = dspy.OutputField()


class OutputImageSignature(dspy.Signature):
    prompt: str = dspy.InputField()
    image: dspy.Image = dspy.OutputField()


class ImageAudioSignature(dspy.Signature):
    audio: Optional[dspy.Audio] = dspy.InputField(default=None)
    image: dspy.Image = dspy.InputField()
    answer: str = dspy.OutputField()


class HistorySignature(dspy.Signature):
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


class LocalOnlySignature(dspy.Signature):
    callback: Callable[..., str] = dspy.InputField()
    answer: str = dspy.OutputField()


def test_text_signature_requires_text_only():
    assert infer_required_modalities(TextSignature) == ["text"]


def test_direct_image_is_detected():
    assert infer_required_modalities(ImageSignature) == ["text", "image"]


def test_dspy_image_subclass_is_detected():
    assert infer_required_modalities(ImageSubclassSignature) == ["text", "image"]


def test_reused_dspy_image_subclass_is_detected():
    assert infer_required_modalities(ReusedImageSubclassSignature) == ["text", "image"]


def test_optional_and_list_dspy_image_subclass_is_detected():
    assert infer_required_modalities(ContainerImageSubclassSignature) == ["text", "image"]


def test_dspy_audio_subclass_is_detected():
    assert infer_required_modalities(AudioSubclassSignature) == ["text", "audio"]


def test_optional_image_is_detected():
    assert infer_required_modalities(OptionalImageSignature) == ["text", "image"]


def test_image_inside_containers_is_detected():
    assert infer_required_modalities(ContainerImageSignature) == ["text", "image"]


def test_image_inside_nested_model_is_detected():
    assert infer_required_modalities(NestedImageSignature) == ["text", "image"]


def test_all_of_wrapped_reference_is_detected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        modalities_module,
        "serialize_signature",
        lambda _signature: {
            "$defs": {"Screenshot": {"type": "dspy.Image"}},
            "properties": {
                "screenshot": {
                    "__dspy_field_type": "input",
                    "allOf": [{"$ref": "#/$defs/Screenshot"}],
                }
            },
        },
    )

    assert infer_required_modalities(TextSignature) == ["text", "image"]


def test_nested_field_named_type_does_not_confuse_schema_walker():
    assert infer_required_modalities(TypeFieldImageSignature) == ["text", "image"]


def test_default_value_named_type_is_not_walked_as_schema():
    assert infer_required_modalities(DefaultTypeValueSignature) == ["text"]


def test_recursive_schema_is_cycle_safe_and_detects_image():
    assert infer_required_modalities(RecursiveImageSignature) == ["text", "image"]


def test_recursive_schema_detects_image_subclass_from_shared_definition():
    assert infer_required_modalities(RecursiveImageSubclassSignature) == ["text", "image"]


def test_output_only_image_does_not_require_image_input_support():
    assert infer_required_modalities(OutputImageSignature) == ["text"]


def test_image_and_audio_use_canonical_order():
    assert infer_required_modalities(ImageAudioSignature) == ["text", "image", "audio"]


def test_history_is_a_text_requirement_not_an_inferred_media_capability():
    assert infer_required_modalities(HistorySignature) == ["text"]


def test_schema_generation_failure_warns_and_falls_back_to_text():
    with pytest.warns(
        RuntimeWarning,
        match=r"Could not infer input modalities.*Falling back to \['text'\]",
    ) as caught:
        assert infer_required_modalities(LocalOnlySignature) == ["text"]

    assert caught[0].filename == __file__


def test_schema_generation_failure_does_not_prevent_making_an_arbiter():
    predict = Predict(LocalOnlySignature, lm=dspy.LM("provider/llama-3.1-8b"))

    with pytest.warns(
        RuntimeWarning,
        match=r"explicitly declared metadata modalities are still preserved",
    ) as caught:
        arbiter = predict.as_arbiter()

    assert arbiter.metadata["modalities"] == ["text"]
    assert caught[0].filename == __file__


def test_direct_make_arbiter_schema_warning_points_to_the_sdk_caller():
    predict = Predict(LocalOnlySignature, lm=dspy.LM("provider/llama-3.1-8b"))

    with pytest.warns(RuntimeWarning) as caught:
        make_arbiter(predict)

    assert caught[0].filename == __file__


def test_schema_fallback_preserves_explicit_media_modalities():
    with pytest.warns(RuntimeWarning):
        assert merge_required_modalities(LocalOnlySignature, ["image"]) == [
            "text",
            "image",
        ]


def test_push_propagates_schema_generation_failure_before_hub_sync(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        PrecompiledProgram,
        "push_to_hub",
        lambda *_args, **_kwargs: pytest.fail("hub sync should not start"),
    )
    predict = Predict(LocalOnlySignature)

    with pytest.raises(PydanticInvalidForJsonSchema, match="CallableSchema"):
        predict.push_to_hub("owner/arbiter")


def test_declared_modalities_are_normalized_after_known_modalities():
    assert canonicalize_modalities(["VIDEO", "image"], "document", ["video", "audio"]) == [
        "text",
        "image",
        "audio",
        "document",
        "video",
    ]


def test_merge_cannot_downgrade_inferred_signature_requirements():
    assert merge_required_modalities(ImageSignature, ["text"]) == ["text", "image"]


def test_reconcile_explicit_modalities_removes_only_previous_inference():
    assert reconcile_explicit_modalities(
        ["text", "image", "audio"],
        previous_inferred=["text", "image"],
        previous_explicit=["text", "audio"],
    ) == ["text", "audio"]


def test_reconcile_explicit_modalities_preserves_legacy_stored_requirements():
    assert reconcile_explicit_modalities(["image"]) == ["text", "image"]


def test_reconcile_explicit_modalities_treats_deleted_metadata_as_reset():
    assert reconcile_explicit_modalities(
        None,
        previous_inferred=["text"],
        previous_explicit=["stale invalid value"],
    ) == ["text"]


@pytest.mark.parametrize(
    "modalities, error",
    [
        ({"image": True}, TypeError),
        (1, TypeError),
        ([None], TypeError),
        ("image, audio", ValueError),
        ([""], ValueError),
    ],
)
def test_declared_modalities_reject_malformed_values(
    modalities: object,
    error: type[Exception],
):
    with pytest.raises(error, match="modalit"):
        canonicalize_modalities(modalities)


def test_make_arbiter_sets_exact_image_modalities_metadata():
    arbiter = Predict(ImageSignature, lm=dspy.LM("provider/llama-3.1-8b")).as_arbiter()

    assert arbiter.metadata == {
        "is_arbiter": True,
        "model": "llama-3.1-8b",
        "size": "small",
        "modalities": ["text", "image"],
    }


def test_make_arbiter_merges_existing_declared_modalities():
    predict = Predict(ImageSignature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.metadata = {"modalities": ["audio"], "custom": True}

    arbiter = predict.as_arbiter()

    assert arbiter.metadata == {
        "modalities": ["text", "image", "audio"],
        "custom": True,
        "is_arbiter": True,
        "model": "llama-3.1-8b",
        "size": "small",
    }


def test_make_arbiter_infers_from_config_without_overwriting_optimized_runtime_instructions():
    optimized_signature = ImageReasoningSignature.with_instructions("GEPA-optimized image instructions.")
    predict = Predict(ImageReasoningSignature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.signature = optimized_signature

    arbiter = predict.as_arbiter()

    assert arbiter.metadata["modalities"] == ["text", "image"]
    assert arbiter.signature.instructions == "GEPA-optimized image instructions."
    assert arbiter.config.signature.instructions == "Base image-judging instructions."


def test_make_arbiter_reasoning_fields_stay_aligned_after_disk_round_trip(tmp_path: Path):
    optimized_signature = dspy.make_signature(
        {name: (field.annotation, copy.deepcopy(field)) for name, field in ImageInstructionSignature.fields.items()},
        instructions="GEPA-optimized image instructions.",
        signature_name="OptimizedImageInstructionSignature",
    )
    predict = Predict(ImageInstructionSignature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.signature = optimized_signature

    arbiter = predict.as_arbiter()
    assert arbiter.signature.__name__ == "OptimizedImageInstructionSignature"
    assert arbiter.config.signature.__name__ == "ImageInstructionSignature"
    arbiter.save_precompiled(tmp_path)
    restored = Predict.from_precompiled(tmp_path)

    assert restored.config.signature.__name__ == "ImageInstructionSignature"
    assert restored.signature.instructions == "GEPA-optimized image instructions."
    assert restored.config.signature.instructions == "Base image instructions."
    assert list(restored.signature.fields) == list(restored.config.signature.fields)
    assert restored.signature.output_fields["reasoning"].annotation is dspy.Reasoning
    assert restored.config.signature.output_fields["reasoning"].annotation is dspy.Reasoning


def test_make_arbiter_accepts_structurally_equivalent_types_after_disk_round_trip(tmp_path: Path):
    Predict(StructuredSignature, lm=dspy.LM("provider/llama-3.1-8b")).save_precompiled(tmp_path)
    restored = Predict.from_precompiled(tmp_path)
    restored.signature = StructuredSignature

    assert restored.config.signature.input_fields["request"].annotation is not StructuredInput

    arbiter = restored.as_arbiter()

    assert list(arbiter.signature.fields) == list(arbiter.config.signature.fields)
    assert arbiter.metadata["modalities"] == ["text"]


def test_make_arbiter_aligns_existing_runtime_reasoning_position_after_disk_round_trip(tmp_path: Path):
    runtime_signature = MultiOutputImageSignature.insert(
        0,
        "reasoning",
        dspy.OutputField(desc="Optimized reasoning"),
        dspy.Reasoning,
    ).with_instructions("GEPA-optimized multi-output instructions.")
    predict = Predict(MultiOutputImageSignature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.signature = runtime_signature

    arbiter = predict.as_arbiter()
    assert list(arbiter.signature.output_fields) == ["reasoning", "summary", "answer"]
    assert list(arbiter.config.signature.output_fields) == ["reasoning", "summary", "answer"]
    assert arbiter.config.signature.__name__ == "MultiOutputImageSignature"
    assert arbiter.signature.instructions == "GEPA-optimized multi-output instructions."
    assert arbiter.config.signature.instructions == "Base multi-output image instructions."

    arbiter.save_precompiled(tmp_path)
    restored = Predict.from_precompiled(tmp_path)

    assert list(restored.signature.output_fields) == ["reasoning", "summary", "answer"]
    assert list(restored.config.signature.output_fields) == ["reasoning", "summary", "answer"]
    assert restored.config.signature.__name__ == "MultiOutputImageSignature"
    assert restored.signature.output_fields["reasoning"].json_schema_extra["desc"] == "Optimized reasoning"
    assert restored.signature.output_fields["summary"].json_schema_extra["desc"] == "Short summary"
    assert restored.signature.instructions == "GEPA-optimized multi-output instructions."
    assert restored.config.signature.instructions == "Base multi-output image instructions."


def test_make_arbiter_adds_published_reasoning_to_runtime_at_the_same_position():
    published_signature = MultiOutputImageSignature.insert(
        1,
        "reasoning",
        dspy.OutputField(desc="Published reasoning"),
        dspy.Reasoning,
    )
    predict = Predict(published_signature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.signature = MultiOutputImageSignature

    arbiter = predict.as_arbiter()

    assert list(arbiter.signature.output_fields) == ["summary", "reasoning", "answer"]
    assert list(arbiter.config.signature.output_fields) == ["summary", "reasoning", "answer"]
    assert arbiter.signature.output_fields["reasoning"].annotation is dspy.Reasoning


def test_make_arbiter_moves_published_reasoning_to_runtime_position():
    runtime_signature = MultiOutputImageSignature.insert(
        0,
        "reasoning",
        dspy.OutputField(desc="Runtime reasoning"),
        dspy.Reasoning,
    )
    published_signature = MultiOutputImageSignature.insert(
        1,
        "reasoning",
        dspy.OutputField(desc="Published reasoning"),
        dspy.Reasoning,
    )
    predict = Predict(published_signature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.signature = runtime_signature

    arbiter = predict.as_arbiter()

    assert list(arbiter.signature.output_fields) == ["reasoning", "summary", "answer"]
    assert list(arbiter.config.signature.output_fields) == ["reasoning", "summary", "answer"]
    assert arbiter.config.signature.output_fields["reasoning"].json_schema_extra["desc"] == "Published reasoning"


def test_make_arbiter_normalizes_legacy_string_reasoning_at_matching_position():
    runtime_signature = MultiOutputImageSignature.insert(
        0,
        "reasoning",
        dspy.OutputField(desc="Runtime reasoning"),
        dspy.Reasoning,
    )
    published_signature = MultiOutputImageSignature.insert(
        0,
        "reasoning",
        dspy.OutputField(desc="Legacy published reasoning"),
        str,
    )
    predict = Predict(published_signature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.signature = runtime_signature

    arbiter = predict.as_arbiter()

    assert arbiter.signature.output_fields["reasoning"].annotation is dspy.Reasoning
    assert arbiter.config.signature.output_fields["reasoning"].annotation is dspy.Reasoning
    assert arbiter.config.signature.output_fields["reasoning"].json_schema_extra["desc"] == (
        "Legacy published reasoning"
    )


@pytest.mark.parametrize(
    "runtime_signature",
    [ImageSignature, ReorderedMultiOutputImageSignature],
    ids=["different-output-count", "different-output-order"],
)
def test_make_arbiter_rejects_runtime_published_output_structure_mismatch(runtime_signature: type[dspy.Signature]):
    predict = Predict(MultiOutputImageSignature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.signature = runtime_signature

    with pytest.raises(ValueError, match="reasoning structural mismatch.*non-reasoning output fields"):
        predict.as_arbiter()


def test_make_arbiter_rejects_runtime_published_input_structure_mismatch():
    predict = Predict(MultiOutputImageSignature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.signature = AlternateInputMultiOutputImageSignature

    with pytest.raises(ValueError, match="reasoning structural mismatch.*input fields"):
        predict.as_arbiter()


@pytest.mark.parametrize(
    ("runtime_signature", "field_kind"),
    [
        (AlternateInputTypeMultiOutputImageSignature, "input"),
        (AlternateOutputTypeMultiOutputImageSignature, "output"),
    ],
)
def test_make_arbiter_rejects_runtime_published_field_type_mismatch(
    runtime_signature: type[dspy.Signature],
    field_kind: str,
):
    predict = Predict(MultiOutputImageSignature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.signature = runtime_signature

    with pytest.raises(ValueError, match=rf"reasoning structural mismatch.*{field_kind} field.*different types"):
        predict.as_arbiter()


def test_make_arbiter_malformed_existing_modalities_identifies_metadata_source():
    predict = Predict(TextSignature, lm=dspy.LM("provider/llama-3.1-8b"))
    predict.metadata = {"modalities": {"image": True}}

    with pytest.raises(TypeError, match="existing arbiter metadata.*README/Hub"):
        predict.as_arbiter()


def test_predict_push_merges_local_caller_and_inferred_modalities(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def capture_push(_self: PrecompiledProgram, **kwargs: object) -> str:
        captured.update(kwargs)
        return "commit"

    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", capture_push)
    predict = Predict(ImageSignature)
    predict.metadata = {"modalities": ["video", "audio"], "source": "local"}

    result = predict.push_to_hub(
        "owner/arbiter",
        metadata={"modalities": ["text", "document"], "source": "caller"},
    )

    assert result == "commit"
    assert captured["metadata"] == {
        "modalities": ["text", "image", "audio", "document", "video"],
        "source": "caller",
    }
    assert predict.metadata == {
        "modalities": ["text", "image", "audio", "document", "video"],
        "source": "local",
    }


def test_predict_push_infers_from_published_config_signature(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def capture_push(_self: PrecompiledProgram, **kwargs: object) -> str:
        captured.update(kwargs)
        return "commit"

    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", capture_push)
    predict = Predict(ImageSignature)
    predict.signature = TextSignature

    assert predict.push_to_hub("owner/arbiter") == "commit"
    assert captured["metadata"] == {"modalities": ["text", "image"]}


def test_predict_push_removes_stale_inferred_modality_after_signature_change(monkeypatch: pytest.MonkeyPatch):
    captured: list[dict[str, object]] = []

    def capture_push(_self: PrecompiledProgram, **kwargs: object) -> str:
        captured.append(kwargs)
        return "commit"

    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", capture_push)
    predict = Predict(ImageSignature)

    assert predict.push_to_hub("owner/arbiter") == "commit"
    predict.config.signature = TextSignature
    assert predict.push_to_hub("owner/arbiter") == "commit"

    assert captured[0]["metadata"] == {"modalities": ["text", "image"]}
    assert captured[1]["metadata"] == {"modalities": ["text"]}


def test_as_arbiter_push_then_signature_change_drops_only_inferred_modality(monkeypatch: pytest.MonkeyPatch):
    captured: list[dict[str, object]] = []

    def capture_push(_self: PrecompiledProgram, **kwargs: object) -> str:
        captured.append(kwargs)
        return "commit"

    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", capture_push)
    arbiter = Predict(ImageSignature, lm=dspy.LM("provider/llama-3.1-8b")).as_arbiter()

    assert arbiter.push_to_hub("owner/arbiter") == "commit"
    arbiter.config.signature = TextSignature
    assert arbiter.push_to_hub("owner/arbiter") == "commit"

    assert captured[0]["metadata"]["modalities"] == ["text", "image"]
    assert captured[1]["metadata"]["modalities"] == ["text"]
    assert arbiter.config.modality_provenance == {
        "inferred": ["text"],
        "explicit": ["text"],
    }


def test_modality_provenance_round_trips_in_predict_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", lambda *_args, **_kwargs: "commit")
    predict = Predict(ImageSignature)

    assert "modality_provenance" not in predict.config.model_dump()
    assert "modality_provenance" not in json.loads(predict.config.model_dump_json())
    assert "modality_provenance" not in predict.config.to_dict()
    assert predict.push_to_hub("owner/arbiter") == "commit"

    restored = PredictConfig.from_dict(predict.config.to_dict())
    assert restored.modality_provenance == {
        "inferred": ["text", "image"],
        "explicit": ["text"],
    }


def test_modality_provenance_preserves_unknown_future_keys():
    config = PredictConfig(
        signature=TextSignature,
        modality_provenance={
            "inferred": ["IMAGE"],
            "explicit": ["text"],
            "future": {"version": 2},
        },
    )

    restored = PredictConfig.from_dict(config.to_dict())
    assert restored.modality_provenance == {
        "inferred": ["text", "image"],
        "explicit": ["text"],
        "future": {"version": 2},
    }


@pytest.mark.parametrize(
    "malformed_provenance",
    [
        "image",
        {"inferred": ["text"]},
        {"inferred": ["not a modality"], "explicit": ["text"]},
        {"inferred": "image", "explicit": ["text"]},
    ],
)
def test_malformed_modality_provenance_loaded_from_config_identifies_source(
    malformed_provenance: object,
    tmp_path: Path,
):
    Predict(TextSignature).save_precompiled(tmp_path)
    config_path = tmp_path / "config.json"
    with config_path.open() as config_file:
        config = json.load(config_file)
    config["modality_provenance"] = malformed_provenance
    with config_path.open("w") as config_file:
        json.dump(config, config_file)

    with pytest.raises(ValueError, match=r"Invalid modality_provenance in Predict config \(config\.json\)"):
        Predict.from_precompiled(tmp_path)


def test_disk_loaded_readme_repush_drops_only_stale_inferred_modality(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    captured: list[dict[str, object]] = []

    def capture_push(_self: PrecompiledProgram, **kwargs: object) -> str:
        captured.append(kwargs)
        return "commit"

    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", capture_push)
    predict = Predict(ImageSignature)
    assert predict.push_to_hub("owner/arbiter") == "commit"
    predict.save_precompiled(tmp_path)
    add_metadata_to_readme(tmp_path / "README.md", predict.metadata)

    restored = Predict.from_precompiled(tmp_path)
    assert restored.config.modality_provenance == {
        "inferred": ["text", "image"],
        "explicit": ["text"],
    }
    assert restored.metadata["modalities"] == ["text", "image"]

    restored.config.signature = TextSignature
    assert restored.push_to_hub("owner/arbiter") == "commit"
    assert captured[-1]["metadata"] == {"modalities": ["text"]}


def test_legacy_readme_without_provenance_preserves_modalities_on_repush(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", lambda *_args, **_kwargs: "commit")
    predict = Predict(TextSignature)
    predict.save_precompiled(tmp_path)
    with (tmp_path / "config.json").open() as config_file:
        assert "modality_provenance" not in json.load(config_file)
    add_metadata_to_readme(tmp_path / "README.md", {"modalities": ["image"]})

    restored = Predict.from_precompiled(tmp_path)
    assert restored.config.modality_provenance is None
    assert restored.push_to_hub("owner/arbiter") == "commit"

    assert restored.metadata["modalities"] == ["text", "image"]
    assert restored.config.modality_provenance == {
        "inferred": ["text"],
        "explicit": ["text", "image"],
    }


def test_predict_push_preserves_caller_declared_modality_until_explicit_reset(monkeypatch: pytest.MonkeyPatch):
    captured: list[dict[str, object]] = []

    def capture_push(_self: PrecompiledProgram, **kwargs: object) -> str:
        captured.append(kwargs)
        return "commit"

    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", capture_push)
    predict = Predict(TextSignature)

    assert predict.push_to_hub("owner/arbiter", metadata={"modalities": ["image"]}) == "commit"
    assert predict.push_to_hub("owner/arbiter") == "commit"
    assert captured[0]["metadata"] == {"modalities": ["text", "image"]}
    assert captured[1]["metadata"] == {"modalities": ["text", "image"]}

    del predict.metadata["modalities"]
    assert predict.push_to_hub("owner/arbiter") == "commit"

    assert captured[2]["metadata"] == {"modalities": ["text"]}


def test_predict_push_malformed_stored_modalities_identifies_readme_source(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", lambda *_args, **_kwargs: "commit")
    predict = Predict(TextSignature)
    predict.metadata = {"modalities": {"image": True}}

    with pytest.raises(TypeError, match="stored Predict metadata.*README/Hub"):
        predict.push_to_hub("owner/arbiter")


def test_predict_push_malformed_caller_modalities_identifies_argument(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", lambda *_args, **_kwargs: "commit")
    predict = Predict(TextSignature)

    with pytest.raises(ValueError, match=r"push_to_hub\(metadata=\.\.\.\) caller metadata"):
        predict.push_to_hub("owner/arbiter", metadata={"modalities": "image, audio"})


def test_predict_push_keeps_existing_probe_when_new_validation_fails():
    predict = Predict(TextSignature)
    existing_probe = object()
    replacement_probe = object()
    predict.probe = existing_probe

    with pytest.raises(ValueError, match=r"push_to_hub\(metadata=\.\.\.\) caller metadata"):
        predict.push_to_hub(
            "owner/arbiter",
            metadata={"modalities": "image, audio"},
            probe=replacement_probe,
        )

    assert predict.probe is existing_probe


@pytest.mark.parametrize("failure", [RuntimeError("failed"), KeyboardInterrupt()])
def test_predict_push_restores_state_when_hub_sync_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
):
    def fail_push(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(PrecompiledProgram, "push_to_hub", fail_push)
    predict = Predict(TextSignature)
    existing_probe = object()
    replacement_probe = object()
    old_provenance = {"inferred": ["text"], "explicit": ["audio"]}
    predict.probe = existing_probe
    predict.config.modality_provenance = old_provenance

    with pytest.raises(type(failure), match="failed" if isinstance(failure, RuntimeError) else None):
        predict.push_to_hub("owner/arbiter", probe=replacement_probe)

    assert predict.probe is existing_probe
    assert predict.config.modality_provenance is old_provenance


def test_predict_push_modalities_survive_readme_frontmatter_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    readme = tmp_path / "README.md"

    def local_sync(_program: PrecompiledProgram, _repo_path: str, **kwargs: object) -> str:
        add_metadata_to_readme(readme, kwargs["metadata"])
        return "commit"

    monkeypatch.setattr("modaic.precompiled.sync_and_push", local_sync)
    predict = Predict(ImageAudioSignature)

    assert predict.push_to_hub("owner/arbiter") == "commit"
    assert load_metadata_from_readme(readme) == {"modalities": ["text", "image", "audio"]}
