import copy
import json
import os
import typing as t
from pathlib import Path
from typing import Optional

import dspy
import modaic
import pytest
from dspy.adapters.utils import parse_value
from modaic import PrecompiledConfig, PrecompiledProgram, SerializableSignature
from modaic.serializers import _deserialize_dspy_signatures, serialize_signature
from modaic.utils import smart_rmtree
from pydantic import BaseModel


@pytest.fixture
def clean_folder() -> Path:
    smart_rmtree("tests/artifacts/temp/test_precompiled", ignore_errors=True)
    os.makedirs("tests/artifacts/temp/test_precompiled")
    return Path("tests/artifacts/temp/test_precompiled")


class CustomModel(BaseModel):
    my_model_string: str = dspy.InputField()
    my_model_int: int = dspy.InputField()


class SerializableSig(dspy.Signature):
    """Classify the emotion in the sentence."""

    my_string: str = dspy.InputField(desc="A string")
    my_int: int = dspy.InputField(desc="An integer", default=1)
    my_float: float = dspy.InputField()
    my_basic_list: list = dspy.InputField()
    my_list: list[int] = dspy.InputField(default=[1, 2, 3], desc="A list of integers")
    my_basic_set: set = dspy.InputField()
    my_image: dspy.Image = dspy.InputField()
    my_audio: dspy.Audio = dspy.InputField()
    my_history: dspy.History = dspy.InputField()
    my_code: dspy.Code = dspy.InputField()
    my_custom_model: CustomModel = dspy.InputField(default=CustomModel(my_model_string="Hello", my_model_int=1))
    my_tool: dspy.Tool = dspy.InputField()
    my_tool_list: list[dspy.Tool] = dspy.InputField()
    my_tool_dict: dict[str, dspy.Tool] = dspy.InputField()
    my_image_list: list[dspy.Image] = dspy.InputField()
    my_default_dict: dict[str, int] = dspy.InputField(default={"a": 1, "b": 2})
    my_optional_dict: Optional[dict[str, int]] = dspy.InputField(default=None)

    my_set: set[int] = dspy.OutputField()
    my_tool_calls: dspy.ToolCalls = dspy.OutputField()
    my_tuple: tuple[int, str] = dspy.OutputField()
    my_basic_dict: dict = dspy.OutputField()
    my_dict: dict[str, int | str] = dspy.OutputField()
    custom_model_list: list[CustomModel] = dspy.OutputField()


class HistoryWithSource(dspy.History):
    source: str


class CodeWithDialect(dspy.Code):
    dialect: str


class ToolCallsWithTrace(dspy.ToolCalls):
    trace_id: str


class NonMediaDspySubclassSig(dspy.Signature):
    history: HistoryWithSource = dspy.InputField()
    code: CodeWithDialect = dspy.InputField()
    calls: ToolCallsWithTrace = dspy.OutputField()


class ImageWithCaption(dspy.Image):
    caption: str


class AudioWithLanguage(dspy.Audio):
    language: str


class MediaDspySubclassSig(dspy.Signature):
    image: ImageWithCaption = dspy.InputField()
    second_image: ImageWithCaption = dspy.InputField()
    optional_image: Optional[ImageWithCaption] = dspy.InputField(default=None)
    images: list[ImageWithCaption] = dspy.InputField()
    audio: AudioWithLanguage = dspy.InputField()
    answer: str = dspy.OutputField()


class FalsyDefaultsPayload(BaseModel):
    absent: Optional[str] = None
    zero: int = 0
    empty: str = ""
    disabled: bool = False


class FalsyDefaultsSignature(dspy.Signature):
    absent: Optional[str] = dspy.InputField(default=None)
    zero: int = dspy.InputField(default=0)
    empty: str = dspy.InputField(default="")
    disabled: bool = dspy.InputField(default=False)
    payload: FalsyDefaultsPayload = dspy.InputField()
    answer: str = dspy.OutputField()


class Summarize(dspy.Signature):
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Answer to the question, based on the passage")


class ConfigWithSignature(PrecompiledConfig):
    """Config that includes a DSPy signature as a field."""

    signature: SerializableSignature
    lm: str = "openai/gpt-4o-mini"


class ProgramWithSignatureConfig(PrecompiledProgram):
    """Program that uses a config with a DSPy signature."""

    config: ConfigWithSignature

    def __init__(self, config: ConfigWithSignature, **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(config.signature)
        self.predictor.set_lm(lm=dspy.LM(config.lm))

    def forward(self, **kwargs) -> str:
        return self.predictor(**kwargs)


def test_config_with_dspy_signature_local(clean_folder: Path):
    """Test that configs with DSPy signatures can be serialized and deserialized."""
    config = ConfigWithSignature(signature=SerializableSig)
    config.save_precompiled(clean_folder)

    assert os.path.exists(clean_folder / "config.json")

    # Verify the signature was serialized correctly
    with open(clean_folder / "config.json", "r") as f:
        config_json = json.load(f)
    assert "signature" in config_json

    # Load the config back
    loaded_config = ConfigWithSignature.from_precompiled(clean_folder)
    assert loaded_config.signature.equals(SerializableSig)
    assert loaded_config.lm == "openai/gpt-4o-mini"

    # Test with different signature
    config2 = ConfigWithSignature(signature=Summarize, lm="openai/gpt-4o")
    config2.save_precompiled(clean_folder)
    loaded_config2 = ConfigWithSignature.from_precompiled(clean_folder)
    assert loaded_config2.signature.equals(Summarize)
    assert loaded_config2.lm == "openai/gpt-4o"


def test_program_with_dspy_signature_local(clean_folder: Path):
    """Test that programs with DSPy signature configs can be saved and loaded."""
    config = ConfigWithSignature(signature=SerializableSig)
    program = ProgramWithSignatureConfig(config=config)
    program.save_precompiled(clean_folder)

    assert os.path.exists(clean_folder / "config.json")
    assert os.path.exists(clean_folder / "program.json")

    # Verify the signature was serialized correctly
    with open(clean_folder / "config.json", "r") as f:
        config_json = json.load(f)
    assert "signature" in config_json

    # Load the program back
    loaded_program = ProgramWithSignatureConfig.from_precompiled(clean_folder)
    assert loaded_program.config.signature.equals(SerializableSig)
    assert loaded_program.config.lm == "openai/gpt-4o-mini"


def _round_trip(sig):  # noqa
    """Helper: serialize then deserialize a signature and assert equality."""
    serialized = serialize_signature(sig)
    deserialized = _deserialize_dspy_signatures(serialized)
    assert deserialized.equals(sig), (
        f"Round-trip failed.\nOriginal fields: {dict(sig.fields)}\nDeserialized fields: {dict(deserialized.fields)}"
    )
    return deserialized


def test_serialization_does_not_mutate_dspy_shared_core_schema():
    before = copy.deepcopy(dspy.Image.__pydantic_core_schema__)

    serialize_signature(SerializableSig)

    assert dspy.Image.__pydantic_core_schema__ == before


def test_non_media_dspy_subclasses_keep_their_structural_schema():
    definitions = serialize_signature(NonMediaDspySubclassSig)["$defs"]

    for definition_name, added_field in (
        ("HistoryWithSource", "source"),
        ("CodeWithDialect", "dialect"),
        ("ToolCallsWithTrace", "trace_id"),
    ):
        definition = definitions[definition_name]
        assert definition["type"] == "object"
        assert added_field in definition["properties"]


def test_media_dspy_subclasses_preserve_fields_and_base_type_on_round_trip():
    serialized = serialize_signature(MediaDspySubclassSig)

    for definition_name, base_type, added_field in (
        ("ImageWithCaption", dspy.Image, "caption"),
        ("AudioWithLanguage", dspy.Audio, "language"),
    ):
        definition = serialized["$defs"][definition_name]
        assert definition["x-modaic-dspy-base-type"] == f"dspy.{base_type.__name__}"
        assert added_field in definition["properties"]

    assert "x-modaic-dspy-base-type" not in serialized["properties"]["image"]

    deserialized = _round_trip(MediaDspySubclassSig)
    for annotation, expected_name, base_type, added_field in (
        (deserialized.input_fields["image"].annotation, "ImageWithCaption", dspy.Image, "caption"),
        (deserialized.input_fields["second_image"].annotation, "ImageWithCaption", dspy.Image, "caption"),
        (
            t.get_args(deserialized.input_fields["optional_image"].annotation)[0],
            "ImageWithCaption",
            dspy.Image,
            "caption",
        ),
        (t.get_args(deserialized.input_fields["images"].annotation)[0], "ImageWithCaption", dspy.Image, "caption"),
        (deserialized.input_fields["audio"].annotation, "AudioWithLanguage", dspy.Audio, "language"),
    ):
        assert annotation.__name__ == expected_name
        assert issubclass(annotation, base_type)
        assert added_field in annotation.model_fields

    assert serialize_signature(deserialized) == serialized


def test_deserialized_image_subclass_constructs_and_formats_like_dspy_image():
    deserialized = _deserialize_dspy_signatures(serialize_signature(MediaDspySubclassSig))
    image_type = deserialized.input_fields["image"].annotation

    image = image_type("https://example.com/input.png", caption="primary")

    assert image.caption == "primary"
    assert image.format() == [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/input.png"},
        }
    ]


def test_falsy_defaults_survive_signature_and_nested_model_round_trip():
    serialized = serialize_signature(FalsyDefaultsSignature)
    deserialized = _deserialize_dspy_signatures(serialized)

    expected_defaults = {
        "absent": None,
        "zero": 0,
        "empty": "",
        "disabled": False,
    }
    for field_name, expected in expected_defaults.items():
        assert deserialized.input_fields[field_name].default == expected

    payload_type = deserialized.input_fields["payload"].annotation
    for field_name, expected in expected_defaults.items():
        assert payload_type.model_fields[field_name].default == expected

    assert serialize_signature(deserialized) == serialized


def test_dynamic_signature_append():
    """Test that signatures created with .append() can be serialized and deserialized."""
    sig = Summarize.append("confidence", dspy.OutputField(desc="Confidence score"), float)

    deserialized = _round_trip(sig)

    assert "confidence" in dict(deserialized.output_fields)
    assert deserialized.output_fields["confidence"].json_schema_extra["desc"] == "Confidence score"


def test_dynamic_signature_prepend():
    """Test that signatures created with .prepend() can be serialized and deserialized."""
    sig = Summarize.prepend("system_prompt", dspy.InputField(desc="System prompt"), str)

    deserialized = _round_trip(sig)

    assert "system_prompt" in dict(deserialized.input_fields)
    input_names = list(dict(deserialized.input_fields).keys())
    assert input_names[0] == "system_prompt"


def test_dynamic_signature_insert():
    """Test that signatures created with .insert() can be serialized and deserialized."""
    sig = Summarize.insert(1, "hint", dspy.InputField(desc="A hint"), str)

    deserialized = _round_trip(sig)

    assert "hint" in dict(deserialized.input_fields)
    input_names = list(dict(deserialized.input_fields).keys())
    assert input_names[1] == "hint"


def test_dynamic_signature_chained():
    """Test that chaining append, prepend, and insert produces a serializable signature."""
    sig = (
        Summarize.append("confidence", dspy.OutputField(desc="Confidence"), float)
        .prepend("system_prompt", dspy.InputField(desc="System prompt"), str)
        .insert(2, "hint", dspy.InputField(desc="A hint"), str)
    )

    deserialized = _round_trip(sig)

    assert "system_prompt" in dict(deserialized.input_fields)
    assert "hint" in dict(deserialized.input_fields)
    assert "confidence" in dict(deserialized.output_fields)
    assert "answer" in dict(deserialized.output_fields)


class LiteralSig(dspy.Signature):
    """Classify the category and status."""

    query: str = dspy.InputField()
    category: t.Literal["a", "b", "c"] = dspy.OutputField()
    status: t.Literal["ok"] = dspy.OutputField()
    count: t.Literal[1, 2, 3] = dspy.OutputField()


def test_literal_string_enum_round_trip():
    """Literal with multiple string values should survive serialization round-trip."""
    deserialized = _round_trip(LiteralSig)
    assert deserialized.output_fields["category"].annotation == t.Literal["a", "b", "c"]


def test_literal_string_const_round_trip():
    """Literal with a single string value (serialized as const) should survive round-trip."""
    deserialized = _round_trip(LiteralSig)
    assert deserialized.output_fields["status"].annotation == t.Literal["ok"]


def test_literal_int_enum_round_trip():
    """Literal with integer values should survive serialization round-trip."""
    deserialized = _round_trip(LiteralSig)
    assert deserialized.output_fields["count"].annotation == t.Literal[1, 2, 3]


class ScaleEnumSig(dspy.Signature):
    """Rate and decide."""

    question: str = dspy.InputField()
    rating: modaic.Scale[1, 5] = dspy.OutputField(desc="1-5")
    single: modaic.Scale[3, 3] = dspy.OutputField()
    decision: modaic.Enum["YES", "NO", "MAYBE"] = dspy.OutputField()  # noqa
    only: modaic.Enum["ONLY"] = dspy.OutputField()  # noqa


def test_scale_serialize_emits_marker():
    """Scale serializes to its Literal enum/const plus the round-trip marker keys."""
    props = serialize_signature(ScaleEnumSig)["properties"]
    assert props["rating"]["enum"] == [1, 2, 3, 4, 5]
    assert props["rating"]["x-modaic-type"] == "Scale"
    assert props["rating"]["x-modaic-args"] == [1, 5]
    # a degenerate Scale[n, n] is a single-value Literal => const, not enum
    assert props["single"]["const"] == 3
    assert props["single"]["x-modaic-args"] == [3, 3]


def test_enum_serialize_emits_marker():
    """Enum serializes to its Literal enum/const plus the round-trip marker keys."""
    props = serialize_signature(ScaleEnumSig)["properties"]
    assert props["decision"]["enum"] == ["YES", "NO", "MAYBE"]
    assert props["decision"]["x-modaic-type"] == "Enum"
    assert props["decision"]["x-modaic-args"] == ["YES", "NO", "MAYBE"]
    assert props["only"]["const"] == "ONLY"
    assert props["only"]["x-modaic-args"] == ["ONLY"]


def test_scale_round_trip_preserves_annotation():
    """modaic.Scale must round-trip as a Scale, not collapse to a plain Literal."""
    deserialized = _round_trip(ScaleEnumSig)

    # Scale[lo, hi] is memoized, so a faithful round-trip yields the *same* annotation object.
    rating = deserialized.output_fields["rating"].annotation
    assert rating is modaic.Scale[1, 5]
    assert rating.__args__ == (1, 2, 3, 4, 5)

    single = deserialized.output_fields["single"].annotation
    assert single is modaic.Scale[3, 3]
    assert single.__args__ == (3,)


def test_enum_round_trip_preserves_annotation():
    """modaic.Enum must round-trip as an Enum, not collapse to a plain Literal."""
    deserialized = _round_trip(ScaleEnumSig)

    decision = deserialized.output_fields["decision"].annotation
    assert decision is modaic.Enum["YES", "NO", "MAYBE"]
    assert decision.__args__ == ("YES", "NO", "MAYBE")

    only = deserialized.output_fields["only"].annotation
    assert only is modaic.Enum["ONLY"]
    assert only.__args__ == ("ONLY",)


def test_scale_enum_round_trip_is_idempotent():
    """Serializing the deserialized signature reproduces the original schema exactly."""
    once = serialize_signature(ScaleEnumSig)
    twice = serialize_signature(_deserialize_dspy_signatures(once))
    assert once == twice


def test_scale_enum_round_trip_preserves_coercion():
    """The deserialized annotation keeps the loose-coercion validator (matches the original)."""
    deserialized = _round_trip(ScaleEnumSig)

    rating = deserialized.output_fields["rating"].annotation
    assert parse_value("3.", rating) == 3
    assert parse_value("(4)", rating) == 4
    with pytest.raises(Exception):  # noqa
        parse_value("9", rating)

    decision = deserialized.output_fields["decision"].annotation
    assert parse_value("yes", decision) == "YES"


def test_dynamic_signature_insert_dspy_reasoning():
    """Test that inserting a dspy.Reasoning field works (mirrors arbiters.py usage)."""
    sig = Summarize.insert(-1, "reasoning", dspy.OutputField(desc="Your reasoning"), dspy.Reasoning)

    deserialized = _round_trip(sig)

    assert "reasoning" in dict(deserialized.output_fields)


def test_dynamic_signature_precompiled_round_trip(clean_folder: Path):
    """Test that dynamically-created signatures survive PrecompiledConfig save/load."""
    sig = Summarize.append("confidence", dspy.OutputField(desc="Confidence"), float).insert(
        -1, "reasoning", dspy.OutputField(desc="Reasoning"), dspy.Reasoning
    )

    config = ConfigWithSignature(signature=sig)
    config.save_precompiled(clean_folder)

    loaded_config = ConfigWithSignature.from_precompiled(clean_folder)
    assert loaded_config.signature.equals(sig)
    assert "confidence" in dict(loaded_config.signature.output_fields)
    assert "reasoning" in dict(loaded_config.signature.output_fields)
