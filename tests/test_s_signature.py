import json
import os
from pathlib import Path
from typing import Optional

import dspy
import pytest
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


def _round_trip(sig):
    """Helper: serialize then deserialize a signature and assert equality."""
    serialized = serialize_signature(sig)
    deserialized = _deserialize_dspy_signatures(serialized)
    assert deserialized.equals(sig), (
        f"Round-trip failed.\nOriginal fields: {dict(sig.fields)}\n"
        f"Deserialized fields: {dict(deserialized.fields)}"
    )
    return deserialized


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


def test_dynamic_signature_insert_dspy_reasoning():
    """Test that inserting a dspy.Reasoning field works (mirrors arbiters.py usage)."""
    sig = Summarize.insert(
        -1, "reasoning", dspy.OutputField(desc="Your reasoning"), dspy.Reasoning
    )

    deserialized = _round_trip(sig)

    assert "reasoning" in dict(deserialized.output_fields)


def test_dynamic_signature_precompiled_round_trip(clean_folder: Path):
    """Test that dynamically-created signatures survive PrecompiledConfig save/load."""
    sig = (
        Summarize.append("confidence", dspy.OutputField(desc="Confidence"), float)
        .insert(-1, "reasoning", dspy.OutputField(desc="Reasoning"), dspy.Reasoning)
    )

    config = ConfigWithSignature(signature=sig)
    config.save_precompiled(clean_folder)

    loaded_config = ConfigWithSignature.from_precompiled(clean_folder)
    assert loaded_config.signature.equals(sig)
    assert "confidence" in dict(loaded_config.signature.output_fields)
    assert "reasoning" in dict(loaded_config.signature.output_fields)
