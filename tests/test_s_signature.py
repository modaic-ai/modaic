import json
import os
from pathlib import Path
from typing import Optional

import dspy
import pytest
from pydantic import BaseModel

from modaic import PrecompiledConfig, PrecompiledProgram, SerializableSignature
from modaic.utils import smart_rmtree


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
