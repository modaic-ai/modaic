from pathlib import Path
from typing import Literal, get_args, get_origin

import dspy
import pytest

from modaic import Predict
from modaic.programs.arbiters import ARBITER_PROBES
from modaic.programs.utils import PredictField, PredictYamlSpec

YAML_DIR = Path(__file__).parent / "artifacts" / "yaml"


class TestFromYaml:
    def test_basic(self):
        pred = Predict.from_yaml(YAML_DIR / "summarizer.yaml")
        sig = pred.config.signature
        assert "text" in sig.input_fields
        assert "summary" in sig.output_fields
        assert sig.input_fields["text"].annotation is str
        assert sig.output_fields["summary"].annotation is str

    def test_with_options(self):
        pred = Predict.from_yaml(YAML_DIR / "spam_classifier.yaml")
        sig = pred.config.signature
        is_spam_type = sig.output_fields["is_spam"].annotation
        assert get_origin(is_spam_type) is Literal
        assert set(get_args(is_spam_type)) == {"spam", "not spam"}

    def test_model_set(self):
        pred = Predict.from_yaml(YAML_DIR / "summarizer.yaml")
        assert pred.lm is not None
        assert pred.lm.model == "openai/gpt-4o-mini"

    def test_instructions(self):
        pred = Predict.from_yaml(YAML_DIR / "summarizer.yaml")
        sig = pred.config.signature
        assert sig.__doc__ == "Summarize the given text concisely"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            Predict.from_yaml(YAML_DIR / "nonexistent.yaml")

    def test_multiple_inputs_outputs(self):
        pred = Predict.from_yaml(YAML_DIR / "sentiment.yaml")
        sig = pred.config.signature
        assert set(sig.input_fields.keys()) == {"title", "review_text", "rating"}
        assert set(sig.output_fields.keys()) == {"sentiment", "explanation"}
        assert sig.input_fields["rating"].annotation is int
        assert sig.output_fields["explanation"].annotation is str
        # sentiment should be Literal
        sentiment_type = sig.output_fields["sentiment"].annotation
        assert get_origin(sentiment_type) is Literal
        assert set(get_args(sentiment_type)) == {"positive", "negative", "neutral"}

    def test_field_descriptions(self):
        pred = Predict.from_yaml(YAML_DIR / "sentiment.yaml")
        sig = pred.config.signature
        assert sig.input_fields["title"].json_schema_extra["desc"] == "The title of the review"
        assert sig.output_fields["explanation"].json_schema_extra["desc"] == "Brief explanation of the sentiment classification"


class TestArbiter:
    def _make_predict(self, signature):
        """Create a Predict with a supported arbiter model."""
        model = f"provider/{next(iter(ARBITER_PROBES.keys()))}"
        return Predict(signature, lm=dspy.LM(model))

    def test_reasoning_field_added(self):
        """Arbiter should add a string reasoning field if not present."""
        pred = self._make_predict("question -> answer")
        arbiter = pred.as_arbiter()
        sig = arbiter.signature
        assert "reasoning" in sig.output_fields
        assert sig.output_fields["reasoning"].annotation is str


class TestPredictField:
    def test_resolve_type_string(self):
        f = PredictField(name="x", type="string")
        assert f.resolve_type() is str

    def test_resolve_type_str_alias(self):
        f = PredictField(name="x", type="str")
        assert f.resolve_type() is str

    def test_resolve_type_int(self):
        f = PredictField(name="x", type="int")
        assert f.resolve_type() is int

    def test_resolve_type_dict(self):
        f = PredictField(name="x", type="dict")
        assert f.resolve_type() is dict

    def test_resolve_type_with_options(self):
        f = PredictField(name="x", type="string", options=["a", "b"])
        resolved = f.resolve_type()
        assert get_origin(resolved) is Literal
        assert set(get_args(resolved)) == {"a", "b"}

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown type"):
            PredictField(name="x", type="tensor")


class TestPredictYamlSpec:
    def test_defaults(self):
        spec = PredictYamlSpec()
        assert spec.model is None
        assert spec.instructions is None
        assert spec.inputs == []
        assert spec.outputs == []

    def test_full_parse(self):
        spec = PredictYamlSpec(
            model="openai/gpt-4o",
            instructions="Do something",
            inputs=[{"name": "q", "type": "string"}],
            outputs=[{"name": "a", "type": "string", "options": ["yes", "no"]}],
        )
        assert spec.model == "openai/gpt-4o"
        assert len(spec.inputs) == 1
        assert spec.outputs[0].options == ["yes", "no"]
