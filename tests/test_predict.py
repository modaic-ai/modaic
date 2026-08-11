import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, get_args, get_origin

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from modaic import Predict, SafeLM
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
        assert (
            sig.output_fields["explanation"].json_schema_extra["desc"]
            == "Brief explanation of the sentiment classification"
        )


class TestArbiter:
    def _make_predict(self, signature):  # noqa
        """Create a Predict with a supported arbiter model."""
        model = f"provider/{next(iter(ARBITER_PROBES.keys()))}"
        return Predict(signature, lm=dspy.LM(model))

    def test_reasoning_field_added(self):
        """Arbiter should add a dspy.Reasoning field if not present."""
        pred = self._make_predict("question -> answer")
        arbiter = pred.as_arbiter()
        sig = arbiter.signature
        assert "reasoning" in sig.output_fields
        assert sig.output_fields["reasoning"].annotation is dspy.Reasoning

    @pytest.mark.parametrize(
        "model, expected_probe_model",
        [
            ("openai/gpt-5.5", "gpt-5.5"),
            ("anthropic/claude-opus-4-8", "claude-opus-4-8"),
        ],
    )
    def test_supported_arbiter_models_write_metadata(self, model: str, expected_probe_model: str):
        """Supported arbiter models write their probe metadata
        (model + size + supports_reasoning)."""
        arbiter = Predict("question -> answer", lm=dspy.LM(model)).as_arbiter()
        assert arbiter.metadata["is_arbiter"] is True
        assert arbiter.metadata["model"] == expected_probe_model
        assert arbiter.metadata["size"] == "medium"
        assert arbiter.metadata["supports_reasoning"] is True

    @pytest.mark.parametrize("model", ["openai/gpt-4o", "openai/gpt-3.5-turbo"])
    def test_unsupported_models_still_rejected(self, model: str):
        """Models that aren't supported arbiter models still raise."""
        with pytest.raises(ValueError, match="Arbiters are not supported"):
            Predict("question -> answer", lm=dspy.LM(model)).as_arbiter()


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


class TestLMStateRoundTrip:
    """dspy>=3.3.0 stamps the LM class into serialized state and refuses to import
    non-builtin classes on load. Arbiters pushed while modaic.SafeLM was required
    carry a modaic class path, so Predict.load_state normalizes it away."""

    SIG = "question -> answer"

    def _roundtrip(self, state: dict) -> dict:
        return json.loads(json.dumps(state))

    def test_plain_lm_records_builtin_class(self):
        pred = Predict(self.SIG)
        pred.lm = dspy.LM(model="openai/gpt-4o-mini")
        state = self._roundtrip(pred.dump_state())
        assert state["lm"]["_dspy_lm_class"] == "dspy.clients.lm.LM"

    def test_plain_lm_round_trips(self):
        pred = Predict(self.SIG)
        pred.lm = dspy.LM(model="openai/gpt-4o-mini")
        state = self._roundtrip(pred.dump_state())

        loaded = Predict(self.SIG)
        loaded.load_state(state)
        assert type(loaded.lm) is dspy.LM
        assert loaded.lm.model == "openai/gpt-4o-mini"

    def test_legacy_safelm_state_loads_as_builtin_lm(self):
        pred = Predict(self.SIG)
        pred.lm = SafeLM(model="openai/gpt-4o-mini")
        state = self._roundtrip(pred.dump_state())
        assert state["lm"]["_dspy_lm_class"] == "modaic.safe_lm.SafeLM"

        # stock dspy refuses this state outright
        with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
            dspy.Predict(self.SIG).load_state(self._roundtrip(state))

        loaded = Predict(self.SIG)
        loaded.load_state(self._roundtrip(state))
        assert type(loaded.lm) is dspy.LM
        assert loaded.lm.model == "openai/gpt-4o-mini"

    def test_legacy_marker_does_not_leak_into_lm_kwargs(self):
        pred = Predict(self.SIG)
        pred.lm = SafeLM(model="openai/gpt-4o-mini")
        loaded = Predict(self.SIG)
        loaded.load_state(self._roundtrip(pred.dump_state()))
        assert "_dspy_lm_class" not in loaded.lm.kwargs

    def test_state_without_lm_keeps_existing_lm(self):
        pred = Predict(self.SIG)
        existing = dspy.LM(model="openai/gpt-4o-mini")
        pred.lm = existing
        state = self._roundtrip(pred.dump_state())
        state["lm"] = None

        pred.load_state(state)
        assert pred.lm is existing


class TestReturnMessages:
    """Message capture binds a per-call lm.copy() instead of requiring modaic.SafeLM."""

    SIG = "question -> answer"

    def test_capture_with_plain_lm(self):
        pred = Predict(self.SIG)
        pred.lm = DummyLM([{"answer": "blue"}])

        out = pred(question="what colour is the sky?", return_messages=True)
        assert isinstance(out._messages, list) and out._messages
        assert "what colour is the sky?" in json.dumps(out._messages)
        assert "text" in out._outputs

    def test_shared_lm_history_untouched(self):
        pred = Predict(self.SIG)
        pred.lm = DummyLM([{"answer": "blue"}])

        pred(question="q", return_messages=True)
        assert pred.lm.history == []

    def test_concurrent_calls_do_not_cross_contaminate(self):
        n = 8
        pred = Predict(self.SIG)
        pred.lm = DummyLM([{"answer": f"a{i}"} for i in range(n * 4)])

        def one(i: int) -> tuple[int, str]:
            out = pred(question=f"q-{i}", return_messages=True)
            return i, json.dumps(out._messages)

        with ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(one, range(n)))

        for i, blob in results:
            assert f"q-{i}" in blob, f"call {i} lost its own messages"
            for j in range(n):
                if j != i:
                    assert f"q-{j}" not in blob, f"call {i} saw call {j}'s messages"

    def test_no_lm_configured_raises(self):
        pred = Predict(self.SIG)
        pred.lm = None
        with dspy.context(lm=None), pytest.raises(ValueError, match="return_messages requires a configured LM"):
            pred(question="q", return_messages=True)
