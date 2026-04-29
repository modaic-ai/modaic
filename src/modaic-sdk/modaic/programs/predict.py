import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import dspy
import yaml
from dspy import InputField, OutputField
from dspy.signatures import ensure_signature, make_signature

from ..hub import Commit
from ..precompiled import PrecompiledConfig, PrecompiledProgram
from ..safe_lm import SafeLM
from ..serializers import SerializableSignature
from .arbiters import make_arbiter
from .utils import PredictYamlSpec

if TYPE_CHECKING:
    from ..batch.clients import BatchClient
    from ..batch.types import ABatchResult
    from ..probe import ProbeModel


# Config takes in a signature and also an LM since sometimes dspy.configure does not set the lm that is serialized.
class PredictConfig(PrecompiledConfig):
    signature: SerializableSignature

    # CAVEAT: modaic.Predict and dspy.Predict both use self.config, but they mean completely different things.
    # modaic stores a PredictConfig (Pydantic model) while DSPy expects a plain dict of LM kwargs.
    # DSPy's _forward_preprocess does {**self.config, ...}, which requires the mapping protocol.
    # We implement keys() / __getitem__ here so that unpacking yields {} — no PredictConfig fields
    # (e.g. signature, model) should ever leak into DSPy's LM call config.
    def keys(self):
        # CAVEAT: intentionally returns empty — PredictConfig holds modaic metadata, not LM kwargs.
        return []

    def __getitem__(self, key):
        # CAVEAT: unreachable as long as keys() returns [] but required to satisfy the mapping protocol.
        raise KeyError(key)


ConfigType = PredictConfig | dict
SignatureType = dspy.Signature | str


class Predict(PrecompiledProgram, dspy.Predict):
    config: PredictConfig
    probe: Optional["ProbeModel"] = None

    def __init__(self, config: ConfigType | SignatureType, lm: Optional[dspy.LM] = None, **lm_kwargs):
        """
        Args:
            config: A config type (dict or PredictConfig) or a signature type (dspy.Signature or str)
            lm: Optional dspy.LM
            **kwargs: Additional keyword arguments to pass to dspy.Predict
        """
        if not isinstance(config, (dict, PredictConfig)):
            signature = ensure_signature(config)
            config = PredictConfig(signature=signature)
        super().__init__(config, signature=config.signature, **lm_kwargs)
        self.config = self.ensure_config(config)
        self.lm_kwargs = lm_kwargs
        if lm is not None:
            self.lm = lm

    def as_arbiter(self) -> "Predict":
        return make_arbiter(self)

    def push_to_hub(
        self,
        repo_path: str,
        access_token: str = None,
        commit_message: str = "(no commit message)",
        with_code: Optional[bool] = None,
        private: bool = True,
        branch: str = "main",
        tag: str = None,
        probe: Optional["ProbeModel"] = None,
        metadata: dict = None,
        extra_files: Optional[list[str | Path]] = None,
        clean: Optional[bool] = None,
    ) -> Commit:
        if with_code is not None:
            warnings.warn(
                "push_to_hub(with_code=...) is not supported for modaic.Predict, it will be ignored", stacklevel=2
            )
        self.probe = probe
        return super().push_to_hub(
            repo_path=repo_path,
            access_token=access_token,
            commit_message=commit_message,
            with_code=False,
            private=private,
            branch=branch,
            tag=tag,
            metadata=metadata,
            extra_files=extra_files,
            clean=clean,
        )

    def save_precompiled(
        self, path: str, _with_auto_classes: bool = False, extra_files: Optional[list[str | Path]] = None
    ) -> None:
        super().save_precompiled(path, _with_auto_classes, extra_files=extra_files)
        path = Path(path)
        # save probe model if it exists
        if self.probe is not None:
            self.probe.save(path)
        # otherwise copy it over from source dir.
        elif self._source is not None:
            pweights_path = self._source / "probe.safetensors"
            pconfig_path = self._source / "probe.json"
            if pweights_path.exists() and pconfig_path.exists():
                shutil.copy2(pweights_path, path / "probe.safetensors")
                shutil.copy2(pconfig_path, path / "probe.json")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Predict":
        path = Path(path)
        with open(path) as f:
            spec = PredictYamlSpec(**yaml.safe_load(f))

        fields = {}
        for field_def in spec.inputs:
            kwargs = {"description": field_def.description} if field_def.description else {}
            fields[field_def.name] = (field_def.resolve_type(), InputField(**kwargs))

        for field_def in spec.outputs:
            kwargs = {"description": field_def.description} if field_def.description else {}
            fields[field_def.name] = (field_def.resolve_type(), OutputField(**kwargs))

        signature = make_signature(fields, instructions=spec.instructions)
        if spec.lm:
            lm_kwargs = spec.lm.model_dump()
            model = lm_kwargs.pop("model")
            lm = dspy.LM(model, **lm_kwargs)
        elif spec.model:
            lm = dspy.LM(spec.model)
        else:
            lm = None
        return cls(signature, lm=lm)

    async def abatch(
        self,
        inputs: list[dict],
        show_progress: bool = True,
        poll_interval: float = 30,
        max_poll_time: str = "24h",
        return_messages: bool = False,
        client: Optional["BatchClient"] = None,
    ) -> "ABatchResult":
        from ..batch import abatch

        grouped_results = await abatch(
            [(self, inputs)],
            show_progress=show_progress,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            return_messages=return_messages,
            client=client,
        )
        return grouped_results[0][1]

    def __call__(self, **kwargs: dict[str, Any]) -> dspy.Prediction:
        from .arbiters import is_reasoning_model, register_reasoning_model

        if self.lm is not None and is_reasoning_model(self.lm.model):
            register_reasoning_model(self.lm.model)
            existing = self.lm.kwargs.get("allowed_openai_params", [])
            if "reasoning_effort" not in existing:
                self.lm.kwargs["allowed_openai_params"] = existing + ["reasoning_effort"]
        prediction = super().__call__(**kwargs)
        if kwargs.pop("return_messages", False):
            lm, _, _, _, _ = self._forward_preprocess(**kwargs)
            if not isinstance(lm, SafeLM):
                raise ValueError(
                    "return_messages is only supported with modaic.SafeLM. Please dspy.configure(lm=modaic.SafeLM(...)) or pass in a modaic.SafeLM instance as the lm argument."
                )
            if not lm.local_history:
                warnings.warn("No local history found for return_messages", UserWarning, stacklevel=2)
                prediction._messages = []
                prediction._outputs = {}
                return prediction

            history = lm.local_history[-1]
            prediction._messages = list(history.get("messages") or [])
            assistant_text = self._extract_assistant_text(history)
            reasoning_content = self._extract_reasoning_content(history)

            outputs = {"text": assistant_text}
            if reasoning_content is not None:
                outputs["reasoning_content"] = reasoning_content
            prediction._outputs = outputs
        return prediction

    def _extract_assistant_text(self, history: dict[str, Any]) -> str | None:
        outputs = history.get("outputs")
        text_from_outputs = self._extract_text_from_outputs(outputs)
        if text_from_outputs:
            return text_from_outputs

        response = history.get("response")
        return self._extract_text_from_response(response)

    def _extract_text_from_outputs(self, outputs: Any) -> str | None:
        if not isinstance(outputs, list) or len(outputs) == 0:
            return None

        first = outputs[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                return text
            if text is not None:
                return str(text)
        return None

    def _extract_text_from_response(self, response: Any) -> str | None:
        if response is None:
            return None

        try:
            choices = getattr(response, "choices", None)
            if choices and len(choices) > 0:
                choice = choices[0]
                message = getattr(choice, "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if isinstance(content, str):
                        return content
                    if content is not None:
                        return str(content)

                choice_text = getattr(choice, "text", None)
                if isinstance(choice_text, str):
                    return choice_text
                if choice_text is not None:
                    return str(choice_text)
        except Exception:
            return None

        return None

    def _extract_reasoning_content(self, history: dict[str, Any]) -> str | None:
        response = history.get("response")
        if response is None:
            return None
        try:
            choices = getattr(response, "choices", None)
            if choices and len(choices) > 0:
                message = getattr(choices[0], "message", None)
                if message is not None:
                    reasoning = getattr(message, "reasoning_content", None)
                    if isinstance(reasoning, str):
                        return reasoning
                    if reasoning is not None:
                        return str(reasoning)
        except Exception:
            return None
        return None

    def _forward_preprocess(self, **kwargs):
        # CAVEAT: modaic.Predict stores a PredictConfig in self.config, but dspy.Predict._forward_preprocess
        # does {**self.config, **kwargs.pop("config", {})} expecting self.config to be a dict of LM kwargs.
        # PredictConfig.keys() returns [] so {**self.config} safely yields {}, preventing any modaic metadata
        # from leaking into the LM call. We inject self.lm_kwargs via kwargs["config"] here — a per-call
        # local variable — so no shared mutable state is touched. This makes the method thread-safe under
        # dspy.Parallel, where multiple threads may call this concurrently on the same Predict instance.
        kwargs["config"] = {**self.lm_kwargs, **kwargs.pop("config", {})}
        return super()._forward_preprocess(**kwargs)

    def update_config(self, **kwargs):
        warnings.warn(
            "modaic.Predict does not store extra lm kwargs in self.config, use update_lm_kwargs instead",
            UserWarning,
            stacklevel=2,
        )
        self.lm_kwargs = {**self.lm_kwargs, **kwargs}

    def get_config(self) -> dict:
        warnings.warn(
            "modaic.Predict does not store extra lm kwargs in self.config, use get_lm_kwargs instead",
            UserWarning,
            stacklevel=2,
        )
        return self.lm_kwargs

    def get_lm_kwargs(self) -> dict:
        return self.lm_kwargs

    def update_lm_kwargs(self, **kwargs: Any) -> None:
        self.lm_kwargs = {**self.lm_kwargs, **kwargs}

    def load_state(self, state: dict) -> "Predict":
        """Load state, keeping existing LM only if state has no LM."""
        existing_lm = self.lm

        # DSPy bug workaround: For reasoning models, LM.__init__ stores
        # max_completion_tokens in self.kwargs, and dump_state() flattens it
        # into the state dict. On load, LM(**state) passes it via **kwargs,
        # but __init__ then does dict(max_completion_tokens=max_tokens, **kwargs)
        # which collides. Fix by mapping it back to the max_tokens param name.
        if state.get("lm") is not None:
            lm_state = state["lm"]
            if "max_completion_tokens" in lm_state:
                lm_state.setdefault("max_tokens", lm_state.pop("max_completion_tokens"))

        result = super().load_state(state)
        if state.get("lm") is None and existing_lm is not None:
            self.lm = existing_lm
        return result
