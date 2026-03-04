import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import dspy
from dspy.signatures import ensure_signature

from ..batch import FailedPrediction, abatch
from ..hub import Commit
from ..precompiled import PrecompiledConfig, PrecompiledProgram
from ..safe_lm import SafeLM
from ..serializers import SerializableSignature

if TYPE_CHECKING:
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

    def push_to_hub(
        self,
        repo_path: str,
        access_token: str = None,
        commit_message: str = "(no commit message)",
        with_code: Optional[bool] = None,
        private: bool = False,
        branch: str = "main",
        tag: str = None,
        probe: Optional["ProbeModel"] = None,
        make_arbiter: bool = False,
        metadata: dict = None,
    ) -> Commit:
        if with_code is not None:
            warnings.warn(
                "push_to_hub(with_code=...) is not supported for modaic.Predict, it will be ignored", stacklevel=2
            )
        self.probe = probe
        if make_arbiter:
            if metadata is None:
                metadata = {}
            metadata["is_arbiter"] = True
        return super().push_to_hub(
            repo_path=repo_path,
            access_token=access_token,
            commit_message=commit_message,
            with_code=False,
            private=private,
            branch=branch,
            tag=tag,
            metadata=metadata,
        )

    def save_precompiled(self, path: str, _with_auto_classes: bool = False) -> None:
        super().save_precompiled(path, _with_auto_classes)
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

    async def abatch(
        self, inputs: list[dict], show_progress: bool = True, poll_interval: float = 30, max_poll_time: str = "24h"
    ) -> list[dspy.Prediction | FailedPrediction]:
        return await abatch(
            self, inputs, show_progress=show_progress, poll_interval=poll_interval, max_poll_time=max_poll_time
        )

    def __call__(self, **kwargs: dict[str, Any]) -> dspy.Prediction:
        prediction = super().__call__(**kwargs)
        if kwargs.pop("return_messages", False):
            lm, _, _, _, _ = self._forward_preprocess(**kwargs)
            if not isinstance(lm, SafeLM):
                raise ValueError(
                    "return_messages is only supported with SafeLM. Please dspy.configure(lm=SafeLM(...)) or pass in a SafeLM instance as the lm argument."
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
        result = super().load_state(state)
        if state.get("lm") is None and existing_lm is not None:
            self.lm = existing_lm
        return result
