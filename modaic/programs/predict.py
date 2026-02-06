import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import dspy
from dspy.signatures import ensure_signature

from ..batch import FailedPrediction, abatch
from ..hub import Commit
from ..precompiled import PrecompiledConfig, PrecompiledProgram
from ..serializers import SerializableSignature

if TYPE_CHECKING:
    from ..probe import ProbeModel


# Config takes in a signature and also an LM since sometimes dspy.configure does not set the lm that is serialized.
class PredictConfig(PrecompiledConfig):
    signature: SerializableSignature


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

    def _forward_preprocess(self, **kwargs):
        # safely call super._forward_preprocess so that it does not override config
        modaic_config = self.config
        self.config = self.lm_kwargs
        results = super()._forward_preprocess(**kwargs)
        self.lm_kwargs = self.config
        self.config = modaic_config
        return results

    def update_config(self, **kwargs):
        warnings.warn(
            "modaic.Predict does not store extra lm kwargs in self.config, use update_lm_kwargs instead", UserWarning
        )
        self.lm_kwargs = {**self.lm_kwargs, **kwargs}

    def get_config(self):
        warnings.warn(
            "modaic.Predict does not store extra lm kwargs in self.config, use get_lm_kwargs instead", UserWarning
        )
        return self.lm_kwargs

    def get_lm_kwargs(self):
        return self.lm_kwargs

    def update_lm_kwargs(self, **kwargs):
        self.lm_kwargs = {**self.lm_kwargs, **kwargs}

    def load_state(self, state: dict) -> "Predict":
        """Load state, keeping existing LM only if state has no LM."""
        existing_lm = self.lm
        result = super().load_state(state)
        if state.get("lm") is None and existing_lm is not None:
            self.lm = existing_lm
        return result
