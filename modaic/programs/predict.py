import warnings
from typing import Optional

import dspy

from ..hub import Commit
from ..precompiled import PrecompiledConfig, PrecompiledProgram
from ..serializers import SerializableLM, SerializableSignature
from ..batch import abatch, FailedPrediction


# Config takes in a signature and also an LM since sometimes dspy.configure does not set the lm that is serialized.
class PredictConfig(PrecompiledConfig):
    signature: SerializableSignature


class Predict(PrecompiledProgram):
    config: PredictConfig

    def __init__(self, config: PredictConfig, lm: Optional[dspy.LM] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(config.signature)
        if lm is not None:
            self.predictor.set_lm(lm=lm)
            self.set_lm(lm=lm)

    def forward(self, **kwargs) -> dspy.Prediction:
        return self.predictor(**kwargs)

    def push_to_hub(
        self,
        repo_path: str,
        access_token: str = None,
        commit_message: str = "(no commit message)",
        with_code: Optional[bool] = None,
        private: bool = False,
        branch: str = "main",
        tag: str = None,
    ) -> Commit:
        if with_code is not None:
            warnings.warn(
                "push_to_hub(with_code=...) is not supported for modaic.Predict, it will be ignored", stacklevel=2
            )
        return super().push_to_hub(
            repo_path=repo_path,
            access_token=access_token,
            commit_message=commit_message,
            with_code=False,
            private=private,
            branch=branch,
            tag=tag,
        )

    async def abatch(self, inputs: list[dict], show_progress: bool = True, poll_interval: float = 30, max_poll_time: str = "24h") -> list[dspy.Prediction | FailedPrediction]:
        return await abatch(self.predictor, inputs, show_progress=show_progress, poll_interval=poll_interval, max_poll_time=max_poll_time)
