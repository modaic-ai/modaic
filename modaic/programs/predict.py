import warnings
from typing import Optional

import dspy

from ..hub import Commit
from ..precompiled import PrecompiledConfig, PrecompiledProgram
from ..s_signature import SerializableSignature


class PredictConfig(PrecompiledConfig):
    signature: SerializableSignature


class Predict(PrecompiledProgram):
    config: PredictConfig

    def __init__(self, config: PredictConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(config.signature)

    def forward(self, **kwargs) -> str:
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
