import dspy

from ..precompiled import PrecompiledConfig, PrecompiledProgram
from ..s_signature import SerializableSignature


class BasicJudgeConfig(PrecompiledConfig):
    signature: SerializableSignature


class BasicJudge(PrecompiledProgram):
    config: BasicJudgeConfig

    def __init__(self, config: BasicJudgeConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(config.signature)

    def forward(self, **kwargs) -> str:
        return self.predictor(**kwargs)
