import dspy

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
