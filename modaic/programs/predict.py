import warnings
from typing import Optional

import dspy
from dspy import Predict
from dspy.signatures import ensure_signature
from ..hub import Commit
from ..precompiled import PrecompiledConfig, PrecompiledProgram
from ..serializers import SerializableSignature
from ..batch import abatch, FailedPrediction


# Config takes in a signature and also an LM since sometimes dspy.configure does not set the lm that is serialized.
class PredictConfig(PrecompiledConfig):
    signature: SerializableSignature

ConfigType = PredictConfig | dict
SignatureType = dspy.Signature | str

class Predict(PrecompiledProgram, dspy.Predict):
    config: PredictConfig

    def __init__(self, config: ConfigType | SignatureType, lm: Optional[dspy.LM] = None, **lm_kwargs):
        """
        Args:
            config: A config type (dict or PredictConfig) or a signature type (dspy.Signature or str)
            lm: Optional dspy.LM
            **kwargs: Additional keyword arguments to pass to dspy.Predict
        """
        if not isinstance(config, (dict, PredictConfig)):
            print("ITS A SIGNATURE")
            signature = ensure_signature(config)
            config = PredictConfig(signature=signature)
        print("CONF")
        super().__init__(config, signature=config.signature, **lm_kwargs)
        self.config = self.ensure_config(config)
        self.lm_kwargs = lm_kwargs
        if lm is not None:
            self.set_lm(lm=lm)
    
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
        return await abatch(self, inputs, show_progress=show_progress, poll_interval=poll_interval, max_poll_time=max_poll_time)
    

    def _forward_preprocess(self, **kwargs):
        # safely call super._forward_preprocess so that it does not override config
        modaic_config = self.config
        self.config = self.lm_kwargs
        super()._forward_preprocess(**kwargs)
        self.lm_kwargs = self.config
        self.config = modaic_config
    
    def update_config(self, **kwargs):
        warnings.warn("modaic.Predict does not store extra lm kwargs in self.config, use update_lm_kwargs instead", UserWarning)
        self.lm_kwargs = {**self.lm_kwargs, **kwargs}

    def get_config(self):
        warnings.warn("modaic.Predict does not store extra lm kwargs in self.config, use get_lm_kwargs instead", UserWarning)
        return self.lm_kwargs
    
    def get_lm_kwargs(self):
        return self.lm_kwargs
    
    def update_lm_kwargs(self, **kwargs):
        self.lm_kwargs = {**self.lm_kwargs, **kwargs}


# if __name__ == "__main__":
#     class MySig(dspy.Signature):
#         input: str = dspy.InputField(description="Input string")
#         output: str = dspy.OutputField(description="Output string")
    
#     config = PredictConfig(signature=MySig)
#     predict = Predict(config, lm=dspy.LM("gpt-4o"))
#     predict(input="Hello world")