import os
import sys
from typing import Literal

import dspy

from modaic import PrecompiledAgent, PrecompiledConfig
from modaic.hub import get_user_info


class Summarize(dspy.Signature):
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField(desc="Answer to the question, based on the passage")


class ExampleConfig(PrecompiledConfig):
    output_type: Literal["bool", "str"]
    lm: str = "openai/gpt-4o"
    number: int = 1


class ExampleAgent(PrecompiledAgent[ExampleConfig, None]):
    config_class = ExampleConfig

    def __init__(self, config: ExampleConfig, runtime_param: str, **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(Summarize)
        self.predictor.lm = dspy.LM(self.config.lm)
        self.runtime_param = runtime_param

    def forward(self, question: str, context: str) -> str:
        return self.predictor(question=question, context=context)


if __name__ == "__main__":
    username = sys.argv[1]  # ← first arg after script name
    agent = ExampleAgent(ExampleConfig(output_type="str"), runtime_param="hi")
    repo_path = f"{username}/simple_repo"
    agent.push_to_hub(repo_path, with_code=True)
