from typing import Literal

import dspy

from modaic import PrecompiledAgent, PrecompiledConfig


class Summarize(dspy.Signature):
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField(desc="Answer to the question, based on the passage")


class ExampleConfig(PrecompiledConfig):
    output_type: Literal["bool", "str"]
    lm: str = "openai/gpt-4o-mini"
    number: int = 1


class ExampleAgent(PrecompiledAgent[ExampleConfig, None]):
    config_class = ExampleConfig

    def __init__(self, config: ExampleConfig, runtime_param: str, **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(Summarize)
        self.predictor.lm = dspy.LM("openai/gpt-4o-mini")
        self.runtime_param = runtime_param

    def forward(self, question: str, context: str) -> str:
        return self.predictor(question=question, context=context)


agent = ExampleAgent(ExampleConfig(), runtime_param="Hello")
agent.push_to_hub("hub_tests/simple_repo", with_code=False)
