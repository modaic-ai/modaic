import dspy
from modaic.precompiled_agent import PrecompiledAgent, PrecompiledConfig
from typing import Literal
import pytest
import os
import json


class Summarize(dspy.Signature):
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField(desc="Answer to the question, based on the passage")


class ExampleConfig(PrecompiledConfig):
    agent_type = "ExampleAgent"
    output_type: Literal["bool", "str"]

    def __init__(self, output_type: Literal["bool", "str"]):
        self.output_type = output_type


class ExampleAgent(PrecompiledAgent):
    config_class = ExampleConfig

    def __init__(self, config: ExampleConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(Summarize)
        self.predictor.lm = dspy.LM("openai/gpt-4o-mini")

    def forward(self, question: str, context: str) -> str:
        return self.predictor(question=question, context=context)


@pytest.fixture
def example_agent():
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    config = ExampleConfig(output_type="str")
    agent = ExampleAgent(config)
    return agent


@pytest.fixture
def example_config():
    config = ExampleConfig(output_type="str")
    return config


def test_config_save_load_precompiled():
    example_config = ExampleConfig(output_type="str")
    example_config.save_precompiled("test_module")
    assert os.path.exists("test_module")

    loaded_config = ExampleConfig.from_precompiled("test_module")
    assert loaded_config.agent_type == "ExampleAgent"
    assert loaded_config.output_type == example_config.output_type


def test_agent_save_load_precompiled(example_agent):
    example_config = ExampleConfig(output_type="bool")
    example_agent = ExampleAgent(example_config)
    example_agent.save_precompiled("test_module_2")
    assert os.path.exists("test_module_2")

    loaded_agent = ExampleAgent.from_precompiled("test_module_2")
    assert loaded_agent.config.agent_type == "ExampleAgent"
    assert loaded_agent.config.output_type == example_config.output_type


def test_trace_callback(example_agent):
    example_agent("What is the capital of France?", "France is a country in Europe.")
    assert os.path.exists("traces.jsonl")
    with open("traces.jsonl", "r") as f:
        traces = [json.loads(line) for line in f]
    assert len(traces) == 1
    assert traces[0]["event"] == "module_start"
    assert traces[0]["module"] == "ExampleAgent"
    assert traces[0]["inputs"] == {"input": "Hello, world!"}
