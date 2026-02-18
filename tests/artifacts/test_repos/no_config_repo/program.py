import sys

import dspy
from modaic import PrecompiledProgram


class Summarize(dspy.Signature):
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField(desc="Answer to the question, based on the passage")


class ExampleProgram(PrecompiledProgram):
    def __init__(self, runtime_param: str, **kwargs):
        super().__init__(**kwargs)
        self.predictor = dspy.Predict(Summarize)
        self.predictor.lm = dspy.LM(self.config.lm)
        self.runtime_param = runtime_param

    def forward(self, question: str, context: str) -> str:
        return self.predictor(question=question, context=context)


if __name__ == "__main__":
    username = sys.argv[1]  # ← first arg after script name (username)
    program = ExampleProgram(runtime_param="hi")
    repo_path = f"{username}/no_config_repo"
    program.push_to_hub(repo_path, with_code=True)
