"""Define an Arbiter and push it to Modaic Hub.

Run:
    uv run python arbiter.py

Requirements:
    - MODAIC_TOKEN set in your environment (https://modaic.dev/settings/tokens)
    - The provider's API key (here, TOGETHER_API_KEY) set as an
      Environment Variable on Modaic Hub so future runs can execute.
"""

from typing import Literal

import dspy
import modaic


class CodeCompletionSignature(dspy.Signature):
    """Given a prompt and a code completion in Luau (Roblox's scripting
    language), evaluate the quality and relevance of the completion in
    relation to the prompt.

    Use a 1-4 scale:
      1 - irrelevant or broken
      2 - relevant but incorrect
      3 - mostly correct, minor issues
      4 - correct and idiomatic
    """

    prompt: str = dspy.InputField(desc="The prompt used to generate the completion")
    completion: str = dspy.InputField(desc="The code completion to evaluate")
    quality: Literal[1, 2, 3, 4] = dspy.OutputField(
        desc="Quality score from 1 (worst) to 4 (best)"
    )


# For very long instructions, prefer .with_instructions over the docstring:
#
#   long_rubric = open("rubric.txt").read()
#   CodeCompletionSignature = CodeCompletionSignature.with_instructions(long_rubric)


if __name__ == "__main__":
    arbiter = modaic.Predict(
        CodeCompletionSignature,
        lm=dspy.LM(model="together_ai/openai/gpt-oss-120b"),
    ).as_arbiter()  # MUST call before push_to_hub for it to be recognized as an arbiter

    arbiter.push_to_hub(
        "your-org/code-completions",
        private=True,
        commit_message="initial release",
        tag="v1",
    )
