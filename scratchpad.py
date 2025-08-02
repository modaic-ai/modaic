import dspy


class Summarize(dspy.Signature):
    """Summarize a passage in 1-2 sentences."""

    passage = dspy.InputField()
    summary = dspy.OutputField()


# Use a basic Predictor Module
summarizer = dspy.Predict(Summarize)

from dspy.datasets import HotPotQA  # or your own dataset
from dspy.teleprompt import BootstrapFewShot

# Load a dataset
trainset = HotPotQA().train[:10]

# Choose a Compiler (e.g., BootstrapFewShot)
compiler = BootstrapFewShot(metric=dspy.evaluate.answer_exact_match)

# Compile (train) the module on the dataset
compiled_summarizer = compiler.compile(summarizer, trainset=trainset)

compiled_summarizer.save("summarizer_module.json")
