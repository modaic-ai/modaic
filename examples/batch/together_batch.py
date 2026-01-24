"""Example of running a batch request with Together AI."""

import asyncio

import dspy

from modaic.batch.batch import abatch
from modaic.batch.types import FailedPrediction


async def main():
    # Configure dspy with Together AI and ChatAdapter
    dspy.configure(
        lm=dspy.LM("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        adapter=dspy.ChatAdapter(),
    )

    # Create a simple predictor
    predictor = dspy.Predict("question -> answer")

    # Example inputs
    inputs = [
        {"question": "What is the capital of France?"},
        {"question": "What is 2 + 2?"},
        {"question": "Who wrote Romeo and Juliet?"},
    ]

    # Run batch request
    print("Submitting batch request to Together AI...")
    predictions = await abatch(predictor, inputs)

    # Print results
    for i, pred in enumerate(predictions):
        if isinstance(pred, FailedPrediction):
            print(f"[{i}] FAILED: {pred.error}")
        else:
            print(f"[{i}] Q: {inputs[i]['question']}")
            print(f"    A: {pred.answer}")


if __name__ == "__main__":
    asyncio.run(main())
