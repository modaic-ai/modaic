"""Example of running a batch request with Fireworks AI."""

import asyncio

import dspy
from modaic.batch.batch import abatch
from modaic.batch.types import FailedPrediction


async def main():
    # Configure dspy with Fireworks AI and ChatAdapter
    dspy.configure(
        lm=dspy.LM("fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct"),
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
    print("Submitting batch request to Fireworks AI...")  # noqa: T201
    predictions = await abatch(predictor, inputs)

    # Print results
    for i, pred in enumerate(predictions):
        if isinstance(pred, FailedPrediction):
            print(f"[{i}] FAILED: {pred.error}")  # noqa: T201
        else:
            print(f"[{i}] Q: {inputs[i]['question']}")  # noqa: T201
            print(f"    A: {pred.answer}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
