"""Example of running a batch request with OpenAI."""

import asyncio

import dspy

from modaic.batch.batch import abatch
from modaic.batch.types import FailedPrediction


async def main():
    # Configure dspy with OpenAI and ChatAdapter
    dspy.configure(
        lm=dspy.LM("openai/gpt-4o-mini"),
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
    print("Submitting batch request to OpenAI...")
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
