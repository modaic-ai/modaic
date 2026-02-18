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
        {"question": "How many planets are in the solar system?"},
        {"question": "What is the speed of light?"},
        {"question": "What is the capital of Japan?"},
        {"question": "What is the capital of China?"},
        {"question": "What is the capital of India?"},
        {"question": "What is the capital of Brazil?"},
        {"question": "What is the capital of Russia?"},
    ]

    # Run batch request
    print("Submitting batch request to Together AI...")  # noqa: T201
    result = await abatch(predictor, inputs)

    # Print results
    for i, res in enumerate(result):
        pred = res.prediction
        if isinstance(pred, FailedPrediction):
            print(f"[{i}] FAILED: {pred.error}")  # noqa: T201
        else:
            print(f"[{i}] Q: {inputs[i]['question']}")  # noqa: T201
            print(f"    A: {pred.answer}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
