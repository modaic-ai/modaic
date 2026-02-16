"""Example of running a batch request with OpenAI."""

import asyncio
from typing import Optional

import dspy

from modaic.batch.batch import abatch
from modaic.batch.types import FailedPrediction


def print_status(batch_id: str, status: str, progress: Optional[int], metadata: dict):
    print(f"Batch {batch_id}: Status: {status}, Progress: {progress}%, Metadata: {metadata}")


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
        {"question": "What is the largest planet?"},
        {"question": "Who painted the Mona Lisa?"},
        {"question": "What is the speed of light?"},
        {"question": "What is the capital of Japan?"},
        {"question": "What is the capital of China?"},
        {"question": "What is the capital of India?"},
        {"question": "What is the capital of Brazil?"},
        {"question": "What is the capital of Russia?"},
        {"question": "What is the capital of Germany?"},
    ]

    # Run batch request
    print("Submitting batch request to OpenAI...")
    results = await abatch(predictor, inputs)

    # Print results
    for i, res in enumerate(results):
        pred = res.prediction
        if isinstance(pred, FailedPrediction):
            print(f"[{i}] FAILED: {pred.error}")
        else:
            print(f"[{i}] Q: {inputs[i]['question']}")
            print(f"    A: {pred.answer}")
            # print(f"    Messages: {res['messages']}")


if __name__ == "__main__":
    asyncio.run(main())
