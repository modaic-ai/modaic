"""Stress-test the Fireworks batch client on the tytodd/honesty dataset (567k rows).

Requires env vars: FIREWORKS_AI_API_KEY, FIREWORKS_ACCOUNT_ID
"""

import asyncio
import time

import dspy
from datasets import load_dataset
from modaic.batch.batch import abatch
from modaic.batch.types import FailedPrediction


async def main():
    print("Loading tytodd/honesty train split...")  # noqa: T201
    ds = load_dataset("tytodd/honesty", split="train")
    inputs = [{"question": row["question"]} for row in ds]
    print(f"Loaded {len(inputs)} rows")  # noqa: T201

    dspy.configure(
        lm=dspy.LM("fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct"),
        adapter=dspy.ChatAdapter(),
    )

    predictor = dspy.Predict("question -> answer")

    print("Submitting batch request to Fireworks AI...")  # noqa: T201
    t0 = time.time()
    _, predictions = (await abatch([(predictor, inputs)], max_poll_time="48h"))[0]
    elapsed = time.time() - t0

    successes = 0
    failures = 0
    for res in predictions:
        if isinstance(res.prediction, FailedPrediction):
            failures += 1
        else:
            successes += 1

    print(f"\nCompleted in {elapsed:.1f}s")  # noqa: T201
    print(f"Successes: {successes}  Failures: {failures}")  # noqa: T201

    print("\nSample results (first 10):")  # noqa: T201
    for i, res in enumerate(predictions[:10]):
        pred = res.prediction
        if isinstance(pred, FailedPrediction):
            print(f"  [{i}] FAILED: {pred.error}")  # noqa: T201
        else:
            print(f"  [{i}] Q: {inputs[i]['question']}")  # noqa: T201
            print(f"       A: {pred.answer}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
