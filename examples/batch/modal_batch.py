"""Example of running a batch request with ModalBatchClient."""

import asyncio

import dspy
from modaic.batch import ModalBatchClient, abatch
from modaic.batch.types import FailedPrediction


async def main():
    # Configure DSPy with a Hugging Face model path. Sampling params left unset
    # will be forwarded as None so the remote vLLM server uses its defaults.
    lm = dspy.LM(
        "huggingface/openai/gpt-oss-120b",
        max_tokens=256,
    )
    dspy.configure(
        lm=lm,
        adapter=dspy.ChatAdapter(),
    )

    predictor = dspy.Predict("question -> answer")
    client = ModalBatchClient(lm=lm, gpu="H200:2", reasoning_parser="openai_gptoss", enforce_eager=True)

    inputs = [
        {"question": "What is the capital of France? Answer briefly."},
        {"question": "What is 2 + 2? Answer briefly."},
        {"question": "Who wrote Romeo and Juliet? Answer briefly."},
    ]

    print("Submitting batch request to Modal...")  # noqa: T201
    results = await abatch(
        predictor,
        inputs,
        client=client,
        show_progress=False,
        return_messages=True,
    )

    for i, res in enumerate(results):
        pred = res.prediction
        if isinstance(pred, FailedPrediction):
            print(f"[{i}] FAILED: {pred.error}")  # noqa: T201
            continue

        print(f"[{i}] Q: {inputs[i]['question']}")  # noqa: T201
        print(f"    A: {pred.answer}")  # noqa: T201
        if hasattr(pred, "_outputs") and pred._outputs.get("reasoning_content"):
            print(f"    Reasoning: {pred._outputs['reasoning_content']}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
