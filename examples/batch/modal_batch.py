"""Example of running a batch request with ModalBatchClient."""

import asyncio

import dspy
from modaic.batch import ModalBatchClient, abatch
from modaic.batch.types import ABatchResult
from modaic.batch.types import FailedPrediction


def print_group_results(label: str, inputs: list[dict[str, str]], results: ABatchResult) -> None:
    print(f"{label} batch_id: {results.batch_id}")  # noqa: T201
    for i, row in enumerate(results):
        pred = row.prediction
        if isinstance(pred, FailedPrediction):
            print(f"[{label}][{i}] FAILED: {pred.error}")  # noqa: T201
            continue

        print(f"[{label}][{i}] Input: {inputs[i]}")  # noqa: T201
        print(f"    A: {pred.answer}")  # noqa: T201
        if hasattr(pred, "_outputs") and pred._outputs.get("reasoning_content"):
            print(f"    Reasoning: {pred._outputs['reasoning_content']}")  # noqa: T201


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

    predictor2 = dspy.Predict("context, question -> answer")
    inputs2 = [
        {"context": "The capital of France is Paris.", "question": "What is the capital of France? Answer briefly."},
        {"context": "The answer is 4.", "question": "What is 2 + 2? Answer briefly."},
        {
            "context": "William Shakespeare wrote Romeo and Juliet.",
            "question": "Who wrote Romeo and Juliet? Answer briefly.",
        },
    ]

    grouped_inputs = [(predictor, inputs), (predictor2, inputs2)]

    print("Submitting one grouped batch request to Modal...")  # noqa: T201
    grouped_results = await abatch(
        grouped_inputs,
        client=client,
        show_progress=False,
        return_messages=True,
    )

    _, first_results = grouped_results[0]
    _, second_results = grouped_results[1]

    print_group_results("question -> answer", inputs, first_results)
    print_group_results("context, question -> answer", inputs2, second_results)


if __name__ == "__main__":
    asyncio.run(main())
