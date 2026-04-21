"""Run VLLMBatchClient with vLLM's Python API inside a Modal container."""

from __future__ import annotations

import modal

APP_NAME = "modaic-vllm-batch-example"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_pip_install(
        "modal",
        "httpx>=0.27.0",
        "platformdirs>=4.3.8",
        "pydantic-settings>=2.13.0",
        "dspy>=3.1.0",
        "gitpython>=3.1.45",
        "safetensors>=0.7.0",
        "tomlkit>=0.13.3",
        "python-frontmatter>=1.1.0",
        "duckdb>=1.2.0",
        "cloudpickle>=3.0.0",
        "lmdb>=1.4.0",
        "pandas",
        "datasets",
        "vllm>=0.18.0",
        "huggingface-hub",
        "hf_transfer",
        "hf-xet>=1.1.7",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("modaic", "modaic_client")
)


@app.function(image=image, gpu="H100", timeout=60 * 60)
async def run_example() -> list[dict[str, object]]:
    import dspy
    from dspy.signatures.signature import make_signature
    from modaic.batch import VLLMBatchClient, abatch
    from modaic.batch.types import FailedPrediction

    lm = dspy.LM("huggingface/Qwen/Qwen3.5-4B", max_tokens=1024)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

    predictor = dspy.Predict("question -> answer")
    predictor2 = dspy.Predict(make_signature("context, question -> answer"))

    inputs = [
        {"question": "What is the capital of France? Answer briefly."},
        {"question": "What is 2 + 2? Answer briefly."},
        {"question": "Who wrote Romeo and Juliet? Answer briefly."},
    ]
    inputs2 = [
        {"context": "The capital of France is Paris.", "question": "What is the capital of France? Answer briefly."},
        {"context": "The answer is 4.", "question": "What is 2 + 2? Answer briefly."},
        {
            "context": "William Shakespeare wrote Romeo and Juliet.",
            "question": "Who wrote Romeo and Juliet? Answer briefly.",
        },
    ]

    grouped_inputs = [
        (predictor, inputs),
        (predictor2, inputs2),
    ]

    client = VLLMBatchClient(
        lm=lm,
        reasoning_parser="qwen3",
        enforce_eager=True,
        enable_thinking=True,
        thinking_budget=1024,
    )

    async with client.session():
        grouped_results = await abatch(
            grouped_inputs,
            client=client,
            show_progress=False,
            return_messages=True,
        )

    output_rows: list[dict[str, object]] = []
    for label, source_inputs, (_, result) in zip(
        ["question -> answer", "context, question -> answer"],
        [inputs, inputs2],
        grouped_results,
        strict=True,
    ):
        for row_index, row in enumerate(result):
            prediction = row.prediction
            if isinstance(prediction, FailedPrediction):
                output_rows.append(
                    {
                        "label": label,
                        "batch_id": result.batch_id,
                        "index": row_index,
                        "input": source_inputs[row_index],
                        "status": "failed",
                        "error": prediction.error,
                    }
                )
                continue

            answer = getattr(prediction, "answer", None)
            output_rows.append(
                {
                    "label": label,
                    "batch_id": result.batch_id,
                    "index": row_index,
                    "input": source_inputs[row_index],
                    "status": "ok",
                    "answer": "" if answer is None else str(answer),
                    "text": "" if row.outputs["text"] is None else str(row.outputs["text"]),
                    "reasoning": row.outputs["reasoning_content"],
                }
            )

    return output_rows


@app.local_entrypoint()
def main() -> None:
    rows = run_example.remote()
    for row in rows:
        print(row)  # noqa: T201
