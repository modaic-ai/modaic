from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

APP_NAME = "modaic-gpt-oss-batch-test"
MODEL_ID = "openai/gpt-oss-120b"
REASONING_PARSER = "openai_gptoss"
GPU_CONFIG = "H200:4"
INPUT_PATH = Path("/tmp/gpt_oss_batch_input.jsonl")
OUTPUT_PATH = Path("/tmp/gpt_oss_batch_output.jsonl")

REQUESTS = [
    {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 17 times 19? Think carefully, then give the final answer.",
                }
            ],
            "max_completion_tokens": 512,
            "temperature": 0.0,
            "include_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
]

app = modal.App(APP_NAME)

image = modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12").uv_pip_install(
    "vllm>=0.17.0",
    "flashinfer-python",
    "torch>=2.10,<2.11",
    "datasets",
    "huggingface-hub",
    "hf_transfer",
    "hf-xet>=1.1.7",
    "tqdm",
    "transformers>=4.50,<5",
    "python-dotenv",
    "pandas",
)


def _write_requests(path: Path) -> None:
    lines = [json.dumps(item) for item in REQUESTS]
    path.write_text("\n".join(lines) + "\n")


@app.function(gpu=GPU_CONFIG, image=image, timeout=60 * 60 * 2)
def run_batch_test() -> None:
    _write_requests(INPUT_PATH)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    cmd = [
        "vllm",
        "run-batch",
        "-i",
        str(INPUT_PATH),
        "-o",
        str(OUTPUT_PATH),
        "--model",
        MODEL_ID,
        "--reasoning-parser",
        REASONING_PARSER,
    ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    print("\n=== input.jsonl ===", flush=True)
    print(INPUT_PATH.read_text(), flush=True)
    print("\n=== output.jsonl ===", flush=True)
    print(OUTPUT_PATH.read_text(), flush=True)


@app.local_entrypoint()
def main() -> None:
    run_batch_test.remote()
