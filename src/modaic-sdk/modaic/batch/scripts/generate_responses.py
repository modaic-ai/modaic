# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "flashinfer-python",
#     "huggingface-hub",
#     "hf_transfer",
#     "hf-xet>= 1.1.7",
#     "torch",
#     "transformers",
#     "vllm>=0.17.0",
# ]
#
# ///
"""
Generate responses for prompts in a dataset using a local vLLM OpenAI-compatible server.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from datasets import load_dataset
from huggingface_hub import DatasetCard, get_token, login
from torch import cuda
from tqdm.auto import tqdm

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UV_SCRIPT_REPO_ID = "modaic/batch-vllm"
UV_SCRIPT_FILENAME = "generate_responses.py"
UV_SCRIPT_URL = f"https://huggingface.co/datasets/{UV_SCRIPT_REPO_ID}/resolve/main/{UV_SCRIPT_FILENAME}"
SERVER_PORT = 8000
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
MAX_IN_FLIGHT_REQUESTS = 256


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    max_tokens: int
    repetition_penalty: float


def check_gpu_availability() -> int:
    if not cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with NVIDIA GPU or use HF Jobs with GPU flavor.")
        sys.exit(1)

    num_gpus = cuda.device_count()
    for i in range(num_gpus):
        gpu_name = cuda.get_device_name(i)
        gpu_memory = cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name} with {gpu_memory:.1f} GB memory")

    return num_gpus


def _wait_for_server(timeout_seconds: int = 1800) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("Timed out waiting for the vLLM server to become healthy")


def _post_chat_completion(request_body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _build_request_body(
    model_id: str,
    messages: list[dict[str, Any]],
    generation: GenerationConfig,
    enable_thinking: bool,
) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": generation.max_tokens,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "min_p": generation.min_p,
        "repetition_penalty": generation.repetition_penalty,
        "include_reasoning": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }


def _extract_output_payload(response_body: dict[str, Any]) -> dict[str, Optional[str]]:
    choices = response_body.get("choices") or []
    if not choices:
        return {"text": "", "reasoning_content": None}

    message = choices[0].get("message") or {}
    return {
        "text": message.get("content") or "",
        "reasoning_content": message.get("reasoning"),
    }


def _run_generation_via_server(
    conversations: list[list[dict[str, Any]]],
    *,
    model_id: str,
    generation: GenerationConfig,
    enable_thinking: bool,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    reasoning_parser: Optional[str],
) -> list[dict[str, Optional[str]]]:
    env = os.environ.copy()
    server_cmd = [
        "vllm",
        "serve",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if max_model_len is not None:
        server_cmd.extend(["--max-model-len", str(max_model_len)])
    if reasoning_parser:
        server_cmd.extend(["--reasoning-parser", reasoning_parser])

    logger.info("Starting vLLM server: %s", " ".join(server_cmd))
    server = subprocess.Popen(server_cmd, env=env)

    try:
        _wait_for_server()
        logger.info("vLLM server is healthy; keeping up to %d requests in flight", MAX_IN_FLIGHT_REQUESTS)

        responses: list[dict[str, Optional[str]]] = [{"text": "", "reasoning_content": None} for _ in conversations]
        submitted = 0
        completed = 0
        futures: dict[Future[dict[str, Any]], int] = {}

        with ThreadPoolExecutor(max_workers=min(MAX_IN_FLIGHT_REQUESTS, len(conversations))) as executor:
            with tqdm(total=len(conversations), desc="Generating responses") as progress:
                while submitted < len(conversations) or futures:
                    while submitted < len(conversations) and len(futures) < MAX_IN_FLIGHT_REQUESTS:
                        request_body = _build_request_body(
                            model_id=model_id,
                            messages=conversations[submitted],
                            generation=generation,
                            enable_thinking=enable_thinking,
                        )
                        future = executor.submit(_post_chat_completion, request_body)
                        futures[future] = submitted
                        submitted += 1

                    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        index = futures.pop(future)
                        responses[index] = _extract_output_payload(future.result())
                        completed += 1
                        progress.update(1)

        return responses
    finally:
        logger.info("Stopping vLLM server")
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=30)


def create_dataset_card(
    source_dataset: str,
    model_id: str,
    messages_column: str,
    prompt_column: Optional[str],
    generation: GenerationConfig,
    tensor_parallel_size: int,
    num_examples: int,
    generation_time: str,
    num_skipped: int = 0,
    max_model_len_used: Optional[int] = None,
) -> str:
    filtering_section = ""
    input_column_flag = f"--prompt-column {prompt_column}" if prompt_column else f"--messages-column {messages_column}"
    max_model_len_flag = f" \\\n    --max-model-len {max_model_len_used}" if max_model_len_used else ""

    if num_skipped > 0:
        skip_percentage = (num_skipped / num_examples) * 100
        processed = num_examples - num_skipped
        filtering_section = f"""

### Filtering Statistics

- **Total Examples**: {num_examples:,}
- **Processed**: {processed:,} ({100 - skip_percentage:.1f}%)
- **Skipped (too long)**: {num_skipped:,} ({skip_percentage:.1f}%)
- **Max Model Length Used**: {max_model_len_used:,} tokens

Note: Prompts exceeding the maximum model length were skipped and have empty responses."""

    return f"""---
tags:
- generated
- vllm
- uv-script
---

# Generated Responses Dataset

This dataset contains generated responses for prompts from [{source_dataset}](https://huggingface.co/datasets/{source_dataset}).

## Generation Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Input Column**: `{prompt_column if prompt_column else messages_column}` ({"plain text prompts" if prompt_column else "chat messages"})
- **Model**: [{model_id}](https://huggingface.co/{model_id})
- **Number of Examples**: {num_examples:,}
- **Generation Date**: {generation_time}{filtering_section}

### Sampling Parameters

- **Temperature**: {generation.temperature}
- **Top P**: {generation.top_p}
- **Top K**: {generation.top_k}
- **Min P**: {generation.min_p}
- **Max Tokens**: {generation.max_tokens}
- **Repetition Penalty**: {generation.repetition_penalty}

### Hardware Configuration

- **Tensor Parallel Size**: {tensor_parallel_size}
- **GPU Configuration**: {tensor_parallel_size} GPU(s)

## Dataset Structure

The dataset contains all columns from the source dataset plus:
- `outputs`: The generated outputs dict from the model

## Generation Script

Generated using the vLLM inference script from [{UV_SCRIPT_REPO_ID}](https://huggingface.co/datasets/{UV_SCRIPT_REPO_ID}).

To reproduce this generation:

```bash
uv run {UV_SCRIPT_URL} \\
    {source_dataset} \\
    <output-dataset> \\
    --model-id {model_id} \\
    {input_column_flag} \\
    --temperature {generation.temperature} \\
    --top-p {generation.top_p} \\
    --top-k {generation.top_k} \\
    --max-tokens {generation.max_tokens}{max_model_len_flag}
```
"""


def main(
    src_dataset_hub_id: str,
    output_dataset_hub_id: str,
    model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    reasoning_parser: Optional[str] = None,
    messages_column: str = "messages",
    prompt_column: Optional[str] = None,
    output_column: str = "outputs",
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    max_tokens: int = 16384,
    repetition_penalty: float = 1.0,
    gpu_memory_utilization: float = 0.90,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    skip_long_prompts: bool = True,
    enable_thinking: bool = False,
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
):
    generation_start_time = datetime.now().isoformat()

    num_gpus = check_gpu_availability()
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus
        logger.info(f"Auto-detected {num_gpus} GPU(s), using tensor_parallel_size={tensor_parallel_size}")
    else:
        logger.info(f"Using specified tensor_parallel_size={tensor_parallel_size}")
        if tensor_parallel_size > num_gpus:
            logger.warning(f"Requested {tensor_parallel_size} GPUs but only {num_gpus} available")

    hf_access_token = hf_token or os.environ.get("HF_TOKEN") or get_token()
    if not hf_access_token:
        logger.error("No HuggingFace token found. Please provide token via:")
        logger.error("  1. --hf-token argument")
        logger.error("  2. HF_TOKEN environment variable")
        logger.error("  3. Run 'huggingface-cli login' or use login() in Python")
        sys.exit(1)

    logger.info("HuggingFace token found, authenticating...")
    login(token=hf_access_token)

    generation = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )

    logger.info(f"Loading dataset: {src_dataset_hub_id}")
    dataset = load_dataset(src_dataset_hub_id, split="train")

    if max_samples is not None and max_samples < len(dataset):
        logger.info(f"Limiting dataset to {max_samples} samples")
        dataset = dataset.select(range(max_samples))

    total_examples = len(dataset)
    logger.info(f"Dataset loaded with {total_examples:,} examples")

    if prompt_column:
        if prompt_column not in dataset.column_names:
            logger.error(f"Column '{prompt_column}' not found. Available columns: {dataset.column_names}")
            sys.exit(1)
        logger.info(f"Using prompt column mode with column: '{prompt_column}'")
        use_messages = False
    else:
        if messages_column not in dataset.column_names:
            logger.error(f"Column '{messages_column}' not found. Available columns: {dataset.column_names}")
            sys.exit(1)
        logger.info(f"Using messages column mode with column: '{messages_column}'")
        use_messages = True

    if skip_long_prompts:
        logger.info("Prompt length pre-filtering remains disabled; server-side limits will apply.")

    logger.info("Preparing chat messages...")
    conversations: list[list[dict[str, Any]]] = []
    for example in tqdm(dataset, desc="Processing prompts"):
        if use_messages:
            conversations.append(example[messages_column])
        else:
            conversations.append([{"role": "user", "content": example[prompt_column]}])

    if not conversations:
        logger.error("No prompts to process!")
        sys.exit(1)

    responses = _run_generation_via_server(
        conversations,
        model_id=model_id,
        generation=generation,
        enable_thinking=enable_thinking,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        reasoning_parser=reasoning_parser,
    )

    logger.info("Adding responses to dataset...")
    dataset = dataset.add_column(output_column, responses)

    logger.info("Creating dataset card...")
    card_content = create_dataset_card(
        source_dataset=src_dataset_hub_id,
        model_id=model_id,
        messages_column=messages_column,
        prompt_column=prompt_column,
        generation=generation,
        tensor_parallel_size=tensor_parallel_size,
        num_examples=total_examples,
        generation_time=generation_start_time,
        num_skipped=0,
        max_model_len_used=max_model_len,
    )

    logger.info(f"Pushing dataset to: {output_dataset_hub_id}")
    dataset.push_to_hub(output_dataset_hub_id, token=hf_access_token)

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset_hub_id, token=hf_access_token)

    logger.info("Generation complete")
    logger.info(f"Dataset available at: https://huggingface.co/datasets/{output_dataset_hub_id}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Generate responses for dataset prompts using a local vLLM server",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument("src_dataset_hub_id", help="Input dataset on Hugging Face Hub (e.g., username/dataset-name)")
        parser.add_argument("output_dataset_hub_id", help="Output dataset name on Hugging Face Hub")
        parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
        parser.add_argument("--messages-column", type=str, default="messages")
        parser.add_argument("--reasoning-parser", type=str, help="vLLM reasoning parser to use for supported models")
        parser.add_argument("--prompt-column", type=str, help="Column containing plain text prompts")
        parser.add_argument("--output-column", type=str, default="outputs")
        parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--top-p", type=float, default=0.8)
        parser.add_argument("--top-k", type=int, default=20)
        parser.add_argument("--min-p", type=float, default=0.0)
        parser.add_argument("--max-tokens", type=int, default=16384)
        parser.add_argument("--repetition-penalty", type=float, default=1.0)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
        parser.add_argument("--max-model-len", type=int)
        parser.add_argument("--tensor-parallel-size", type=int)
        parser.add_argument("--enable-thinking", action="store_true", default=False)
        parser.add_argument("--hf-token", type=str)
        parser.add_argument("--skip-long-prompts", action="store_true", default=True)
        parser.add_argument("--no-skip-long-prompts", dest="skip_long_prompts", action="store_false")

        args = parser.parse_args()

        main(
            src_dataset_hub_id=args.src_dataset_hub_id,
            output_dataset_hub_id=args.output_dataset_hub_id,
            model_id=args.model_id,
            reasoning_parser=args.reasoning_parser,
            messages_column=args.messages_column,
            prompt_column=args.prompt_column,
            output_column=args.output_column,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            skip_long_prompts=args.skip_long_prompts,
            enable_thinking=args.enable_thinking,
            max_samples=args.max_samples,
            hf_token=args.hf_token,
        )
    else:
        print(f"""
vLLM Response Generation Script
==============================

This script requires arguments. For usage information:
    uv run generate_responses.py --help

Upload this script to the Hub:
    hf upload --repo-type dataset \\
        modaic/batch-vllm \\
        src/modaic-sdk/modaic/batch/generate_responses.py \\
        generate_responses.py

Canonical script URL:
    {UV_SCRIPT_URL}
        """)
