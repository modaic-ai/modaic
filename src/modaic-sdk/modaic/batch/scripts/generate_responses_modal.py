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
"""Generate responses for parquet prompts using a local vLLM OpenAI-compatible server."""

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
from typing import Any, Optional

from datasets import load_dataset
from huggingface_hub import get_token
from torch import cuda
from tqdm.auto import tqdm

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_PORT = 8000
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
MAX_IN_FLIGHT_REQUESTS = 256


@dataclass(frozen=True)
class GenerationConfig:
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    min_p: Optional[float]
    max_tokens: Optional[int]
    repetition_penalty: Optional[float]


def check_gpu_availability() -> int:
    if not cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
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
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "include_reasoning": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if generation.max_tokens is not None:
        body["max_completion_tokens"] = generation.max_tokens
    if generation.temperature is not None:
        body["temperature"] = generation.temperature
    if generation.top_p is not None:
        body["top_p"] = generation.top_p
    if generation.top_k is not None:
        body["top_k"] = generation.top_k
    if generation.min_p is not None:
        body["min_p"] = generation.min_p
    if generation.repetition_penalty is not None:
        body["repetition_penalty"] = generation.repetition_penalty
    return body


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
    enforce_eager: bool,
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
    if enforce_eager:
        server_cmd.append("--enforce-eager")

    logger.info("Starting vLLM server: %s", " ".join(server_cmd))
    server = subprocess.Popen(server_cmd, env=env)

    try:
        _wait_for_server()
        logger.info("vLLM server is healthy; keeping up to %d requests in flight", MAX_IN_FLIGHT_REQUESTS)

        responses: list[dict[str, Optional[str]]] = [{"text": "", "reasoning_content": None} for _ in conversations]
        submitted = 0
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


def main(
    src_dataset_path: str,
    output_dataset_path: str,
    model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    reasoning_parser: Optional[str] = None,
    enforce_eager: bool = False,
    messages_column: str = "messages",
    prompt_column: Optional[str] = None,
    output_column: str = "outputs",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    gpu_memory_utilization: float = 0.90,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    skip_long_prompts: bool = True,
    enable_thinking: bool = False,
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> None:
    num_gpus = check_gpu_availability()
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus
        logger.info(f"Auto-detected {num_gpus} GPU(s), using tensor_parallel_size={tensor_parallel_size}")
    elif tensor_parallel_size > num_gpus:
        logger.warning(f"Requested {tensor_parallel_size} GPUs but only {num_gpus} available")

    resolved_hf_token = hf_token or os.environ.get("HF_TOKEN") or get_token()
    if resolved_hf_token:
        os.environ["HF_TOKEN"] = resolved_hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = resolved_hf_token
        logger.info("Using Hugging Face token for model access")

    generation = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )

    logger.info(f"Loading parquet dataset: {src_dataset_path}")
    dataset = load_dataset("parquet", data_files=src_dataset_path, split="train")

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

    conversations: list[list[dict[str, Any]]] = []
    for example in tqdm(dataset, desc="Processing prompts"):
        if use_messages:
            conversations.append(example[messages_column])
        else:
            conversations.append([{"role": "user", "content": example[prompt_column]}])

    if not conversations:
        logger.error("No prompts to process")
        sys.exit(1)

    responses = _run_generation_via_server(
        conversations,
        model_id=model_id,
        generation=generation,
        enable_thinking=enable_thinking,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        reasoning_parser=reasoning_parser,
    )

    logger.info(f"Writing responses to column '{output_column}'")
    dataset = dataset.add_column(output_column, responses)

    logger.info(f"Writing parquet dataset to: {output_dataset_path}")
    dataset.to_parquet(output_dataset_path)
    logger.info("Generation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses for parquet dataset prompts using a local vLLM server")
    parser.add_argument("src_dataset_path", help="Input parquet dataset path")
    parser.add_argument("output_dataset_path", help="Output parquet dataset path")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Model to use for generation")
    parser.add_argument("--messages-column", type=str, default="messages", help="Column containing chat messages")
    parser.add_argument("--reasoning-parser", type=str, help="vLLM reasoning parser to use for supported models")
    parser.add_argument("--enforce-eager", action="store_true", default=False, help="Disable torch.compile and CUDA graph capture in vLLM")
    parser.add_argument("--prompt-column", type=str, help="Column containing plain text prompts")
    parser.add_argument("--output-column", type=str, default="outputs", help="Column name for generated responses")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, help="Top-k sampling parameter")
    parser.add_argument("--min-p", type=float, help="Minimum probability threshold")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--repetition-penalty", type=float, help="Repetition penalty")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilization factor")
    parser.add_argument("--max-model-len", type=int, help="Maximum model context length")
    parser.add_argument("--tensor-parallel-size", type=int, help="Number of GPUs to use")
    parser.add_argument("--enable-thinking", action="store_true", default=False, help="Enable model thinking/reasoning when supported")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument("--skip-long-prompts", action="store_true", default=True, help="Keep CLI compatibility; prompt pre-filtering is not used in chat mode")
    parser.add_argument("--no-skip-long-prompts", dest="skip_long_prompts", action="store_false", help="Keep CLI compatibility; prompt pre-filtering is not used in chat mode")

    args = parser.parse_args()
    main(
        src_dataset_path=args.src_dataset_path,
        output_dataset_path=args.output_dataset_path,
        model_id=args.model_id,
        reasoning_parser=args.reasoning_parser,
        enforce_eager=args.enforce_eager,
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
