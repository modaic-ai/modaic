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
"""Generate responses for prompts stored in a local parquet dataset using vLLM."""

import argparse
import logging
import os
import sys
from typing import Optional

from datasets import load_dataset
from huggingface_hub import get_token
from torch import cuda
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_output_payload(output) -> dict[str, Optional[str]]:
    """Convert a vLLM chat result into a parquet-serializable dict."""
    completion = output.outputs[0] if getattr(output, "outputs", None) else None
    text = getattr(completion, "text", "") if completion is not None else ""
    print("completion", completion)

    reasoning_content = None
    if completion is not None:
        reasoning_content = getattr(completion, "reasoning_content", None)
        if reasoning_content is None:
            reasoning_content = getattr(completion, "reasoning", None)

    return {
        "text": text,
        "reasoning_content": reasoning_content,
    }


def check_gpu_availability() -> int:
    """Check if CUDA is available and return the number of GPUs."""
    if not cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)

    num_gpus = cuda.device_count()
    for i in range(num_gpus):
        gpu_name = cuda.get_device_name(i)
        gpu_memory = cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name} with {gpu_memory:.1f} GB memory")

    return num_gpus


def main(
    src_dataset_path: str,
    output_dataset_path: str,
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
) -> None:
    """Generate model responses from a parquet dataset and write the result to parquet."""
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

    logger.info(f"Loading model: {model_id}")
    vllm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len
        logger.info(f"Using max_model_len={max_model_len}")
    if reasoning_parser is not None:
        vllm_kwargs["reasoning_parser"] = reasoning_parser
        logger.info(f"Using reasoning_parser={reasoning_parser}")
    llm = LLM(**vllm_kwargs)

    sampling_params = SamplingParams(
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
        logger.info(
            "Prompt length pre-filtering is disabled when using llm.chat(); model limits will be enforced at inference time"
        )

    conversations = []
    for example in tqdm(dataset, desc="Processing prompts"):
        if use_messages:
            messages = example[messages_column]
        else:
            messages = [{"role": "user", "content": example[prompt_column]}]
        conversations.append(messages)

    if not conversations:
        logger.error("No prompts to process")
        sys.exit(1)

    logger.info(f"Starting chat generation for {len(conversations):,} prompts")
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )

    responses = [{"text": "", "reasoning_content": None} for _ in range(total_examples)]
    for idx, output in enumerate(outputs):
        responses[idx] = extract_output_payload(output)

    logger.info(f"Writing responses to column '{output_column}'")
    dataset = dataset.add_column(output_column, responses)

    logger.info(f"Writing parquet dataset to: {output_dataset_path}")
    dataset.to_parquet(output_dataset_path)
    logger.info("Generation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses for parquet dataset prompts using vLLM")
    parser.add_argument("src_dataset_path", help="Input parquet dataset path")
    parser.add_argument("output_dataset_path", help="Output parquet dataset path")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--messages-column",
        type=str,
        default="messages",
        help="Column containing chat messages",
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        help="vLLM reasoning parser to use for supported models",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        help="Column containing plain text prompts",
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default="outputs",
        help="Column name for generated responses",
    )
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling parameter")
    parser.add_argument("--min-p", type=float, default=0.0, help="Minimum probability threshold")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Maximum tokens to generate")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization factor",
    )
    parser.add_argument("--max-model-len", type=int, help="Maximum model context length")
    parser.add_argument("--tensor-parallel-size", type=int, help="Number of GPUs to use")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable model thinking/reasoning when supported",
    )
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument(
        "--skip-long-prompts",
        action="store_true",
        default=True,
        help="Keep CLI compatibility; prompt pre-filtering is not used in chat mode",
    )
    parser.add_argument(
        "--no-skip-long-prompts",
        dest="skip_long_prompts",
        action="store_false",
        help="Keep CLI compatibility; prompt pre-filtering is not used in chat mode",
    )

    args = parser.parse_args()
    main(
        src_dataset_path=args.src_dataset_path,
        output_dataset_path=args.output_dataset_path,
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
