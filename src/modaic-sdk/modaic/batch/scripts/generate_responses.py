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
Generate responses for prompts in a dataset using vLLM for efficient GPU inference.

This script loads a dataset from Hugging Face Hub containing chat-formatted messages,
applies the model's chat template, generates responses using vLLM, and saves the
results back to the Hub with a comprehensive dataset card.

Example usage:
    # Local execution with auto GPU detection
    uv run generate_responses.py \\
        username/input-dataset \\
        username/output-dataset \\
        --messages-column messages

    # With custom model and sampling parameters
    uv run generate_responses.py \\
        username/input-dataset \\
        username/output-dataset \\
        --model-id meta-llama/Llama-3.1-8B-Instruct \\
        --temperature 0.9 \\
        --top-p 0.95 \\
        --max-tokens 2048

    # HF Jobs execution (see script output for full command)
    hf jobs uv run --flavor a100x4 ...
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from huggingface_hub import DatasetCard, get_token, login
from torch import cuda
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# Enable HF Transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UV_SCRIPT_REPO_ID = "modaic/batch-vllm"
UV_SCRIPT_FILENAME = "generate_responses.py"
UV_SCRIPT_URL = f"https://huggingface.co/datasets/{UV_SCRIPT_REPO_ID}/resolve/main/{UV_SCRIPT_FILENAME}"


def extract_output_payload(output) -> dict[str, Optional[str]]:
    """Convert a vLLM chat result into a dataset-serializable dict."""
    completion = output.outputs[0] if getattr(output, "outputs", None) else None
    text = getattr(completion, "text", "") if completion is not None else ""
    print("COMPLETION", completion)
    print("hasattr reasoning_content", hasattr(completion, "reasoning_content"))
    print("hasattr reasoning", hasattr(completion, "reasoning"))
    print("completion attrs", dir(completion))

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
        logger.error("Please run on a machine with NVIDIA GPU or use HF Jobs with GPU flavor.")
        sys.exit(1)

    num_gpus = cuda.device_count()
    for i in range(num_gpus):
        gpu_name = cuda.get_device_name(i)
        gpu_memory = cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name} with {gpu_memory:.1f} GB memory")

    return num_gpus


def create_dataset_card(
    source_dataset: str,
    model_id: str,
    messages_column: str,
    prompt_column: Optional[str],
    sampling_params: SamplingParams,
    tensor_parallel_size: int,
    num_examples: int,
    generation_time: str,
    num_skipped: int = 0,
    max_model_len_used: Optional[int] = None,
) -> str:
    """Create a comprehensive dataset card documenting the generation process."""
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

- **Temperature**: {sampling_params.temperature}
- **Top P**: {sampling_params.top_p}
- **Top K**: {sampling_params.top_k}
- **Min P**: {sampling_params.min_p}
- **Max Tokens**: {sampling_params.max_tokens}
- **Repetition Penalty**: {sampling_params.repetition_penalty}

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
    --temperature {sampling_params.temperature} \\
    --top-p {sampling_params.top_p} \\
    --top-k {sampling_params.top_k} \\
    --max-tokens {sampling_params.max_tokens}{max_model_len_flag}
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
    """
    Main generation pipeline.

    Args:
        src_dataset_hub_id: Input dataset on Hugging Face Hub
        output_dataset_hub_id: Where to save results on Hugging Face Hub
        model_id: Hugging Face model ID for generation
        reasoning_parser: Optional vLLM reasoning parser to enable structured reasoning extraction
        messages_column: Column name containing chat messages
        prompt_column: Column name containing plain text prompts (alternative to messages_column)
        output_column: Column name for generated responses
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        min_p: Minimum probability threshold
        max_tokens: Maximum tokens to generate
        repetition_penalty: Repetition penalty parameter
        gpu_memory_utilization: GPU memory utilization factor
        max_model_len: Maximum model context length (None uses model default)
        tensor_parallel_size: Number of GPUs to use (auto-detect if None)
        skip_long_prompts: Deprecated. Prompt pre-filtering is not used in chat mode.
        enable_thinking: Enable model thinking/reasoning when supported by the chat template
        max_samples: Maximum number of samples to process (None for all)
        hf_token: Hugging Face authentication token
    """
    generation_start_time = datetime.now().isoformat()

    # GPU check and configuration
    num_gpus = check_gpu_availability()
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus
        logger.info(f"Auto-detected {num_gpus} GPU(s), using tensor_parallel_size={tensor_parallel_size}")
    else:
        logger.info(f"Using specified tensor_parallel_size={tensor_parallel_size}")
        if tensor_parallel_size > num_gpus:
            logger.warning(f"Requested {tensor_parallel_size} GPUs but only {num_gpus} available")

    # Authentication - try multiple methods
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN") or get_token()

    if not HF_TOKEN:
        logger.error("No HuggingFace token found. Please provide token via:")
        logger.error("  1. --hf-token argument")
        logger.error("  2. HF_TOKEN environment variable")
        logger.error("  3. Run 'huggingface-cli login' or use login() in Python")
        sys.exit(1)

    logger.info("HuggingFace token found, authenticating...")
    login(token=HF_TOKEN)

    # Initialize vLLM
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

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )

    # Load dataset
    logger.info(f"Loading dataset: {src_dataset_hub_id}")
    dataset = load_dataset(src_dataset_hub_id, split="train")

    # Apply max_samples if specified
    if max_samples is not None and max_samples < len(dataset):
        logger.info(f"Limiting dataset to {max_samples} samples")
        dataset = dataset.select(range(max_samples))

    total_examples = len(dataset)
    logger.info(f"Dataset loaded with {total_examples:,} examples")

    # Determine which column to use and validate
    if prompt_column:
        # Use prompt column mode
        if prompt_column not in dataset.column_names:
            logger.error(f"Column '{prompt_column}' not found. Available columns: {dataset.column_names}")
            sys.exit(1)
        logger.info(f"Using prompt column mode with column: '{prompt_column}'")
        use_messages = False
    else:
        # Use messages column mode
        if messages_column not in dataset.column_names:
            logger.error(f"Column '{messages_column}' not found. Available columns: {dataset.column_names}")
            sys.exit(1)
        logger.info(f"Using messages column mode with column: '{messages_column}'")
        use_messages = True

    if skip_long_prompts:
        logger.info(
            "Prompt length pre-filtering is disabled when using llm.chat(); model limits will be enforced at inference time"
        )

    logger.info("Preparing chat messages...")
    conversations = []

    for example in tqdm(dataset, desc="Processing prompts"):
        if use_messages:
            messages = example[messages_column]
        else:
            user_prompt = example[prompt_column]
            messages = [{"role": "user", "content": user_prompt}]

        conversations.append(messages)

    if not conversations:
        logger.error("No prompts to process!")
        sys.exit(1)

    # Generate responses - vLLM handles batching internally
    logger.info(f"Starting chat generation for {len(conversations):,} prompts...")
    logger.info("vLLM will handle batching and scheduling automatically")

    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )

    # Extract generated text and create full response list
    logger.info("Extracting generated responses...")
    responses = [{"text": "", "reasoning_content": None} for _ in range(total_examples)]

    for idx, output in enumerate(outputs):
        responses[idx] = extract_output_payload(output)

    # Add responses to dataset
    logger.info("Adding responses to dataset...")
    dataset = dataset.add_column(output_column, responses)

    # Create dataset card
    logger.info("Creating dataset card...")
    card_content = create_dataset_card(
        source_dataset=src_dataset_hub_id,
        model_id=model_id,
        messages_column=messages_column,
        prompt_column=prompt_column,
        sampling_params=sampling_params,
        tensor_parallel_size=tensor_parallel_size,
        num_examples=total_examples,
        generation_time=generation_start_time,
        num_skipped=0,
        max_model_len_used=max_model_len,
    )

    # Push dataset to hub
    logger.info(f"Pushing dataset to: {output_dataset_hub_id}")
    dataset.push_to_hub(output_dataset_hub_id, token=HF_TOKEN)

    # Push dataset card
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset_hub_id, token=HF_TOKEN)

    logger.info("✅ Generation complete!")
    logger.info(f"Dataset available at: https://huggingface.co/datasets/{output_dataset_hub_id}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Generate responses for dataset prompts using vLLM",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic usage with default Qwen model
  uv run generate-responses.py input-dataset output-dataset

  # With custom model and parameters
  uv run generate-responses.py input-dataset output-dataset \\
    --model-id meta-llama/Llama-3.1-8B-Instruct \\
    --temperature 0.9 \\
    --max-tokens 2048

  # Force specific GPU configuration
  uv run generate-responses.py input-dataset output-dataset \\
    --tensor-parallel-size 2 \\
    --gpu-memory-utilization 0.95

  # Using environment variable for token
  HF_TOKEN=hf_xxx uv run generate-responses.py input-dataset output-dataset
            """,
        )

        parser.add_argument(
            "src_dataset_hub_id",
            help="Input dataset on Hugging Face Hub (e.g., username/dataset-name)",
        )
        parser.add_argument("output_dataset_hub_id", help="Output dataset name on Hugging Face Hub")
        parser.add_argument(
            "--model-id",
            type=str,
            default="Qwen/Qwen3-30B-A3B-Instruct-2507",
            help="Model to use for generation (default: Qwen3-30B-A3B-Instruct-2507)",
        )
        parser.add_argument(
            "--messages-column",
            type=str,
            default="messages",
            help="Column containing chat messages (default: messages)",
        )
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            help="vLLM reasoning parser to use for supported models",
        )
        parser.add_argument(
            "--prompt-column",
            type=str,
            help="Column containing plain text prompts (alternative to --messages-column)",
        )
        parser.add_argument(
            "--output-column",
            type=str,
            default="outputs",
            help="Column name for generated responses (default: outputs)",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            help="Maximum number of samples to process (default: all)",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.7,
            help="Sampling temperature (default: 0.7)",
        )
        parser.add_argument(
            "--top-p",
            type=float,
            default=0.8,
            help="Top-p sampling parameter (default: 0.8)",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=20,
            help="Top-k sampling parameter (default: 20)",
        )
        parser.add_argument(
            "--min-p",
            type=float,
            default=0.0,
            help="Minimum probability threshold (default: 0.0)",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=16384,
            help="Maximum tokens to generate (default: 16384)",
        )
        parser.add_argument(
            "--repetition-penalty",
            type=float,
            default=1.0,
            help="Repetition penalty (default: 1.0)",
        )
        parser.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=0.90,
            help="GPU memory utilization factor (default: 0.90)",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            help="Maximum model context length (default: model's default)",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            type=int,
            help="Number of GPUs to use (default: auto-detect)",
        )
        parser.add_argument(
            "--enable-thinking",
            action="store_true",
            default=False,
            help="Enable model thinking/reasoning when supported (default: False)",
        )
        parser.add_argument(
            "--hf-token",
            type=str,
            help="Hugging Face token (can also use HF_TOKEN env var)",
        )
        parser.add_argument(
            "--skip-long-prompts",
            action="store_true",
            default=True,
            help="Skip prompts that exceed max_model_len instead of failing (default: True)",
        )
        parser.add_argument(
            "--no-skip-long-prompts",
            dest="skip_long_prompts",
            action="store_false",
            help="Fail on prompts that exceed max_model_len",
        )

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
        # Show HF Jobs example when run without arguments
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

Example HF Jobs command with multi-GPU:
    # If you're logged in with huggingface-cli, token will be auto-detected
    hf jobs uv run \\
        --flavor l4x4 \\
        --secrets HF_TOKEN \\
        {UV_SCRIPT_URL} \\
        username/input-dataset \\
        username/output-dataset \\
        --messages-column messages \\
        --model-id Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --temperature 0.7 \\
        --max-tokens 16384
        """)
