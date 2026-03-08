from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import modal
import pandas as pd

APP_NAME = "modaic-generate-responses"
VOLUME_ROOT = "/cache"
INPUT_FILENAME = "input.parquet"
OUTPUT_FILENAME = "output.parquet"
SHARED_VOLUME_NAME = "modaic-generate-responses-cache"
SCRIPT_REMOTE_PATH = "/root/modaic/generate_responses_modal.py"
SCRIPT_LOCAL_PATH = Path(__file__).parent / "scripts" / "generate_responses_modal.py"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(SHARED_VOLUME_NAME, create_if_missing=True)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
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
        "platformdirs",
    )
    .add_local_file(str(SCRIPT_LOCAL_PATH), SCRIPT_REMOTE_PATH)
)


def _append_optional_arg(args: list[str], flag: str, value: Optional[str | int | float]) -> None:
    if value is not None:
        args.extend([flag, str(value)])


def _volume_path(filename: str) -> str:
    return f"{VOLUME_ROOT}/{filename}"


def _build_cli_args(
    src_dataset_path: str,
    output_dataset_path: str,
    model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    reasoning_parser: Optional[str] = None,
    enforce_eager: bool = False,
    messages_column: str = "messages",
    prompt_column: Optional[str] = None,
    output_column: str = "response",
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
) -> list[str]:
    args = [
        "python",
        SCRIPT_REMOTE_PATH,
        src_dataset_path,
        output_dataset_path,
        "--model-id",
        model_id,
        "--messages-column",
        messages_column,
        "--output-column",
        output_column,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]

    _append_optional_arg(args, "--reasoning-parser", reasoning_parser)
    _append_optional_arg(args, "--prompt-column", prompt_column)
    _append_optional_arg(args, "--temperature", temperature)
    _append_optional_arg(args, "--top-p", top_p)
    _append_optional_arg(args, "--top-k", top_k)
    _append_optional_arg(args, "--min-p", min_p)
    _append_optional_arg(args, "--max-tokens", max_tokens)
    _append_optional_arg(args, "--repetition-penalty", repetition_penalty)
    _append_optional_arg(args, "--max-model-len", max_model_len)
    _append_optional_arg(args, "--tensor-parallel-size", tensor_parallel_size)
    _append_optional_arg(args, "--max-samples", max_samples)
    _append_optional_arg(args, "--hf-token", hf_token)

    if enforce_eager:
        args.append("--enforce-eager")

    if skip_long_prompts:
        args.append("--skip-long-prompts")
    else:
        args.append("--no-skip-long-prompts")

    if enable_thinking:
        args.append("--enable-thinking")

    return args


@app.cls(
    gpu="H200:4",
    image=image,
    timeout=60 * 60 * 24,
    volumes={VOLUME_ROOT: cache_volume},
)
class ResponseGenerator:
    @modal.method()
    def run_generate_responses(
        self,
        src_dataset_path: str,
        output_dataset_path: str,
        model_id: str = "openai/gpt-oss-120b",
        reasoning_parser: Optional[str] = None,
        enforce_eager: bool = False,
        messages_column: str = "messages",
        prompt_column: Optional[str] = None,
        output_column: str = "response",
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
        cli_args = _build_cli_args(
            src_dataset_path=src_dataset_path,
            output_dataset_path=output_dataset_path,
            model_id=model_id,
            reasoning_parser=reasoning_parser,
            enforce_eager=enforce_eager,
            messages_column=messages_column,
            prompt_column=prompt_column,
            output_column=output_column,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            skip_long_prompts=skip_long_prompts,
            enable_thinking=enable_thinking,
            max_samples=max_samples,
            hf_token=hf_token,
        )
        subprocess.run(cli_args, check=True)
        cache_volume.commit()


@app.local_entrypoint()
def main(
    src_dataset_path: str,
    output_dataset_path: str,
    model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    reasoning_parser: Optional[str] = None,
    enforce_eager: bool = False,
    messages_column: str = "messages",
    prompt_column: Optional[str] = None,
    output_column: str = "response",
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
    gpu: str = "H200:4",
) -> None:
    from logging import getLogger

    local_logger = getLogger(__name__)

    caller_cwd = Path(os.environ.get("PWD", os.getcwd()))
    src_path = Path(src_dataset_path).expanduser()
    if not src_path.is_absolute():
        src_path = caller_cwd / src_path
    src_dataset_path = str(src_path.resolve())

    output_path = Path(output_dataset_path).expanduser()
    if not output_path.is_absolute():
        output_path = caller_cwd / output_path
    output_dataset_path = str(output_path.resolve())

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    tmp_output_dir: Optional[str] = None

    try:
        with cache_volume.batch_upload(force=True) as batch:
            batch.put_file(src_dataset_path, f"/{INPUT_FILENAME}")

        ResponseGenerator.with_options(gpu=gpu)().run_generate_responses.remote(
            src_dataset_path=_volume_path(INPUT_FILENAME),
            output_dataset_path=_volume_path(OUTPUT_FILENAME),
            model_id=model_id,
            reasoning_parser=reasoning_parser,
            enforce_eager=enforce_eager,
            messages_column=messages_column,
            prompt_column=prompt_column,
            output_column=output_column,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            skip_long_prompts=skip_long_prompts,
            enable_thinking=enable_thinking,
            max_samples=max_samples,
            hf_token=hf_token,
        )

        local_logger.info("Downloading results from modal")
        output_parent = Path(output_dataset_path).parent
        output_parent.mkdir(parents=True, exist_ok=True)

        tmp_output_dir = tempfile.mkdtemp(prefix="modaic-modal-output-")
        downloaded_output = Path(tmp_output_dir) / OUTPUT_FILENAME
        client_volume = modal.Volume.from_name(SHARED_VOLUME_NAME)
        downloaded_output.write_bytes(b"".join(client_volume.read_file(OUTPUT_FILENAME)))
        downloaded_output.replace(output_dataset_path)
        local_logger.info(f"Results downloaded successfully to {output_dataset_path}")

        preview_df = pd.read_parquet(output_dataset_path)
        print("Results preview:")
        print(preview_df[[output_column]].head(5).to_string(index=False))
    finally:
        if tmp_output_dir is not None:
            shutil.rmtree(tmp_output_dir, ignore_errors=True)
