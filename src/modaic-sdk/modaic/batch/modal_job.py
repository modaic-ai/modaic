"""Modal-based vLLM offline batch generation using uploaded `jsonl.gz` and `vllm run-batch`."""

import gzip
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import modal

APP_NAME = "modaic-generate-responses"
VOLUME_ROOT = "/cache"
INPUT_FILENAME = "input.jsonl.gz"
OUTPUT_FILENAME = "output.jsonl.gz"
SHARED_VOLUME_NAME = "modaic-generate-responses-cache"

app = modal.App(APP_NAME)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
cache_volume = modal.Volume.from_name(SHARED_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.18.0",
        "huggingface-hub",
        "hf_transfer",
        "hf-xet>=1.1.7",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache_vol,
    "/root/.cache/vllm": vllm_cache_vol,
    VOLUME_ROOT: cache_volume,
}

# Sentinel for "not set" since modal.parameter() doesn't support Optional
_UNSET_INT = -1


@app.cls(
    gpu="H100",
    image=image,
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
)
class ResponseGenerator:
    # modal.parameter() only supports str, int, bool, bytes
    model_id: str = modal.parameter(default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    reasoning_parser: str = modal.parameter(default="")
    enforce_eager: bool = modal.parameter(default=False)
    max_model_len: int = modal.parameter(default=_UNSET_INT)
    # gpu_memory_utilization as int percentage (90 = 0.90)
    gpu_memory_utilization_pct: int = modal.parameter(default=90)
    tensor_parallel_size: int = modal.parameter(default=_UNSET_INT)

    @modal.enter()
    def start(self):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        if shutil.which("vllm") is None:
            raise ModuleNotFoundError(
                'modaic.batch requires the `vllm` CLI for Modal batch jobs. Install it with `uv add "modaic[vllm]"`.'
            )

        print(
            f"ResponseGenerator.start model={self.model_id} reasoning_parser={self.reasoning_parser or 'none'} enforce_eager={self.enforce_eager} tensor_parallel_size={self.tensor_parallel_size}"
        )
        print(f"ResponseGenerator.start pid={os.getpid()} vllm_path={shutil.which('vllm')}")

    def _build_command(self, input_path: Path, output_jsonl_path: Path) -> list[str]:
        command = [
            "vllm",
            "run-batch",
            "-i",
            str(input_path),
            "-o",
            str(output_jsonl_path),
            "--model",
            self.model_id,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization_pct / 100.0),
        ]
        if self.reasoning_parser:
            command.extend(["--reasoning-parser", self.reasoning_parser])
        if self.enforce_eager:
            command.append("--enforce-eager")
        if self.max_model_len != _UNSET_INT:
            command.extend(["--max-model-len", str(self.max_model_len)])
        if self.tensor_parallel_size != _UNSET_INT:
            command.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        return command

    @modal.method()
    def generate(
        self,
        input_path: str,
        output_path: str,
    ) -> dict[str, Any]:
        print(f"ResponseGenerator.generate start input={input_path} output={output_path}")
        cache_volume.reload()
        input_jsonl_gz_path = Path(input_path)
        input_jsonl_path = Path("/tmp") / f"modal-vllm-{int(time.time() * 1000)}.input.jsonl"
        output_jsonl_path = input_jsonl_path.with_suffix(".output.jsonl")
        output_jsonl_gz_path = Path(output_path)
        print(
            f"ResponseGenerator.generate volume reloaded input_gz_exists={input_jsonl_gz_path.exists()} input_gz_size={input_jsonl_gz_path.stat().st_size if input_jsonl_gz_path.exists() else 'missing'}"
        )

        with gzip.open(input_jsonl_gz_path, "rt") as src, input_jsonl_path.open("w") as dst:
            dst.write(src.read())
        print(
            f"ResponseGenerator.generate decompressed input_jsonl_exists={input_jsonl_path.exists()} size={input_jsonl_path.stat().st_size if input_jsonl_path.exists() else 'missing'}"
        )

        command = self._build_command(input_jsonl_path, output_jsonl_path)
        print(f"ResponseGenerator.generate command={command}")
        start = time.time()
        completed = subprocess.run(command, check=False, env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        duration_s = time.time() - start
        print(
            f"ResponseGenerator.generate subprocess returncode={completed.returncode} duration_s={duration_s:.1f}"
        )
        if completed.returncode != 0:
            detail = f"exit code {completed.returncode}"
            raise RuntimeError(f"vllm run-batch failed: {detail}")

        failures = 0
        reasoning_rows = 0
        prompt_tokens = 0
        output_tokens = 0
        output_records: list[dict[str, object]] = []

        try:
            with output_jsonl_path.open() as output_file:
                for line in output_file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    output_records.append(record)
                    error = record.get("error")
                    response = record.get("response") or {}
                    body = response.get("body") or {}
                    usage = body.get("usage") or {}
                    prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    output_tokens += int(usage.get("completion_tokens") or 0)

                    choices = body.get("choices") or []
                    if choices:
                        message = (choices[0] or {}).get("message") or {}
                        if message.get("reasoning") is not None or message.get("reasoning_content") is not None:
                            reasoning_rows += 1
                    if error:
                        failures += 1
        finally:
            input_jsonl_path.unlink(missing_ok=True)
            output_jsonl_path.unlink(missing_ok=True)
        print(
            f"ResponseGenerator.generate parsed records={len(output_records)} failures={failures} reasoning_rows={reasoning_rows} prompt_tokens={prompt_tokens} output_tokens={output_tokens}"
        )
        with gzip.open(output_jsonl_gz_path, "wt") as f:
            for record in output_records:
                f.write(json.dumps(record) + "\n")
        print(
            f"ResponseGenerator.generate wrote output_gz path={output_jsonl_gz_path} size={output_jsonl_gz_path.stat().st_size if output_jsonl_gz_path.exists() else 'missing'}"
        )
        cache_volume.commit()
        print("ResponseGenerator.generate volume committed")

        return {
            "total": len(output_records),
            "failures": failures,
            "reasoning_rows": reasoning_rows,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "duration_s": duration_s,
        }

    @modal.exit()
    def stop(self):
        return None
