from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional


class LocalVLLMRunBatchRunner:
    def __init__(
        self,
        model_id: str,
        reasoning_parser: str = "",
        enforce_eager: bool = False,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: Optional[int] = None,
    ) -> None:
        if shutil.which("vllm") is None:
            raise ModuleNotFoundError(
                'modaic.batch requires the `vllm` CLI for local run-batch jobs. Install it with `uv add "modaic[vllm]"`.'
            )

        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.enforce_eager = enforce_eager
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size

    def _build_command(self, input_path: Path, output_path: Path) -> list[str]:
        command = [
            "vllm",
            "run-batch",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--model",
            self.model_id,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        if self.reasoning_parser:
            command.extend(["--reasoning-parser", self.reasoning_parser])
        if self.enforce_eager:
            command.append("--enforce-eager")
        if self.max_model_len is not None:
            command.extend(["--max-model-len", str(self.max_model_len)])
        if self.tensor_parallel_size is not None:
            command.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        return command

    def run_batch(self, input_path: str | Path, output_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        resolved_input = Path(input_path)
        resolved_output = Path(output_path)
        command = self._build_command(resolved_input, resolved_output)
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        start = time.time()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        except FileNotFoundError as exc:
            raise ModuleNotFoundError(
                'modaic.batch requires the `vllm` CLI for local run-batch jobs. Install it with `uv add "modaic[vllm]"`.'
            ) from exc
        duration_s = time.time() - start

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"vllm run-batch failed: {detail}")

        rows: list[dict[str, Any]] = []
        with resolved_output.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        failures = 0
        reasoning_rows = 0
        prompt_tokens = 0
        output_tokens = 0

        for row in rows:
            error = row.get("error")
            if error:
                failures += 1

            response = row.get("response") or {}
            body = response.get("body") or {}
            usage = body.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)

            choices = body.get("choices") or []
            if not choices:
                continue
            message = (choices[0] or {}).get("message") or {}
            reasoning = message.get("reasoning")
            reasoning_content = message.get("reasoning_content")
            if reasoning is not None or reasoning_content is not None:
                reasoning_rows += 1

        return rows, {
            "total": len(rows),
            "failures": failures,
            "reasoning_rows": reasoning_rows,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "duration_s": duration_s,
        }

    def close(self) -> None:
        return None
