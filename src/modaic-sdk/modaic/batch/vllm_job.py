from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LocalVLLMRunner:
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

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
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

    def generate(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile("w", suffix=".jsonl", delete=False) as input_file:
            input_path = Path(input_file.name)
            for index, messages in enumerate(messages_batch):
                body: dict[str, Any] = {
                    "model": self.model_id,
                    "messages": messages,
                }
                if temperature is not None:
                    body["temperature"] = temperature
                if top_p is not None:
                    body["top_p"] = top_p
                if top_k is not None:
                    body["top_k"] = top_k
                if min_p is not None:
                    body["min_p"] = min_p
                if max_tokens is not None:
                    body["max_tokens"] = max_tokens
                if repetition_penalty is not None:
                    body["repetition_penalty"] = repetition_penalty
                if enable_thinking:
                    body["chat_template_kwargs"] = {"enable_thinking": True}
                    if thinking_budget is not None:
                        body["thinking_token_budget"] = thinking_budget

                input_file.write(
                    json.dumps(
                        {
                            "custom_id": f"request-{index}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": body,
                        }
                    )
                    + "\n"
                )

        output_path = input_path.with_suffix(".output.jsonl")
        command = self._build_command(input_path, output_path)
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        logger.info(
            "Starting vllm run-batch model=%s requests=%d input_path=%s output_path=%s",
            self.model_id,
            len(messages_batch),
            input_path,
            output_path,
        )
        logger.info("vllm run-batch command: %s", " ".join(command))

        start = time.time()
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise ModuleNotFoundError(
                'modaic.batch requires the `vllm` CLI for local run-batch jobs. Install it with `uv add "modaic[vllm]"`.'
            ) from exc

        output_lines: list[str] = []
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            output_lines.append(line)
            logger.info("vllm run-batch | %s", line)

        completed_returncode = process.wait()
        duration_s = time.time() - start
        logger.info(
            "vllm run-batch finished returncode=%s duration_s=%.1f output_lines=%d",
            completed_returncode,
            duration_s,
            len(output_lines),
        )

        try:
            if completed_returncode != 0:
                detail = "\n".join(output_lines[-50:]).strip() or f"exit code {completed_returncode}"
                raise RuntimeError(f"vllm run-batch failed: {detail}")

            raw_rows: list[dict[str, Any]] = []
            with output_path.open() as output_file:
                for line in output_file:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
            logger.info("Parsed %d rows from vllm run-batch output", len(raw_rows))
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        output_rows: list[dict[str, Any]] = []
        failures = 0
        reasoning_rows = 0
        prompt_tokens = 0
        output_tokens = 0

        for row in raw_rows:
            error = row.get("error")
            if error:
                failures += 1

            response = row.get("response") or {}
            body = response.get("body") or {}
            usage = body.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)

            choices = body.get("choices") or []
            text = ""
            reasoning_content = None
            if choices:
                message = (choices[0] or {}).get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    text = content
                reasoning_content = message.get("reasoning_content")
                if reasoning_content is None:
                    reasoning_content = message.get("reasoning")
                if reasoning_content is not None:
                    reasoning_rows += 1

            output_rows.append(
                {
                    "response": {"text": text, "reasoning_content": reasoning_content},
                    "error": None if error in (None, "") else str(error),
                }
            )

        return output_rows, {
            "total": len(output_rows),
            "failures": failures,
            "reasoning_rows": reasoning_rows,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "duration_s": duration_s,
        }

    def close(self) -> None:
        return None
