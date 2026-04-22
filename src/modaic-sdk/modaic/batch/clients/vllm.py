from __future__ import annotations

import inspect
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

import dspy

from ..enqueued_limits import none_enqueued_limits
from ..token_counting import count_tokens_none
from ..types import BatchRequestItem, ResultItem, ShardOutcome
from .base import BatchClient, ShardReporter, _extract_openai_compatible_message, _stringify_content

logger = logging.getLogger(__name__)


def _require_vllm() -> None:
    try:
        import vllm  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "modaic.batch.vllm requires the vLLM package for vLLM batch jobs. "
            'Install it with `uv add "modaic[vllm]"`.'
        ) from exc


class VLLMBatchClient(BatchClient):
    """Local vLLM batch client. Engine lives in session(); each shard is one run_batch call.

    Cache partition/merge is owned by the runner. Concurrency is locked to sequential
    since only one engine run_batch call can be in flight at a time.
    """

    name = "vllm"
    reqs_per_file = sys.maxsize
    max_file_size = sys.maxsize
    concurrency = "sequential"
    safety_margin = 1.0
    endpoint = "/v1/chat/completions"
    resumable = False
    token_counter = staticmethod(count_tokens_none)
    enqueued_limits_fn = staticmethod(none_enqueued_limits)

    def __init__(
        self,
        lm: dspy.LM,
        *,
        batch_size: Optional[int] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        reasoning_parser: Optional[str] = None,
        enforce_eager: bool = False,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: Optional[int] = None,
        data_parallel_size: Optional[int] = None,
        on_chunk_complete: Optional[Callable[[], None]] = None,
        reqs_per_file: Optional[int] = None,
        max_file_size: Optional[int] = None,
        tokens_per_file: Optional[int] = None,
        default_enqueued_reqs: Optional[int] = None,
        default_enqueued_tokens: Optional[int] = None,
        default_enqueued_jobs: Optional[int] = None,
        enable_concurrent_jobs: Optional[bool] = None,
    ):
        model = getattr(lm, "model", None)
        if not isinstance(model, str) or not model.startswith("huggingface/"):
            raise ValueError("VLLMBatchClient requires a dspy.LM with model='huggingface/<repo_path>'")

        # batch_size maps to reqs_per_file unless the explicit override is given.
        if reqs_per_file is None and batch_size and batch_size > 0:
            reqs_per_file = batch_size

        super().__init__(
            api_key=None,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            reqs_per_file=reqs_per_file,
            max_file_size=max_file_size,
            tokens_per_file=tokens_per_file,
            default_enqueued_reqs=default_enqueued_reqs,
            default_enqueued_tokens=default_enqueued_tokens,
            default_enqueued_jobs=default_enqueued_jobs,
            enable_concurrent_jobs=enable_concurrent_jobs,
        )
        self.lm = lm
        self.model_id = model.removeprefix("huggingface/")
        self.batch_size = batch_size
        self.reasoning_parser = reasoning_parser or ""
        self.enforce_eager = enforce_eager
        self.enable_thinking = enable_thinking or bool(reasoning_parser)
        self.thinking_budget = thinking_budget
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.on_chunk_complete = on_chunk_complete
        self._engine_client: Any = None

    def format_line(self, item: BatchRequestItem) -> dict[str, Any]:
        return {
            "custom_id": item["id"],
            "method": "POST",
            "url": self.endpoint,
            "body": self._build_body(item),
        }

    def parse_result(self, raw: dict[str, Any]) -> ResultItem:
        error = raw.get("error")
        if isinstance(error, str) and error.strip():
            raise ValueError(error.strip())

        response = raw.get("response", {})
        if isinstance(response, dict) and isinstance(response.get("text"), str):
            result: ResultItem = {"text": response["text"]}
            reasoning = response.get("reasoning_content")
            if reasoning is not None:
                result["reasoning_content"] = str(reasoning)
            return result

        body = response.get("body", {})
        choices = body.get("choices", [])
        if not choices:
            err = raw.get("error") or body.get("error", {})
            raise ValueError(f"Batch request failed: {err}")

        choice = choices[0]
        message = choice.get("message", {})
        text, reasoning = _extract_openai_compatible_message(message)
        if reasoning is None and message.get("reasoning") is not None:
            reasoning = _stringify_content(message["reasoning"])

        result: ResultItem = {"text": text}
        if reasoning is not None:
            result["reasoning_content"] = reasoning
        if choice.get("logprobs") is not None:
            result["logprobs"] = choice["logprobs"]
        if message.get("tool_calls") is not None:
            result["tool_calls"] = message["tool_calls"]
        return result

    def _build_body(self, item: BatchRequestItem) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model_id, "messages": item["messages"]}
        body.update({k: v for k, v in item.get("lm_kwargs", {}).items() if k != "api_key"})
        if self.enable_thinking:
            chat_template_kwargs = dict(body.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = True
            body["chat_template_kwargs"] = chat_template_kwargs
            if self.thinking_budget is not None:
                body["thinking_token_budget"] = self.thinking_budget
        return body

    def _build_cli_args(self, input_file: str = "/dev/null", output_file: str = "/dev/null") -> list[str]:
        cli: list[str] = [
            "--input-file", input_file,
            "--output-file", output_file,
            "--model", self.model_id,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        if self.reasoning_parser:
            cli.extend(["--reasoning-parser", self.reasoning_parser])
        if self.enforce_eager:
            cli.append("--enforce-eager")
        if self.max_model_len is not None:
            cli.extend(["--max-model-len", str(self.max_model_len)])
        if self.tensor_parallel_size is not None:
            cli.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        if self.data_parallel_size is not None:
            cli.extend(["--data-parallel-size", str(self.data_parallel_size)])
        return cli

    def _parse_vllm_args(self, input_file: str = "/dev/null", output_file: str = "/dev/null"):
        _require_vllm()
        from vllm.entrypoints.openai.run_batch import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser(description="vLLM batch runner")
        parser = make_arg_parser(parser)
        return parser.parse_args(self._build_cli_args(input_file, output_file))

    @asynccontextmanager
    async def session(self) -> AsyncIterator[None]:
        if self._engine_client is not None:
            yield
            return

        args = self._parse_vllm_args()
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
        from vllm.usage.usage_lib import UsageContext
        engine_args = AsyncEngineArgs.from_cli_args(args)

        logger.info(
            "VLLMBatchClient starting engine model=%s tp=%s gpu_mem=%s",
            self.model_id, self.tensor_parallel_size, self.gpu_memory_utilization,
        )
        async with build_async_engine_client_from_engine_args(
            engine_args, usage_context=UsageContext.OPENAI_BATCH_RUNNER,
        ) as engine_client:
            self._engine_client = engine_client
            try:
                yield
            finally:
                self._engine_client = None

    async def execute_shard(self, shard: Path, reporter: ShardReporter) -> ShardOutcome:
        _require_vllm()
        from vllm.entrypoints.openai.run_batch import run_batch

        if self._engine_client is None:
            raise RuntimeError(
                "VLLMBatchClient.execute_shard called outside session(). Runner must wrap in `async with client.session():`."
            )

        batch_id = f"vllm-{uuid.uuid4().hex[:8]}"
        reporter.started(batch_id)

        output_path = shard.with_suffix(shard.suffix + ".out")
        args = self._parse_vllm_args(str(shard), str(output_path))

        # vLLM's EngineArgs.create_engine_config() transfers top-level CLI values into
        # nested config objects; the batch runner's init_app_state reads the nested
        # config but never calls create_engine_config(). Replicate the transfers here.
        so_cfg = getattr(args, "structured_outputs_config", None)
        if so_cfg is not None:
            for attr in ("reasoning_parser", "reasoning_parser_plugin"):
                val = getattr(args, attr, None)
                if val and hasattr(so_cfg, attr):
                    setattr(so_cfg, attr, val)

        start = time.time()
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        try:
            await run_batch(self._engine_client, args)
            with output_path.open() as of:
                for raw_line in of:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    row = json.loads(stripped)
                    if row.get("error"):
                        errors.append(row)
                    else:
                        results.append(row)
        finally:
            output_path.unlink(missing_ok=True)

        reporter.percent(100)
        logger.info(
            "VLLMBatchClient shard %s complete: results=%d errors=%d duration_s=%.1f",
            batch_id, len(results), len(errors), time.time() - start,
        )

        if self.on_chunk_complete is not None:
            cb_result = self.on_chunk_complete()
            if inspect.isawaitable(cb_result):
                await cb_result

        return ShardOutcome(batch_id=batch_id, results=results, errors=errors)
