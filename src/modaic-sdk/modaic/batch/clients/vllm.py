from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Optional

import dspy

from ..types import BatchReponse, BatchRequest, ResultItem
from .base import BatchClient, _extract_openai_compatible_message, _stringify_content

if TYPE_CHECKING:
    from ..lmdb_cache import LmdbLMCache

logger = logging.getLogger(__name__)


class VLLMBatchClient(BatchClient):
    """Resumable vLLM batch client using the Python API with mini-batching and LMDB caching.

    Uses the vLLM Python API directly (``run_batch`` +
    ``build_async_engine_client_from_engine_args``).
    The engine is created once and reused across mini-batches.

    When *batch_size* is set, large batches are split into smaller mini-batches.
    When *cache* is provided, results are cached after each mini-batch so that a
    crash-and-restart only re-processes uncached work.
    """

    provider: str = "vllm"

    def __init__(
        self,
        lm: dspy.LM,
        *,
        batch_size: Optional[int] = None,
        cache: Optional[LmdbLMCache] = None,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
        reasoning_parser: Optional[str] = None,
        enforce_eager: bool = False,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: Optional[int] = None,
    ):
        model = getattr(lm, "model", None)
        if not isinstance(model, str) or not model.startswith("huggingface/"):
            raise ValueError("VLLMBatchClient requires a dspy.LM with model='huggingface/<repo_path>'")

        super().__init__(
            api_key=None,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )
        self.lm = lm
        self.model_id = model.removeprefix("huggingface/")
        self.batch_size = batch_size
        self.cache = cache
        self.reasoning_parser = reasoning_parser or ""
        self.enforce_eager = enforce_eager
        self.enable_thinking = enable_thinking or bool(reasoning_parser)
        self.thinking_budget = thinking_budget
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self._engine_client: Any = None
        self._engine_ctx: Any = None

    # ------------------------------------------------------------------
    # Request body / format / parse
    # ------------------------------------------------------------------

    def _build_request_body(self, request: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": request["messages"],
        }
        body.update({k: v for k, v in request.get("lm_kwargs", {}).items() if k != "api_key"})

        if self.enable_thinking:
            chat_template_kwargs = dict(body.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = True
            body["chat_template_kwargs"] = chat_template_kwargs
            if self.thinking_budget is not None:
                body["thinking_token_budget"] = self.thinking_budget

        return body

    def format(self, batch_request: BatchRequest) -> list[dict]:
        return [
            {
                "custom_id": request["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": self._build_request_body(request),
            }
            for request in batch_request["requests"]
        ]

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        error = raw_result.get("error")
        if isinstance(error, str) and error.strip():
            raise ValueError(error.strip())

        response = raw_result.get("response", {})
        if isinstance(response, dict) and isinstance(response.get("text"), str):
            result: ResultItem = {"text": response["text"]}
            reasoning_content = response.get("reasoning_content")
            if reasoning_content is not None:
                result["reasoning_content"] = str(reasoning_content)
            return result

        body = response.get("body", {})
        choices = body.get("choices", [])

        if not choices:
            error = raw_result.get("error") or body.get("error", {})
            raise ValueError(f"Batch request failed: {error}")

        choice = choices[0]
        message = choice.get("message", {})

        text, reasoning_content = _extract_openai_compatible_message(message)
        if reasoning_content is None and message.get("reasoning") is not None:
            reasoning_content = _stringify_content(message["reasoning"])

        result: ResultItem = {"text": text}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        if "logprobs" in choice and choice["logprobs"] is not None:
            result["logprobs"] = choice["logprobs"]
        if "tool_calls" in message and message["tool_calls"] is not None:
            result["tool_calls"] = message["tool_calls"]

        return result

    # ------------------------------------------------------------------
    # vLLM Python API args
    # ------------------------------------------------------------------

    def _build_cli_args(self, input_file: str = "/dev/null", output_file: str = "/dev/null") -> list[str]:
        """Build CLI-style arg list for vLLM's ``make_arg_parser``."""
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
        return cli

    def _parse_vllm_args(self, input_file: str = "/dev/null", output_file: str = "/dev/null"):
        """Parse CLI args through vLLM's own argument parser to get a complete Namespace."""
        from vllm.entrypoints.openai.run_batch import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser(description="vLLM batch runner")
        parser = make_arg_parser(parser)
        return parser.parse_args(self._build_cli_args(input_file, output_file))

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def start(self) -> AsyncIterator[None]:
        """Create the vLLM engine and hold it open for the duration."""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
        from vllm.usage.usage_lib import UsageContext

        args = self._parse_vllm_args()
        engine_args = AsyncEngineArgs.from_cli_args(args)

        logger.info(
            "VLLMBatchClient starting engine model=%s tp=%s gpu_mem=%s",
            self.model_id,
            self.tensor_parallel_size,
            self.gpu_memory_utilization,
        )
        async with build_async_engine_client_from_engine_args(
            engine_args,
            usage_context=UsageContext.OPENAI_BATCH_RUNNER,
        ) as engine_client:
            self._engine_client = engine_client
            try:
                yield
            finally:
                self._engine_client = None

    def _ensure_engine(self):
        if self._engine_client is None:
            raise RuntimeError(
                "VLLMBatchClient requires `async with client.start():` before calling submit_and_wait."
            )
        return self._engine_client

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _build_cache_key_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Build a dict for cache key generation from a BatchRequest item."""
        cache_request: dict[str, Any] = {
            "model": request.get("model", self.model_id),
            "messages": request["messages"],
        }
        cache_request.update({k: v for k, v in request.get("lm_kwargs", {}).items() if k != "api_key"})
        return cache_request

    def _check_cache(
        self, requests: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """Partition requests into cached and uncached."""
        cached_by_id: dict[str, dict[str, Any]] = {}
        uncached: list[dict[str, Any]] = []

        for request in requests:
            cache_req = self._build_cache_key_request(request)
            cached_value = self.cache.get(cache_req)
            if isinstance(cached_value, dict) and ("response" in cached_value or "error" in cached_value):
                result = dict(cached_value)
                result["custom_id"] = request["id"]
                cached_by_id[request["id"]] = result
            else:
                uncached.append(request)

        return cached_by_id, uncached

    def _cache_results(
        self, requests: list[dict[str, Any]], raw_results: list[dict[str, Any]]
    ) -> None:
        """Write mini-batch results into the cache."""
        for request, result in zip(requests, raw_results, strict=True):
            cache_req = self._build_cache_key_request(request)
            self.cache.put(cache_req, result)

    # ------------------------------------------------------------------
    # Mini-batching
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_batches(requests: list[dict[str, Any]], batch_size: Optional[int]) -> list[list[dict[str, Any]]]:
        if batch_size is None or batch_size <= 0 or len(requests) <= batch_size:
            return [requests]
        return [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]

    # ------------------------------------------------------------------
    # Run a single mini-batch through the engine
    # ------------------------------------------------------------------

    async def _run_mini_batch(
        self,
        engine_client: Any,
        requests: list[dict[str, Any]],
        batch_index: int,
        total_batches: int,
    ) -> list[dict[str, Any]]:
        """Write JSONL, call ``run_batch``, read output, return parsed rows."""
        from vllm.entrypoints.openai.run_batch import run_batch

        with NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            input_path = Path(f.name)
            for request in requests:
                body = self._build_request_body(request)
                line = {
                    "custom_id": request["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(line) + "\n")

        output_path = input_path.with_suffix(".output.jsonl")
        args = self._parse_vllm_args(str(input_path), str(output_path))

        # vLLM's EngineArgs.create_engine_config() transfers top-level CLI
        # values into nested config objects (e.g. --reasoning-parser →
        # structured_outputs_config.reasoning_parser).  The batch runner
        # passes the raw argparse Namespace to init_app_state which reads
        # the nested config, but never calls create_engine_config() on it.
        # Replicate the transfers the serving layer needs here.
        so_cfg = getattr(args, "structured_outputs_config", None)
        if so_cfg is not None:
            for attr in ("reasoning_parser", "reasoning_parser_plugin"):
                val = getattr(args, attr, None)
                if val and hasattr(so_cfg, attr):
                    setattr(so_cfg, attr, val)

        logger.info(
            "VLLMBatchClient mini-batch %d/%d: requests=%d input=%s",
            batch_index + 1,
            total_batches,
            len(requests),
            input_path,
        )

        start = time.time()
        try:
            await run_batch(engine_client, args)

            raw_rows_by_id: dict[str, dict[str, Any]] = {}
            with output_path.open() as of:
                for raw_line in of:
                    stripped = raw_line.strip()
                    if stripped:
                        row = json.loads(stripped)
                        custom_id = row.get("custom_id", "")
                        raw_rows_by_id[custom_id] = row
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        duration_s = time.time() - start

        # Transform rows in request order, matched by custom_id
        output_rows: list[dict[str, Any]] = []
        failures = 0
        reasoning_rows = 0
        prompt_tokens = 0
        output_tokens = 0

        for request in requests:
            row = raw_rows_by_id.get(request["id"], {})
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

        logger.info(
            "VLLMBatchClient mini-batch %d/%d complete: total=%d failures=%d reasoning=%d "
            "prompt_tokens=%d output_tokens=%d duration_s=%.1f",
            batch_index + 1,
            total_batches,
            len(output_rows),
            failures,
            reasoning_rows,
            prompt_tokens,
            output_tokens,
            duration_s,
        )

        return output_rows

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def submit_and_wait(
        self,
        batch_request: BatchRequest,
        show_progress: bool = True,
    ) -> BatchReponse:
        del show_progress
        all_requests = batch_request["requests"]
        batch_id = str(uuid.uuid4())

        # Step 1: Check cache
        if self.cache is not None:
            cached_by_id, uncached_requests = self._check_cache(all_requests)
        else:
            cached_by_id = {}
            uncached_requests = list(all_requests)

        logger.info(
            "VLLMBatchClient submit_and_wait: batch_id=%s total=%d cached=%d uncached=%d batch_size=%s",
            batch_id,
            len(all_requests),
            len(cached_by_id),
            len(uncached_requests),
            self.batch_size,
        )

        if self.status_callback is not None:
            self.status_callback(
                batch_id,
                "running",
                None,
                {"provider": self.provider, "num_requests": len(all_requests), "num_cached": len(cached_by_id)},
            )

        # Step 2: Early return if all cached
        if not uncached_requests:
            ordered = [cached_by_id[r["id"]] for r in all_requests]
            return BatchReponse(
                batch_id=batch_id,
                status="completed",
                results=ordered,
                errors=None,
                raw_response={"source": "cache"},
            )

        # Step 3: Mini-batch
        mini_batches = self._split_into_batches(uncached_requests, self.batch_size)

        # Step 4: Run each mini-batch
        engine_client = self._ensure_engine()
        new_results_by_id: dict[str, dict[str, Any]] = {}

        for i, mb_requests in enumerate(mini_batches):
            mb_results = await self._run_mini_batch(engine_client, mb_requests, i, len(mini_batches))

            # Cache immediately after each mini-batch
            for req, result in zip(mb_requests, mb_results, strict=True):
                result_with_id = {**result, "custom_id": req["id"]}
                new_results_by_id[req["id"]] = result_with_id
                if self.cache is not None:
                    cache_req = self._build_cache_key_request(req)
                    self.cache.put(cache_req, result_with_id)

        # Step 5: Stitch together in original order
        merged_results: list[dict[str, Any]] = []
        for request in all_requests:
            rid = request["id"]
            raw = cached_by_id.get(rid) or new_results_by_id.get(rid)
            if raw is not None:
                merged_results.append(raw)

        return BatchReponse(
            batch_id=batch_id,
            status="completed",
            results=merged_results,
            errors=None,
            raw_response={"source": "vllm-python-api"},
        )

    # ------------------------------------------------------------------
    # Unused but required by BatchClient interface
    # ------------------------------------------------------------------

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        raise NotImplementedError("Use submit_and_wait directly")

    async def _get_status_impl(self, batch_id: str):
        return "completed", 100

    async def get_results(self, batch_id: str) -> BatchReponse:
        raise NotImplementedError("Use submit_and_wait directly")

    async def cancel(self, batch_id: str) -> bool:
        return False
