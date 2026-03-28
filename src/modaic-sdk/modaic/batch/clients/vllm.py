from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

import dspy

from ..types import BatchReponse, BatchRequest, ResultItem
from ..vllm_job import LocalVLLMRunner
from .base import BatchClient, _extract_openai_compatible_message, _stringify_content

logger = logging.getLogger(__name__)


class VLLMBatchClient(BatchClient):
    provider: str = "vllm"

    def __init__(
        self,
        lm: dspy.LM,
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
        self.reasoning_parser = reasoning_parser or ""
        self.enforce_eager = enforce_eager
        self.enable_thinking = enable_thinking or bool(reasoning_parser)
        self.thinking_budget = thinking_budget
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self._responses_by_batch_id: dict[str, list[dict[str, Any]]] = {}
        self._runner: Optional[LocalVLLMRunner] = None

    def _build_request_body(self, request: dict[str, Any]) -> dict[str, Any]:
        body = {
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

    async def _ensure_runner(self) -> LocalVLLMRunner:
        if self._runner is None:
            logger.info(
                "Initializing LocalVLLMRunner model=%s reasoning_parser=%s enforce_eager=%s max_model_len=%s gpu_memory_utilization=%s tensor_parallel_size=%s",
                self.model_id,
                self.reasoning_parser,
                self.enforce_eager,
                self.max_model_len,
                self.gpu_memory_utilization,
                self.tensor_parallel_size,
            )
            self._runner = await asyncio.to_thread(
                LocalVLLMRunner,
                self.model_id,
                self.reasoning_parser,
                self.enforce_eager,
                self.max_model_len,
                self.gpu_memory_utilization,
                self.tensor_parallel_size,
            )
        return self._runner

    @asynccontextmanager
    async def start(self) -> AsyncIterator[None]:
        await self._ensure_runner()
        try:
            yield
        finally:
            if self._runner is not None:
                runner = self._runner
                self._runner = None
                await asyncio.to_thread(runner.close)

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        batch_id = str(uuid.uuid4())
        batch_response = await self._run_batch(batch_request, batch_id=batch_id)
        self._responses_by_batch_id[batch_id] = batch_response.results
        return batch_id

    async def _run_batch(self, batch_request: BatchRequest, batch_id: Optional[str] = None) -> BatchReponse:
        runner = await self._ensure_runner()
        resolved_batch_id = batch_id or str(uuid.uuid4())
        request_ids = [request["id"] for request in batch_request["requests"]]
        logger.info(
            "Submitting vLLM batch batch_id=%s requests=%d model=%s",
            resolved_batch_id,
            len(request_ids),
            self.model_id,
        )

        if self.status_callback is not None:
            self.status_callback(
                resolved_batch_id,
                "running",
                None,
                {"provider": self.provider, "num_requests": len(batch_request["requests"])},
            )

        rows, summary = await asyncio.to_thread(
            runner.generate,
            [list(request["messages"]) for request in batch_request["requests"]],
            temperature=self.lm.kwargs.get("temperature"),
            top_p=self.lm.kwargs.get("top_p"),
            top_k=self.lm.kwargs.get("top_k"),
            min_p=self.lm.kwargs.get("min_p"),
            max_tokens=self.lm.kwargs.get("max_tokens"),
            repetition_penalty=self.lm.kwargs.get("repetition_penalty"),
            enable_thinking=self.enable_thinking,
            thinking_budget=self.thinking_budget,
        )

        results = [{"custom_id": request_id, **row} for request_id, row in zip(request_ids, rows, strict=True)]
        logger.info(
            "vLLM run-batch complete: batch_id=%s total=%d failures=%d reasoning_rows=%d prompt_tokens=%d output_tokens=%d duration_s=%.1f",
            resolved_batch_id,
            summary["total"],
            summary["failures"],
            summary["reasoning_rows"],
            summary["prompt_tokens"],
            summary["output_tokens"],
            summary["duration_s"],
        )
        return BatchReponse(
            batch_id=resolved_batch_id,
            status="completed",
            results=results,
            errors=None,
            raw_response={"source": "vllm-run-batch"},
        )

    async def _get_status_impl(self, batch_id: str):
        if batch_id in self._responses_by_batch_id:
            return "completed", 100
        return "in_progress", 0

    async def get_results(self, batch_id: str) -> BatchReponse:
        if batch_id not in self._responses_by_batch_id:
            raise ValueError(f"Unknown vLLM batch id: {batch_id}")
        return BatchReponse(
            batch_id=batch_id,
            status="completed",
            results=self._responses_by_batch_id.pop(batch_id),
            errors=None,
            raw_response={"source": "vllm-run-batch"},
        )

    async def cancel(self, batch_id: str) -> bool:
        del batch_id
        return False

    async def submit_and_wait(
        self,
        batch_request: BatchRequest,
        show_progress: bool = True,
    ) -> BatchReponse:
        del show_progress
        return await self._run_batch(batch_request)


class VLLMBatchClient2(VLLMBatchClient):
    pass
