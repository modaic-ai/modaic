from __future__ import annotations

import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Optional

import dspy
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import modal
    import pandas as pd
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch requires Modal and pandas for Modal batch jobs. Install them with `uv add "modaic[modal]"`.'
    ) from exc

from .clients import BatchClient
from .modal_job import ResponseGenerator, app, cache_volume
from .storage import get_modal_batch_parquet_paths
from .types import BatchReponse, BatchRequest, ResultItem


class _ModalBatchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")


class ModalBatchClient(BatchClient):
    provider: str = "modal"

    def __init__(
        self,
        lm: dspy.LM,
        poll_interval: float = 30.0,
        max_poll_time: str = "24h",
        status_callback=None,
        gpu: str = "A100:2",
        reasoning_parser: Optional[str] = None,
        enforce_eager: bool = False,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        hf_token: Optional[str] = None,
    ):
        model = getattr(lm, "model", None)
        if not isinstance(model, str) or not model.startswith("huggingface/"):
            raise ValueError("ModalBatchClient requires a dspy.LM with model='huggingface/<repo_path>'")

        super().__init__(
            api_key=None,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=status_callback,
        )
        self.lm = lm
        self.model_id = model.removeprefix("huggingface/")
        self.gpu = gpu
        self.reasoning_parser = reasoning_parser
        self.enforce_eager = enforce_eager
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.hf_token = hf_token if hf_token is not None else _ModalBatchSettings().hf_token
        self.temperature = lm.kwargs.get("temperature")
        self.top_p = lm.kwargs.get("top_p")
        self.top_k = lm.kwargs.get("top_k")
        self.min_p = lm.kwargs.get("min_p")
        self.max_tokens = lm.kwargs.get("max_tokens")
        self.repetition_penalty = lm.kwargs.get("repetition_penalty")
        self._responses_by_batch_id: dict[str, list[dict[str, Any]]] = {}

    def format(self, batch_request: BatchRequest) -> list[list[dict[str, Any]]]:
        return [list(request["messages"]) for request in batch_request["requests"]]

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        error = raw_result.get("error")
        if isinstance(error, str) and error.strip():
            raise ValueError(error.strip())

        response = raw_result.get("response")
        if not isinstance(response, dict):
            raise ValueError("Modal batch result is missing a response payload")

        text = response.get("text")
        if not isinstance(text, str):
            raise ValueError("Modal batch result is missing response text")

        result: ResultItem = {"text": text}
        reasoning_content = response.get("reasoning_content")
        if reasoning_content is not None:
            result["reasoning_content"] = str(reasoning_content)
        return result

    def _notify_status(self, batch_id: Optional[str], status: str, progress: Optional[int], metadata: dict[str, Any]) -> None:
        if self.status_callback is not None:
            self.status_callback(batch_id, status, progress, metadata)

    @asynccontextmanager
    async def start(self) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            stack.enter_context(modal.enable_output())
            await stack.enter_async_context(app.run())
            yield

    async def _run_modal_job(self, batch_id: str, batch_request: BatchRequest) -> list[dict[str, Any]]:
        input_path, output_path = get_modal_batch_parquet_paths(batch_id)
        remote_input_path = f"/cache/{input_path.name}"
        remote_output_path = f"/cache/{output_path.name}"
        request_ids = [request["id"] for request in batch_request["requests"]]
        messages_batches = self.format(batch_request)

        pd.DataFrame({"messages": messages_batches}).to_parquet(input_path, index=False)

        async with cache_volume.batch_upload(force=True) as batch:
            batch.put_file(str(input_path), f"/{input_path.name}")

        generator = ResponseGenerator.with_options(gpu=self.gpu)()
        await generator.run_generate_responses.remote.aio(
            src_dataset_path=remote_input_path,
            output_dataset_path=remote_output_path,
            model_id=self.model_id,
            reasoning_parser=self.reasoning_parser,
            enforce_eager=self.enforce_eager,
            messages_column="messages",
            output_column="response",
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            enable_thinking=self.enable_thinking,
            thinking_budget=self.thinking_budget,
            hf_token=self.hf_token,
        )

        output_path.write_bytes(b"".join([chunk async for chunk in cache_volume.read_file.aio(output_path.name)]))
        records = pd.read_parquet(output_path).to_dict(orient="records")

        return [
            {
                "custom_id": request_id,
                "response": record.get("response"),
                "error": record.get("response_error"),
            }
            for request_id, record in zip(request_ids, records, strict=True)
        ]

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        batch_id = str(uuid.uuid4())
        responses = await self._run_modal_job(batch_id, batch_request)
        self._responses_by_batch_id[batch_id] = responses
        return batch_id

    async def submit(self, batch_request: BatchRequest) -> str:
        self.num_requests = len(batch_request["requests"])
        self.start_time = time.time()
        metadata = {"provider": self.provider, "num_requests": self.num_requests}
        self._notify_status(None, "submitting", 0, metadata)
        batch_id = await self._submit_batch_request(batch_request)
        self._notify_status(batch_id, "completed", 100, metadata)
        return batch_id

    async def _get_status_impl(self, batch_id: str):
        if batch_id in self._responses_by_batch_id:
            return "completed", 100
        return "in_progress", 0

    async def get_results(self, batch_id: str) -> BatchReponse:
        if batch_id not in self._responses_by_batch_id:
            raise ValueError(f"Unknown Modal batch id: {batch_id}")
        return BatchReponse(
            batch_id=batch_id,
            status="completed",
            results=self._responses_by_batch_id.pop(batch_id),
            errors=None,
            raw_response={"source": "modal"},
        )

    async def cancel(self, batch_id: str) -> bool:
        return False

    async def submit_and_wait(
        self,
        batch_request: BatchRequest,
        show_progress: bool = True,
    ) -> BatchReponse:
        del show_progress
        batch_id = await self.submit(batch_request)
        return await self.get_results(batch_id)
