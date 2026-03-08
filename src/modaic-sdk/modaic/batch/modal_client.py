from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import dspy
import modal
import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .clients import BatchClient
from .modal_job import ResponseGenerator, app, cache_volume
from .types import BatchReponse, BatchRequest, ResultItem

DEFAULT_BATCH_ID = "default_batch_id"


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

    async def _run_modal_job(self, messages_batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="modaic-modal-batch-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.parquet"
            output_path = tmp_path / "output.parquet"
            input_filename = f"modal-batch-input-{uuid.uuid4().hex}.parquet"
            output_filename = f"modal-batch-output-{uuid.uuid4().hex}.parquet"
            remote_input_path = f"/cache/{input_filename}"
            remote_output_path = f"/cache/{output_filename}"

            pd.DataFrame({"messages": messages_batches}).to_parquet(input_path, index=False)

            async with cache_volume.batch_upload(force=True) as batch:
                batch.put_file(str(input_path), f"/{input_filename}")

            with modal.enable_output():
                async with app.run():
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
                        hf_token=self.hf_token,
                    )

            output_path.write_bytes(b"".join([chunk async for chunk in cache_volume.read_file.aio(output_filename)]))
            records = pd.read_parquet(output_path).to_dict(orient="records")

        return [
            {
                "custom_id": f"request-{index}",
                "response": record.get("response"),
            }
            for index, record in enumerate(records)
        ]

    async def _submit_batch_request(self, batch_request: BatchRequest) -> str:
        responses = await self._run_modal_job(self.format(batch_request))
        self._responses_by_batch_id[DEFAULT_BATCH_ID] = responses
        return DEFAULT_BATCH_ID

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
