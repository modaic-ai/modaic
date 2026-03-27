from __future__ import annotations

import gzip
import json
import logging
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import dspy
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import modal
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'modaic.batch requires Modal for Modal batch jobs. Install it with `uv add "modaic[modal]"`.'
    ) from exc

from .clients import BatchClient
from .modal_job import (
    INPUT_FILENAME,
    OUTPUT_FILENAME,
    ResponseGenerator,
    _UNSET_INT,
    app,
    cache_volume,
)
from .storage import get_modal_batch_jsonl_gz_paths
from .types import BatchReponse, BatchRequest, ResultItem

logger = logging.getLogger(__name__)


class _ModalBatchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    tags: tuple[str, ...] = ()
    mode: Optional[str] = None


class WandbTracker:
    def __init__(self, config: WandbConfig, run_config: dict[str, Any]):
        self._run = None
        if not config.enabled:
            return
        try:
            import wandb

            init_kwargs: dict[str, Any] = {
                "project": config.project,
                "entity": config.entity,
                "name": config.name,
                "group": config.group,
                "job_type": config.job_type,
                "tags": list(config.tags) or None,
                "mode": config.mode,
                "config": run_config,
            }
            self._run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
        except Exception:
            logger.warning("Failed to initialize wandb, continuing without tracking", exc_info=True)

    def log(self, data: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.log(data)

    def finish_success(self, summary: dict[str, Any]) -> None:
        if self._run is None:
            return
        for k, v in summary.items():
            self._run.summary[k] = v
        self._run.finish(exit_code=0)
        self._run = None

    def finish_failure(self, exc: BaseException) -> None:
        if self._run is None:
            return
        self._run.summary["status"] = "failed"
        self._run.summary["error"] = str(exc)
        self._run.finish(exit_code=1)
        self._run = None


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
        hf_transfer: bool = False,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: Optional[int] = None,
        language_model_only: bool = False,
        wandb_config: Optional[WandbConfig] = None,
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
        self.reasoning_parser = reasoning_parser or ""
        self.enforce_eager = enforce_eager
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.hf_token = hf_token if hf_token is not None else _ModalBatchSettings().hf_token
        self.hf_transfer = hf_transfer
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.language_model_only = language_model_only
        self.wandb_config = wandb_config or WandbConfig()
        self.temperature = lm.kwargs.get("temperature")
        self.top_p = lm.kwargs.get("top_p")
        self.top_k = lm.kwargs.get("top_k")
        self.min_p = lm.kwargs.get("min_p")
        self.max_tokens = lm.kwargs.get("max_tokens")
        self.repetition_penalty = lm.kwargs.get("repetition_penalty")
        self._responses_by_batch_id: dict[str, list[dict[str, Any]]] = {}

    def format(self, batch_request: BatchRequest) -> list[list[dict[str, Any]]]:
        return [list(request["messages"]) for request in batch_request["requests"]]

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

    def create_run_batch_jsonl_gz(self, batch_request: BatchRequest, path: str) -> None:
        with gzip.open(path, "wt") as f:
            for request in batch_request["requests"]:
                f.write(
                    json.dumps(
                        {
                            "custom_id": request["id"],
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": self._build_request_body(request),
                        }
                    )
                    + "\n"
                )

    def parse(self, raw_result: dict[str, Any]) -> ResultItem:
        error = raw_result.get("error")
        if isinstance(error, str) and error.strip():
            raise ValueError(error.strip())

        response = raw_result.get("response")
        logger.info("[DEBUG] parse() raw_result keys=%s, response_type=%s, response=%s",
                     list(raw_result.keys()), type(response).__name__,
                     repr(response)[:300] if response else repr(response))
        if not isinstance(response, dict):
            raise ValueError("Modal batch result is missing a response payload")

        text = response.get("text")
        logger.info("[DEBUG] parse() text_type=%s, text_preview=%s",
                     type(text).__name__, repr(text[:200]) if isinstance(text, str) else repr(text))
        if not isinstance(text, str):
            raise ValueError(f"Modal batch result is missing response text (got {type(text).__name__}: {text!r})")

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

    def _resolve_hf_dataset_id(self) -> str:
        from huggingface_hub import whoami

        user_info = whoami(token=self.hf_token)
        username = user_info["name"]
        return f"{username}/modal_batch_job"

    async def _run_modal_job(self, batch_id: str, batch_request: BatchRequest) -> list[dict[str, Any]]:
        request_ids = [request["id"] for request in batch_request["requests"]]

        env: dict[str, str] = {}
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token

        run_config = {
            "model_id": self.model_id,
            "gpu": self.gpu,
            "reasoning_parser": self.reasoning_parser,
            "enforce_eager": self.enforce_eager,
            "enable_thinking": self.enable_thinking,
            "thinking_budget": self.thinking_budget,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "num_requests": len(batch_request["requests"]),
            "hf_transfer": self.hf_transfer,
        }
        tracker = WandbTracker(self.wandb_config, run_config)

        remote_input_path = f"{'/cache'}/{INPUT_FILENAME}"
        input_path, _ = get_modal_batch_jsonl_gz_paths(batch_id)
        self.create_run_batch_jsonl_gz(batch_request, str(input_path))
        async with cache_volume.batch_upload(force=True) as batch:
            batch.put_file(str(input_path), f"/{INPUT_FILENAME}")
        logger.info("Uploaded input jsonl.gz to Modal volume")

        # --- Run generation ---
        generator = ResponseGenerator.with_options(gpu=self.gpu, env=env)(
            model_id=self.model_id,
            reasoning_parser=self.reasoning_parser,
            enforce_eager=self.enforce_eager,
            gpu_memory_utilization_pct=int(self.gpu_memory_utilization * 100),
            max_model_len=self.max_model_len if self.max_model_len is not None else _UNSET_INT,
            tensor_parallel_size=self.tensor_parallel_size if self.tensor_parallel_size is not None else _UNSET_INT,
        )

        remote_output_path = f"{'/cache'}/{OUTPUT_FILENAME}"

        start_time = time.time()
        try:
            summary = await generator.generate.remote.aio(
                input_path=remote_input_path,
                output_path=remote_output_path,
            )
            duration_s = time.time() - start_time

            logger.info(
                "Generation complete: %d results (%d failed, %d with reasoning) in %.1fs",
                summary["total"], summary["failures"], summary["reasoning_rows"], duration_s,
            )

            tracker.log({
                "timing/generation_seconds": duration_s,
                "results/total": summary["total"],
                "results/failed": summary["failures"],
                "results/reasoning_rows": summary["reasoning_rows"],
                "tokens/prompt_total": summary["prompt_tokens"],
                "tokens/output_total": summary["output_tokens"],
                "tokens/throughput": summary["output_tokens"] / summary["duration_s"] if summary["duration_s"] > 0 else 0,
            })
            tracker.finish_success({
                "status": "completed",
                "total": summary["total"],
                "failed": summary["failures"],
                "generation_seconds": duration_s,
            })
        except BaseException as exc:
            tracker.finish_failure(exc)
            raise

        _, output_path = get_modal_batch_jsonl_gz_paths(batch_id)
        output_path.write_bytes(
            b"".join([chunk async for chunk in cache_volume.read_file.aio(OUTPUT_FILENAME)])
        )
        records: list[dict[str, Any]] = []
        with gzip.open(output_path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        return [
            {
                "custom_id": record.get("custom_id", request_id),
                "response": record.get("response"),
                "error": record.get("error"),
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
