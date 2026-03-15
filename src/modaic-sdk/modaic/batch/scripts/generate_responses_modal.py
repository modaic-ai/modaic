# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "flashinfer-python",
#     "huggingface-hub",
#     "hf_transfer",
#     "hf-xet>= 1.1.7",
#     "torch",
#     "transformers",
#     "vllm>=0.17.0",
#     "wandb",
#     "redis",
# ]
#
# ///
"""Generate responses for parquet prompts using a local vLLM OpenAI-compatible server."""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Optional

from datasets import load_dataset
from huggingface_hub import get_token
from torch import cuda
from tqdm.auto import tqdm
import wandb

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_PORT = 8000
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
MAX_IN_FLIGHT_REQUESTS = 256
WANDB_PROGRESS_LOG_INTERVAL_SECONDS = 10.0
WANDB_PROGRESS_LOG_INTERVAL_COMPLETIONS = 100


@dataclass(frozen=True)
class GenerationConfig:
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    min_p: Optional[float]
    max_tokens: Optional[int]
    repetition_penalty: Optional[float]


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: Optional[str]
    entity: Optional[str]
    name: Optional[str]
    group: Optional[str]
    job_type: Optional[str]
    tags: tuple[str, ...]
    mode: Optional[str]


ResponsePayload = dict[str, Optional[str]]


class WandbTracker:
    def __init__(self, config: WandbConfig, run_config: dict[str, Any]):
        self._config = config
        self._run: Optional[wandb.sdk.wandb_run.Run] = None
        self._last_progress_log_time = 0.0
        self._last_progress_logged_completed = 0

        if not config.enabled:
            return

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
        self._run = wandb.init(**{key: value for key, value in init_kwargs.items() if value is not None})

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, data: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.log(data)

    def maybe_log_progress(self, *, completed: int, total: int, failures: int) -> None:
        if self._run is None:
            return

        now = time.time()
        completed_delta = completed - self._last_progress_logged_completed
        should_log = (
            completed == total
            or completed_delta >= WANDB_PROGRESS_LOG_INTERVAL_COMPLETIONS
            or now - self._last_progress_log_time >= WANDB_PROGRESS_LOG_INTERVAL_SECONDS
        )
        if not should_log:
            return

        self.log(
            {
                "progress/completed": completed,
                "progress/total": total,
                "progress/failures": failures,
                "progress/completion_rate": completed / total if total else 0.0,
            }
        )
        self._last_progress_log_time = now
        self._last_progress_logged_completed = completed

    def finish_success(self, summary: dict[str, Any]) -> None:
        if self._run is None:
            return
        for key, value in summary.items():
            self._run.summary[key] = value
        self._run.finish(exit_code=0)
        self._run = None

    def finish_failure(self, exc: BaseException) -> None:
        if self._run is None:
            return
        self._run.summary["status"] = "failed"
        self._run.summary["error_type"] = exc.__class__.__name__
        self._run.summary["error_message"] = str(exc)
        self._run.finish(exit_code=1)
        self._run = None


def _parse_wandb_tags(raw_tags: Optional[str]) -> tuple[str, ...]:
    if not raw_tags:
        return ()
    return tuple(tag.strip() for tag in raw_tags.split(",") if tag.strip())


def check_gpu_availability() -> int:
    if not cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)

    num_gpus = cuda.device_count()
    for i in range(num_gpus):
        gpu_name = cuda.get_device_name(i)
        gpu_memory = cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name} with {gpu_memory:.1f} GB memory")

    return num_gpus


def _wait_for_server(timeout_seconds: int = 1800) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("Timed out waiting for the vLLM server to become healthy")


def _post_chat_completion(request_body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _build_request_body(
    model_id: str,
    messages: list[dict[str, Any]],
    generation: GenerationConfig,
    enable_thinking: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "include_reasoning": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if generation.max_tokens is not None:
        body["max_completion_tokens"] = generation.max_tokens
    if generation.temperature is not None:
        body["temperature"] = generation.temperature
    if generation.top_p is not None:
        body["top_p"] = generation.top_p
    if generation.top_k is not None:
        body["top_k"] = generation.top_k
    if generation.min_p is not None:
        body["min_p"] = generation.min_p
    if generation.repetition_penalty is not None:
        body["repetition_penalty"] = generation.repetition_penalty
    return body


def _extract_output_payload(response_body: dict[str, Any]) -> dict[str, Optional[str]]:
    choices = response_body.get("choices") or []
    if not choices:
        return {"text": "", "reasoning_content": None, "error": None}

    message = choices[0].get("message") or {}
    return {
        "text": message.get("content") or "",
        "reasoning_content": message.get("reasoning"),
        "error": None,
    }


def _error_payload(error: str) -> ResponsePayload:
    return {"text": None, "reasoning_content": None, "error": error}


def _stringify_request_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        error_details = exc.reason or str(exc)
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        if body:
            return f"HTTP {exc.code}: {error_details}. {body}"
        return f"HTTP {exc.code}: {error_details}"
    return str(exc) or exc.__class__.__name__


def _run_generation_via_server(
    conversations: list[list[dict[str, Any]]],
    *,
    model_id: str,
    generation: GenerationConfig,
    enable_thinking: bool,
    enforce_eager: bool,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    reasoning_parser: Optional[str],
    tracker: Optional[WandbTracker],
    cache: Any = None,
) -> list[dict[str, Optional[str]]]:
    env = os.environ.copy()
    server_cmd = [
        "vllm",
        "serve",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if max_model_len is not None:
        server_cmd.extend(["--max-model-len", str(max_model_len)])
    if reasoning_parser:
        server_cmd.extend(["--reasoning-parser", reasoning_parser])
    if enforce_eager:
        server_cmd.append("--enforce-eager")

    logger.info("Starting vLLM server: %s", " ".join(server_cmd))
    server_start_time = time.time()
    server = subprocess.Popen(server_cmd, env=env)

    try:
        _wait_for_server()
        server_startup_seconds = time.time() - server_start_time
        logger.info("vLLM server is healthy; keeping up to %d requests in flight", MAX_IN_FLIGHT_REQUESTS)
        if tracker is not None:
            tracker.log({"timing/server_startup_seconds": server_startup_seconds})

        responses: list[ResponsePayload] = [_error_payload("request was never submitted") for _ in conversations]
        submitted = 0
        completed = 0
        failures = 0
        cache_hits = 0
        futures: dict[Future[dict[str, Any]], int] = {}
        # Map future -> request_body so we can cache the response on success
        future_request_bodies: dict[Future[dict[str, Any]], dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=min(MAX_IN_FLIGHT_REQUESTS, len(conversations))) as executor:
            with tqdm(total=len(conversations), desc="Generating responses") as progress:
                while submitted < len(conversations) or futures:
                    while submitted < len(conversations) and len(futures) < MAX_IN_FLIGHT_REQUESTS:
                        request_body = _build_request_body(
                            model_id=model_id,
                            messages=conversations[submitted],
                            generation=generation,
                            enable_thinking=enable_thinking,
                        )

                        # Check cache before submitting to vLLM
                        if cache is not None:
                            try:
                                cached = cache.get(request_body)
                            except Exception:
                                cached = None
                            if cached is not None:
                                responses[submitted] = cached
                                cache_hits += 1
                                completed += 1
                                if tracker is not None:
                                    tracker.maybe_log_progress(completed=completed, total=len(conversations), failures=failures)
                                progress.update(1)
                                submitted += 1
                                continue

                        future = executor.submit(_post_chat_completion, request_body)
                        futures[future] = submitted
                        future_request_bodies[future] = request_body
                        submitted += 1

                    if not futures:
                        break

                    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        index = futures.pop(future)
                        req_body = future_request_bodies.pop(future)
                        try:
                            payload = _extract_output_payload(future.result())
                            responses[index] = payload
                            if cache is not None and payload.get("error") is None:
                                try:
                                    cache.set(req_body, payload)
                                except Exception:
                                    logger.debug("Failed to write cache for request %d", index)
                        except Exception as exc:
                            error_message = _stringify_request_error(exc)
                            logger.warning("Request %d failed: %s", index, error_message)
                            responses[index] = _error_payload(error_message)
                            failures += 1
                        completed += 1
                        if tracker is not None:
                            tracker.maybe_log_progress(completed=completed, total=len(conversations), failures=failures)
                        progress.update(1)

        if cache_hits:
            logger.info("Cache hits: %d / %d requests", cache_hits, len(conversations))

        return responses
    finally:
        logger.info("Stopping vLLM server")
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=30)


def main(
    src_dataset_path: str,
    output_dataset_path: str,
    model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    reasoning_parser: Optional[str] = None,
    enforce_eager: bool = False,
    messages_column: str = "messages",
    prompt_column: Optional[str] = None,
    output_column: str = "outputs",
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
    wandb_enabled: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_tags: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_job_type: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    use_cache: bool = False,
) -> None:
    num_gpus = check_gpu_availability()
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus
        logger.info(f"Auto-detected {num_gpus} GPU(s), using tensor_parallel_size={tensor_parallel_size}")
    elif tensor_parallel_size > num_gpus:
        logger.warning(f"Requested {tensor_parallel_size} GPUs but only {num_gpus} available")

    resolved_hf_token = hf_token or os.environ.get("HF_TOKEN") or get_token()
    if resolved_hf_token:
        os.environ["HF_TOKEN"] = resolved_hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = resolved_hf_token
        logger.info("Using Hugging Face token for model access")

    generation = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )
    tracker = WandbTracker(
        WandbConfig(
            enabled=wandb_enabled,
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            group=wandb_group,
            job_type=wandb_job_type,
            tags=_parse_wandb_tags(wandb_tags),
            mode=wandb_mode,
        ),
        run_config={
            "src_dataset_path": src_dataset_path,
            "output_dataset_path": output_dataset_path,
            "model_id": model_id,
            "reasoning_parser": reasoning_parser,
            "enforce_eager": enforce_eager,
            "messages_column": messages_column,
            "prompt_column": prompt_column,
            "output_column": output_column,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "skip_long_prompts": skip_long_prompts,
            "enable_thinking": enable_thinking,
            "max_samples": max_samples,
            "gpu_count": num_gpus,
        },
    )

    run_started_at = time.time()
    try:
        logger.info(f"Loading parquet dataset: {src_dataset_path}")
        dataset = load_dataset("parquet", data_files=src_dataset_path, split="train")

        if max_samples is not None and max_samples < len(dataset):
            logger.info(f"Limiting dataset to {max_samples} samples")
            dataset = dataset.select(range(max_samples))

        total_examples = len(dataset)
        logger.info(f"Dataset loaded with {total_examples:,} examples")
        if tracker.enabled:
            tracker.log({"dataset/rows": total_examples})

        if prompt_column:
            if prompt_column not in dataset.column_names:
                logger.error(f"Column '{prompt_column}' not found. Available columns: {dataset.column_names}")
                sys.exit(1)
            logger.info(f"Using prompt column mode with column: '{prompt_column}'")
            use_messages = False
        else:
            if messages_column not in dataset.column_names:
                logger.error(f"Column '{messages_column}' not found. Available columns: {dataset.column_names}")
                sys.exit(1)
            logger.info(f"Using messages column mode with column: '{messages_column}'")
            use_messages = True

        if skip_long_prompts:
            logger.info("Prompt length pre-filtering remains disabled; server-side limits will apply.")

        conversations: list[list[dict[str, Any]]] = []
        for example in tqdm(dataset, desc="Processing prompts"):
            if use_messages:
                conversations.append(example[messages_column])
            else:
                conversations.append([{"role": "user", "content": example[prompt_column]}])

        if not conversations:
            logger.error("No prompts to process")
            sys.exit(1)

        request_cache = None
        if use_cache:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from redis_cache import RedisRequestCache

                request_cache = RedisRequestCache()
                logger.info("Redis request cache enabled")
            except Exception as exc:
                logger.warning("Failed to connect to Redis cache, proceeding without cache: %s", exc)

        responses = _run_generation_via_server(
            conversations,
            model_id=model_id,
            generation=generation,
            enable_thinking=enable_thinking,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            reasoning_parser=reasoning_parser,
            tracker=tracker,
            cache=request_cache,
        )

        logger.info(f"Writing responses to column '{output_column}'")
        output_responses = [
            {"text": response["text"], "reasoning_content": response["reasoning_content"]} for response in responses
        ]
        output_error_column = f"{output_column}_error"
        output_errors = [response["error"] for response in responses]
        dataset = dataset.add_column(output_column, output_responses)
        dataset = dataset.add_column(output_error_column, output_errors)

        logger.info(f"Writing parquet dataset to: {output_dataset_path}")
        dataset.to_parquet(output_dataset_path)

        failed_rows = sum(1 for error in output_errors if error)
        empty_text_rows = sum(1 for response in responses if not response["error"] and not (response["text"] or ""))
        reasoning_rows = sum(1 for response in responses if response["reasoning_content"])
        summary = {
            "status": "completed",
            "dataset_rows": total_examples,
            "failed_rows": failed_rows,
            "empty_text_rows": empty_text_rows,
            "reasoning_rows": reasoning_rows,
            "generation_seconds": time.time() - run_started_at,
            "output_dataset_path": output_dataset_path,
        }
        if tracker.enabled:
            tracker.log(
                {
                    "results/failed_rows": failed_rows,
                    "results/empty_text_rows": empty_text_rows,
                    "results/reasoning_rows": reasoning_rows,
                    "timing/total_seconds": summary["generation_seconds"],
                }
            )
            tracker.finish_success(summary)
        logger.info("Generation complete")
    except BaseException as exc:
        tracker.finish_failure(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses for parquet dataset prompts using a local vLLM server")
    parser.add_argument("src_dataset_path", help="Input parquet dataset path")
    parser.add_argument("output_dataset_path", help="Output parquet dataset path")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Model to use for generation")
    parser.add_argument("--messages-column", type=str, default="messages", help="Column containing chat messages")
    parser.add_argument("--reasoning-parser", type=str, help="vLLM reasoning parser to use for supported models")
    parser.add_argument("--enforce-eager", action="store_true", default=False, help="Disable torch.compile and CUDA graph capture in vLLM")
    parser.add_argument("--prompt-column", type=str, help="Column containing plain text prompts")
    parser.add_argument("--output-column", type=str, default="outputs", help="Column name for generated responses")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, help="Top-k sampling parameter")
    parser.add_argument("--min-p", type=float, help="Minimum probability threshold")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--repetition-penalty", type=float, help="Repetition penalty")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilization factor")
    parser.add_argument("--max-model-len", type=int, help="Maximum model context length")
    parser.add_argument("--tensor-parallel-size", type=int, help="Number of GPUs to use")
    parser.add_argument("--enable-thinking", action="store_true", default=False, help="Enable model thinking/reasoning when supported")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument("--skip-long-prompts", action="store_true", default=True, help="Keep CLI compatibility; prompt pre-filtering is not used in chat mode")
    parser.add_argument("--no-skip-long-prompts", dest="skip_long_prompts", action="store_false", help="Keep CLI compatibility; prompt pre-filtering is not used in chat mode")
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true", default=False, help="Enable Weights & Biases tracking")
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project")
    parser.add_argument("--wandb-entity", type=str, help="Weights & Biases entity")
    parser.add_argument("--wandb-name", type=str, help="Weights & Biases run name")
    parser.add_argument("--wandb-tags", type=str, help="Comma-separated Weights & Biases tags")
    parser.add_argument("--wandb-group", type=str, help="Weights & Biases run group")
    parser.add_argument("--wandb-job-type", type=str, help="Weights & Biases job type")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), help="Weights & Biases mode")
    parser.add_argument("--use-cache", action="store_true", default=False, help="Enable Redis request caching to deduplicate vLLM requests")

    args = parser.parse_args()
    main(
        src_dataset_path=args.src_dataset_path,
        output_dataset_path=args.output_dataset_path,
        model_id=args.model_id,
        reasoning_parser=args.reasoning_parser,
        enforce_eager=args.enforce_eager,
        messages_column=args.messages_column,
        prompt_column=args.prompt_column,
        output_column=args.output_column,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        skip_long_prompts=args.skip_long_prompts,
        enable_thinking=args.enable_thinking,
        max_samples=args.max_samples,
        hf_token=args.hf_token,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_mode=args.wandb_mode,
        use_cache=args.use_cache,
    )
