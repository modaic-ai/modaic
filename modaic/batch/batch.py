import asyncio
import logging
import re
from typing import Any, Callable, Optional, Tuple

import dspy
from litellm import get_llm_provider

from .clients import (
    AnthropicBatchClient,
    AzureBatchClient,
    BatchClient,
    FireworksBatchClient,
    OpenAIBatchClient,
    TogetherBatchClient,
    VertexAIBatchClient,
)
from .types import ABatchResult, BatchReponse, BatchRequest, BatchRequestItem, FailedPrediction, ResultItem

logger = logging.getLogger(__name__)

"""
Azure format/ OpenAI
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "task-0", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}

Together format
{"custom_id": "request-1", "body": {"model": "deepseek-ai/DeepSeek-V3", "messages": [{"role": "user", "content": "Hello, world!"}], "max_tokens": 200}}
{"custom_id": "request-2", "body": {"model": "deepseek-ai/DeepSeek-V3", "messages": [{"role": "user", "content": "Explain quantum computing"}], "max_tokens": 200}}

Fireworks format (model is specified when you submit a job)
{"custom_id": "request-1", "body": {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}], "max_tokens": 100}}
{"custom_id": "request-2", "body": {"messages": [{"role": "user", "content": "Explain quantum computing"}], "temperature": 0.7}}
{"custom_id": "request-3", "body": {"messages": [{"role": "user", "content": "Tell me a joke"}]}}

Vertex AI
{"request":{"contents": [{"role": "user", "parts": [{"text": "List objects in this image."}, {"file_data": {"file_uri": "gs://cloud-samples-data/generative-ai/image/office-desk.jpeg", "mime_type": "image/jpeg"}}]}],"generationConfig":{"temperature": 0.4}}}


"""

CLIENTS: dict[str, type[BatchClient]] = {
    "openai": OpenAIBatchClient,
    "anthropic": AnthropicBatchClient,
    "together_ai": TogetherBatchClient,
    # "vertex_ai": VertexAIBatchClient, # noqa: ERA001
    "azure": AzureBatchClient,
    "fireworks_ai": FireworksBatchClient,
}


class BatchAdapter:
    """
    Base class for batch adapters that handle DSPy adapter logic for batch processing.

    BatchAdapter is responsible for:
    - Creating BatchRequests from predictors and inputs (format)
    - Parsing ResultItems back into DSPy Predictions (parse)
    - Orchestrating the batch process via __call__

    This separates the DSPy adapter logic from the BatchClient, which only handles
    the API interactions (submitting jobs, polling status, fetching results).

    Subclasses can override format/parse to support adapters that need multiple passes
    or have special formatting requirements.
    """

    adapter: dspy.Adapter

    def format(self, predictor: dspy.Predict, inputs_list: list[dict]) -> BatchRequest:
        """
        Create a BatchRequest from a predictor and list of inputs.

        Args:
            predictor: The dspy.Predict instance to use.
            inputs_list: List of input dictionaries for the predictor.

        Returns:
            A BatchRequest ready to be submitted to a BatchClient.
        """
        requests = []

        # Track global model and lm_kwargs (None if they differ across requests)
        global_model = None
        global_lm_kwargs = None

        for i, inputs in enumerate(inputs_list):
            lm, config, signature, demos, kwargs = predictor._forward_preprocess(**inputs)
            with dspy.settings.context(send_stream=None):
                processed_signature = self.adapter._call_preprocess(lm, config, signature, inputs)
                messages = self.adapter.format(processed_signature, demos, inputs)

            model_name = get_llm_provider(lm.model)[0]

            requests.append(
                BatchRequestItem(
                    id=f"request-{i}",
                    model=model_name,
                    messages=messages,
                    lm_kwargs=config,
                )
            )

            if i == 0:
                global_model = model_name
                global_lm_kwargs = config
            else:
                if global_model != model_name:
                    global_model = None
                if global_lm_kwargs != config:
                    global_lm_kwargs = None

        return BatchRequest(requests=requests, model=global_model, lm_kwargs=global_lm_kwargs)

    def parse(
        self,
        predictor: dspy.Predict,
        inputs_list: list[dict],
        results: list[ResultItem | None],
    ) -> list[dspy.Prediction | FailedPrediction]:
        """
        Parse ResultItems back into DSPy Predictions.

        Args:
            predictor: The predictor used for the batch request.
            inputs_list: The original list of inputs (must be in same order as results).
            results: The ResultItem list, already sorted to match inputs_list order.
                    May contain None for items that failed at the API level.

        Returns:
            A list of dspy.Prediction objects or FailedPrediction for failures.
        """
        predictions: list[dspy.Prediction | FailedPrediction] = []

        for i, (inputs, result) in enumerate(zip(inputs_list, results, strict=True)):
            if result is None:
                predictions.append(FailedPrediction(error="API level failure or parse error", index=i))
                continue

            # Recompute the signature processing for this input
            lm, config, signature, demos, kwargs = predictor._forward_preprocess(**inputs)
            processed_signature = self.adapter._call_preprocess(lm, config, signature, inputs)

            # Build output in the format _call_postprocess expects
            # Ensure delimiters are on their own lines as ChatAdapter expects
            text = result["text"]
            text = re.sub(r"([^\n])(\[\[\s*##)", r"\1\n\2", text)
            output: dict[str, Any] = {"text": text}
            if "logprobs" in result:
                output["logprobs"] = result["logprobs"]
            if "tool_calls" in result:
                output["tool_calls"] = result["tool_calls"]

            # Parse using the adapter
            try:
                parsed_outputs = self.adapter._call_postprocess(processed_signature, signature, [output], lm)
                if parsed_outputs:
                    predictions.append(dspy.Prediction(**parsed_outputs[0]))
                else:
                    predictions.append(FailedPrediction(error="empty output", index=i))
            except Exception as e:
                predictions.append(FailedPrediction(error=str(e), index=i))

        return predictions

    async def __call__(
        self,
        predictor: dspy.Predict,
        inputs: list[dict],
        batch_client: BatchClient,
        show_progress: bool = True,
    ) -> list[ABatchResult]:
        """
        Execute a batch job: format inputs, submit to client, and parse results.

        Args:
            predictor: The dspy.Predict instance to use for processing.
            inputs: List of input dictionaries for the predictor.
            batch_client: The BatchClient to use for submitting and polling.
            show_progress: Whether to show a progress display while waiting.

        Returns:
            A list of ABatchResult objects, one per input.

        Raises:
            TimeoutError: If the job doesn't complete within max_poll_time.
            RuntimeError: If the batch job fails.
        """
        # Format inputs into a BatchRequest
        batch_request = self.format(predictor, inputs)

        # Submit and wait for results
        batch_result = await batch_client.submit_and_wait(batch_request, show_progress=show_progress)

        # Combine results and errors from the API
        all_raw_results = list(batch_result.results)
        if batch_result.errors:
            all_raw_results.extend(batch_result.errors)

        # Sort raw results by custom_id to match original inputs order
        # custom_id format is "request-{i}" where i is the index
        def get_request_index(raw_res: dict) -> int:
            custom_id = raw_res.get("custom_id")
            if custom_id and isinstance(custom_id, str) and custom_id.startswith("request-"):
                try:
                    return int(custom_id.split("-")[-1])
                except (ValueError, IndexError):
                    pass
            return 0

        sorted_raw_results = sorted(
            all_raw_results,
            key=get_request_index,
        )

        # Convert raw results to ResultItems, handling parse failures per-item
        result_items: list[ResultItem | None] = []
        for raw_result in sorted_raw_results:
            try:
                result_items.append(batch_client.parse(raw_result))
            except Exception:
                # If a single item fails to parse, we track it as None
                # so the rest of the batch can still be processed.
                result_items.append(None)

        # Final safety check: ensure the result count matches the input count
        if len(result_items) != len(inputs):
            raise RuntimeError(
                f"Batch result count mismatch: expected {len(inputs)}, "
                f"got {len(result_items)} ({len(batch_result.results)} successes, "
                f"{len(batch_result.errors or [])} errors from API)"
            )

        # Parse into predictions
        predictions = self.parse(predictor, inputs, result_items)

        # Map predictions back to ABatchResult with their formatted messages
        return [
            ABatchResult(prediction=pred, messages=req["messages"])
            for pred, req in zip(predictions, batch_request["requests"], strict=True)
        ]


class BatchJSONAdapter(BatchAdapter):
    """BatchAdapter that uses JSONAdapter for structured output parsing."""

    def __init__(self):
        self.adapter = dspy.JSONAdapter()


class BatchChatAdapter(BatchAdapter):
    """
    BatchAdapter that uses ChatAdapter with fallback to JSONAdapter for failures.

    On first pass, uses ChatAdapter to parse results. Any failures are collected
    and retried using BatchJSONAdapter. The final results combine successes from
    both passes.
    """

    def __init__(self):
        self.adapter = dspy.ChatAdapter()

    async def __call__(
        self,
        predictor: dspy.Predict,
        inputs: list[dict],
        batch_client: BatchClient,
        show_progress: bool = True,
    ) -> list[ABatchResult]:
        """
        Execute batch with ChatAdapter, retry failures with JSONAdapter.

        Args:
            predictor: The dspy.Predict instance to use for processing.
            inputs: List of input dictionaries for the predictor.
            batch_client: The BatchClient to use for submitting and polling.
            show_progress: Whether to show a progress display while waiting.

        Returns:
            A list of ABatchResult objects, one per input.
        """
        # First pass: run with ChatAdapter
        results = await super().__call__(predictor, inputs, batch_client, show_progress)

        # Collect failed predictions and their original indices
        failed_indices: list[int] = []
        failed_inputs: list[dict] = []
        for result in results:
            pred = result["prediction"]
            if isinstance(pred, FailedPrediction):
                failed_indices.append(pred.index)
                failed_inputs.append(inputs[pred.index])

        # If no failures, return as-is
        if not failed_inputs:
            return results

        if show_progress:
            print(f"{len(failed_inputs)} failures detected, retrying with JSONAdapter")

        # Second pass: retry failures with JSONAdapter
        json_adapter = BatchJSONAdapter()
        retry_results = await json_adapter(predictor, failed_inputs, batch_client, show_progress=show_progress)

        # Merge retry results back into the original results list
        for original_index, retry_result in zip(failed_indices, retry_results, strict=True):
            results[original_index] = retry_result

        return results


# Mapping from dspy Adapter types to BatchAdapter classes
BATCH_ADAPTERS: dict[type[dspy.Adapter], type[BatchAdapter]] = {
    dspy.ChatAdapter: BatchChatAdapter,
    dspy.JSONAdapter: BatchJSONAdapter,
}


def get_batch_adapter(dspy_adapter: Optional[dspy.Adapter] = None) -> BatchAdapter:
    """
    Get the appropriate BatchAdapter for the given dspy Adapter.

    Args:
        dspy_adapter: The dspy Adapter to match. If None, uses dspy.settings.adapter
                     or falls back to ChatAdapter.

    Returns:
        An instance of the appropriate BatchAdapter subclass.

    Raises:
        NotImplementedError: If no BatchAdapter exists for the given dspy Adapter type.
    """
    adapter = dspy_adapter or dspy.settings.adapter or dspy.ChatAdapter()
    adapter_type = type(adapter)

    if adapter_type not in BATCH_ADAPTERS:
        raise NotImplementedError(
            f"No BatchAdapter implemented for {adapter_type.__name__}. "
            f"Supported adapters: {[a.__name__ for a in BATCH_ADAPTERS.keys()]}"
        )

    return BATCH_ADAPTERS[adapter_type]()


def get_batch_client(
    provider: str,
    api_key: Optional[str] = None,
    poll_interval: float = 30.0,
    max_poll_time: str = "24h",
    status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
) -> BatchClient:
    """
    Get a batch client for the given provider.

    Args:
        provider: The provider name (e.g., "openai", "together_ai", "fireworks_ai", "azure").
        api_key: Optional API key for the provider. If not provided, will use environment variables.
        poll_interval: Seconds between status checks (default: 30.0).
        max_poll_time: Maximum time to wait for completion as a string like "30s", "5m", or "24h" (default: "24h").

    Returns:
        An instance of the appropriate BatchClient subclass.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider not in CLIENTS:
        raise ValueError(
            f"Provider '{provider}' does not support batching. Supported providers: {list(CLIENTS.keys())}"
        )
    return CLIENTS[provider](
        api_key=api_key, poll_interval=poll_interval, max_poll_time=max_poll_time, status_callback=status_callback
    )


async def submit_batch_job(
    predictor: dspy.Predict,
    inputs: list[dict],
) -> str:
    """
    Submit a batch job and return the batch ID immediately.

    This function creates a batch request, formats it for the appropriate provider,
    and submits it without waiting for completion.

    The BatchAdapter is automatically determined from dspy.settings.adapter.

    Args:
        predictor: A dspy.Predict instance to use for processing.
        inputs: A list of input dictionaries for the predictor.

    Returns:
        The batch_id string for tracking the job.

    Raises:
        ValueError: If no LM is configured or provider doesn't support batching.
        NotImplementedError: If no BatchAdapter exists for the configured dspy Adapter.

    Example:
        ```python
        import dspy
        from modaic.batch import submit_batch_job

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
        predictor = dspy.Predict("question -> answer")

        inputs = [
            {"question": "What is 2+2?"},
            {"question": "What is the capital of France?"},
        ]

        # Submit and get batch_id for later tracking
        batch_id = await submit_batch_job(predictor, inputs)
        print(f"Submitted batch: {batch_id}")
        ```
    """
    lm = getattr(predictor, "lm", None) or dspy.settings.lm
    if lm is None:
        raise ValueError(
            "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. "
            "e.g, `dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
        )

    model = lm.model
    _, provider, _, _ = get_llm_provider(model)

    # Extract API key from LM kwargs if available
    api_key = getattr(lm, "kwargs", {}).get("api_key")

    # Get the appropriate batch client and adapter from settings
    batch_client = get_batch_client(provider, api_key=api_key)
    batch_adapter = get_batch_adapter()

    # Format and submit
    batch_request = batch_adapter.format(predictor, inputs)
    return await batch_client.submit(batch_request)


async def abatch(
    predictor: dspy.Predict,
    inputs: list[dict],
    show_progress: bool = True,
    poll_interval: float = 30.0,
    max_poll_time: str = "24h",
    status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    batch_size: Optional[int] = None,
) -> list[ABatchResult]:
    """
    Submit a batch of inputs and wait for completion.

    This function creates a batch request, formats it for the appropriate provider,
    submits it, and waits for completion before returning the results.

    The BatchAdapter is automatically determined from dspy.settings.adapter.

    Args:
        predictor: A dspy.Predict instance to use for processing.
        inputs: A list of input dictionaries for the predictor.
        show_progress: If True, show a progress display while waiting (requires rich).
        poll_interval: Seconds between status checks (default: 30.0).
        max_poll_time: Maximum time to wait for completion as a string like "30s", "5m", or "24h" (default: "24h").

    Returns:
        A list of ABatchResult objects, one per input.

    Raises:
        ValueError: If no LM is configured or provider doesn't support batching.
        NotImplementedError: If no BatchAdapter exists for the configured dspy Adapter.
        TimeoutError: If job doesn't complete within max_poll_time.
        RuntimeError: If the batch job fails.

    Example:
        ```python
        import dspy
        from modaic.batch import abatch

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
        predictor = dspy.Predict("question -> answer")

        inputs = [
            {"question": "What is 2+2?"},
            {"question": "What is the capital of France?"},
        ]

        # Wait for results with progress display
        results = await abatch(predictor, inputs)
        for res in results:
            print(res["prediction"].answer)

        # Custom polling interval and max wait time
        results = await abatch(predictor, inputs, poll_interval=60.0, max_poll_time="1h")
        ```
    """
    lm = getattr(predictor, "lm", None) or dspy.settings.lm
    if lm is None:
        raise ValueError(
            "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. "
            "e.g, `dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
        )

    model = lm.model
    _, provider, _, _ = get_llm_provider(model)

    # Extract API key from LM kwargs if available
    api_key = getattr(lm, "kwargs", {}).get("api_key")

    async def stagger(delay_s: float, coro):
        await asyncio.sleep(delay_s)
        return await coro

    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    if batch_size:
        input_batches = list(chunk_list(inputs, batch_size))
    else:
        input_batches = [inputs]

    batch_statuses = {}  # index -> {overall_start_time, attempts: [ {id, status, progress, metadata, start_time} ]}
    results_list = [None] * len(input_batches)

    def get_status_callback(idx):
        def cb(batch_id, status, progress, metadata):
            import time

            now = time.time()
            if idx not in batch_statuses:
                batch_statuses[idx] = {
                    "overall_start_time": now,
                    "attempts": [],
                }

            current_batch = batch_statuses[idx]

            # Find the attempt with this batch_id or create new one
            attempt = None
            for a in current_batch["attempts"]:
                if a["id"] == batch_id:
                    attempt = a
                    break

            if attempt is None:
                attempt = {
                    "id": batch_id,
                    "status": status,
                    "progress": progress,
                    "metadata": metadata,
                    "start_time": now,
                }
                current_batch["attempts"].append(attempt)
            else:
                attempt["status"] = status
                attempt["progress"] = progress
                attempt["metadata"] = metadata

            if status_callback:
                status_callback(batch_id, status, progress, metadata)

        return cb

    async def run_batch(idx, batch_inputs):
        client = get_batch_client(
            provider,
            api_key=api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=get_status_callback(idx),
        )
        adapter = get_batch_adapter()
        res = await adapter(predictor, batch_inputs, client, show_progress=False)
        results_list[idx] = res
        return res

    if show_progress:
        try:
            import time

            from rich.console import Console, Group
            from rich.live import Live
            from rich.panel import Panel
            from rich.spinner import Spinner
            from rich.table import Table
            from rich.text import Text

            console = Console()

            def format_elapsed(seconds: float) -> str:
                if seconds < 60:
                    return f"{seconds:.0f}s"
                elif seconds < 3600:
                    mins, secs = divmod(int(seconds), 60)
                    return f"{mins}m {secs}s"
                else:
                    hours, remainder = divmod(int(seconds), 3600)
                    mins, secs = divmod(remainder, 60)
                    return f"{hours}h {mins}m {secs}s"

            def get_status_style(s: str) -> tuple[str, str]:
                s_lower = s.lower()
                if s_lower == "completed":
                    return "green", "[green]completed[/green]"
                elif s_lower in ("failed", "cancelled", "expired"):
                    return "red", f"[red]{s_lower}[/red]"
                elif s_lower in ("in_progress", "running", "validating", "pending", "submitted"):
                    return "yellow", f"[yellow]{s_lower}[/yellow]"
                else:
                    return "yellow", f"[yellow]{s}[/yellow]"

            def make_batch_panel(idx, status_data) -> Panel:
                attempts = status_data.get("attempts", [])
                # overall_start_time = status_data.get("overall_start_time")

                table = Table.grid(padding=(0, 4))
                table.add_column(style="cyan", justify="right")  # Label column

                # Add a column for each attempt
                for i in range(len(attempts)):
                    table.add_column(style="white", min_width=20)

                # Row: Batch ID
                id_row = ["Batch ID:"]
                for a in attempts:
                    id_row.append(a["id"] or "[dim]submitting...[/dim]")
                table.add_row(*id_row)

                # Row: Provider
                provider_row = ["Provider:"]
                for _ in attempts:
                    provider_row.append(f"[magenta]{provider}[/magenta]")
                table.add_row(*provider_row)

                table.add_row()  # Spacer

                # Row: Requests
                req_row = ["Requests:"]
                for a in attempts:
                    num = a["metadata"].get("num_requests") or len(input_batches[idx])
                    req_row.append(f"[bold]{num}[/bold]")
                table.add_row(*req_row)

                # Row: Status
                status_row = ["Status:"]
                for a in attempts:
                    _, styled = get_status_style(a["status"])
                    status_row.append(styled)
                table.add_row(*status_row)

                # Row: Progress
                prog_row = ["Progress:"]
                for a in attempts:
                    p = a["progress"]
                    prog_row.append(f"[bold]{p}%[/bold]" if p is not None else "[dim]N/A[/dim]")
                table.add_row(*prog_row)

                # Row: Elapsed
                elapsed_row = ["Elapsed:"]
                for a in attempts:
                    e = time.time() - a["start_time"]
                    elapsed_row.append(f"[dim]{format_elapsed(e)}[/dim]")
                table.add_row(*elapsed_row)

                # Row: Attempt
                attempt_row = ["Attempt:"]
                for i in range(len(attempts)):
                    attempt_row.append(f"[bold yellow]{i + 1}[/bold yellow]")
                table.add_row(*attempt_row)

                # Decide if we show the spinner based on the LAST attempt
                show_spinner = False
                if attempts:
                    last_status = attempts[-1]["status"].lower()
                    if last_status not in ("completed", "failed", "cancelled", "expired"):
                        show_spinner = True

                if show_spinner:
                    spinner = Spinner("dots", text=Text(" Processing...", style="dim"))
                    content = Group(table, Text(""), spinner)
                else:
                    content = table

                return Panel(
                    content,
                    title=f"[bold blue]Batch {idx + 1}/{len(input_batches)}[/bold blue]",
                    border_style="blue",
                    padding=(1, 2),
                )

            def make_display() -> Panel:
                panels = []
                for i in range(len(input_batches)):
                    status_data = batch_statuses.get(i, {})
                    panels.append(make_batch_panel(i, status_data))

                return Panel(
                    Group(*panels),
                    title="[bold blue]Batch Processing[/bold blue]",
                    border_style="bold blue",
                    padding=(1, 2),
                )

            with Live(make_display(), console=console, refresh_per_second=4) as live:
                tasks = []
                for i, batch in enumerate(input_batches):
                    tasks.append(asyncio.create_task(stagger(i * 1.0, run_batch(i, batch))))

                while any(res is None for res in results_list):
                    live.update(make_display())
                    done, pending = await asyncio.wait(tasks, timeout=0.5)
                    for task in done:
                        if task.exception():
                            raise task.exception()
                    if len(done) == len(tasks):
                        break

                live.update(make_display())

        except ImportError:
            show_progress = False

    if not show_progress:
        tasks = []
        for i, batch in enumerate(input_batches):
            tasks.append(stagger(i * 1.0, run_batch(i, batch)))
        await asyncio.gather(*tasks)

    # Flatten results
    final_results = []
    for res in results_list:
        if res is None:
            raise RuntimeError("One or more batches failed to return results")
        final_results.extend(res)

    return final_results


async def aget_batch_status(batch_id: str, provider: str, api_key: Optional[str] = None) -> Tuple[str, Optional[int]]:
    """
    Get the status of a batch job.

    Args:
        batch_id: The batch ID to check.
        provider: The provider name (e.g., "openai", "together_ai").
        api_key: Optional API key for the provider.

    Returns:
        Status string (e.g., "completed", "in_progress", "failed").
    """
    batch_client = get_batch_client(provider, api_key=api_key)
    return await batch_client.get_status(batch_id)


async def aget_batch_results(batch_id: str, provider: str, api_key: Optional[str] = None) -> BatchReponse:
    """
    Get the results of a completed batch job.

    Args:
        batch_id: The batch ID to get results for.
        provider: The provider name (e.g., "openai", "together_ai").
        api_key: Optional API key for the provider.

    Returns:
        BatchReponse containing the results.
    """
    batch_client = get_batch_client(provider, api_key=api_key)
    return await batch_client.get_results(batch_id)


async def acancel_batch(batch_id: str, provider: str, api_key: Optional[str] = None) -> bool:
    """
    Cancel a batch job.

    Args:
        batch_id: The batch ID to cancel.
        provider: The provider name (e.g., "openai", "together_ai").
        api_key: Optional API key for the provider.

    Returns:
        True if cancellation was successful.
    """
    batch_client = get_batch_client(provider, api_key=api_key)
    return await batch_client.cancel(batch_id)


def supports_abatch(lm_or_model: dspy.LM | str) -> bool:
    """
    Check if the given LM or predictor supports batching.
    """
    if isinstance(lm_or_model, dspy.LM):
        model = lm_or_model.model
    else:
        model = lm_or_model
    _, provider, _, _ = get_llm_provider(model)
    if provider in CLIENTS:
        return True
    return False
