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
    # VertexAIBatchClient,
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
        logger.debug("Formatting batch request with %d inputs", len(inputs_list))
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
                parsed_outputs = self.adapter._call_postprocess(processed_signature, signature, [output], lm, config)
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
        logger.debug(
            "BatchAdapter start: inputs=%d, show_progress=%s, adapter=%s",
            len(inputs),
            show_progress,
            type(self.adapter).__name__,
        )
        # Format inputs into a BatchRequest
        batch_request = self.format(predictor, inputs)
        logger.debug(
            "BatchAdapter formatted request: requests=%d, model=%s",
            len(batch_request["requests"]),
            batch_request.get("model"),
        )

        # Submit and wait for results
        batch_result = await batch_client.submit_and_wait(batch_request, show_progress=show_progress)
        logger.debug(
            "BatchAdapter received results: batch_id=%s, status=%s, results=%d, errors=%d",
            batch_result.batch_id,
            batch_result.status,
            len(batch_result.results),
            len(batch_result.errors or []),
        )

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
            logger.error(
                "Batch result count mismatch: expected=%d got=%d successes=%d errors=%d",
                len(inputs),
                len(result_items),
                len(batch_result.results),
                len(batch_result.errors or []),
            )
            raise RuntimeError(
                f"Batch result count mismatch: expected {len(inputs)}, "
                f"got {len(result_items)} ({len(batch_result.results)} successes, "
                f"{len(batch_result.errors or [])} errors from API)"
            )

        # Parse into predictions
        predictions = self.parse(predictor, inputs, result_items)
        logger.debug("BatchAdapter parsed predictions: count=%d", len(predictions))

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
        logger.debug("BatchChatAdapter first pass complete: results=%d", len(results))

        # Collect failed predictions and their original indices
        failed_indices: list[int] = []
        failed_inputs: list[dict] = []
        for result in results:
            pred = result.prediction
            if isinstance(pred, FailedPrediction):
                failed_indices.append(pred.index)
                failed_inputs.append(inputs[pred.index])

        # If no failures, return as-is
        if not failed_inputs:
            logger.info("BatchChatAdapter completed with no failures returning results: results=%d", len(results))
            return results

        logger.info(
            "BatchChatAdapter retrying failures with JSONAdapter: failures=%d",
            len(failed_inputs),
        )

        # Second pass: retry failures with JSONAdapter
        json_adapter = BatchJSONAdapter()
        retry_results = await json_adapter(predictor, failed_inputs, batch_client, show_progress=show_progress)

        # Merge retry results back into the original results list
        for original_index, retry_result in zip(failed_indices, retry_results, strict=True):
            results[original_index] = retry_result

        failed = sum(1 for result in results if isinstance(result.prediction, FailedPrediction))
        logger.info("BatchChatAdapter retry merge complete: failed=%d total=%d", failed, len(results))
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

    logger.debug("Selected batch adapter: %s", adapter_type.__name__)
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
    logger.debug(
        "Creating batch client: provider=%s poll_interval=%s max_poll_time=%s",
        provider,
        poll_interval,
        max_poll_time,
    )
    return CLIENTS[provider](
        api_key=api_key, poll_interval=poll_interval, max_poll_time=max_poll_time, status_callback=status_callback
    )


def _get_batch_context(predictor: dspy.Predict) -> Tuple[str, Optional[str]]:
    """Helper to extract provider and API key from predictor or settings."""
    lm = getattr(predictor, "lm", None) or dspy.settings.lm
    if lm is None:
        raise ValueError(
            "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. "
            "e.g, `dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
        )

    model = lm.model
    _, provider, _, _ = get_llm_provider(model)
    api_key = getattr(lm, "kwargs", {}).get("api_key")
    return provider, api_key


class BatchProgressDisplay:
    """Handles the rich-based progress display for batch jobs."""

    def __init__(
        self,
        num_requests: int,
        provider: str,
        status_callback: Optional[Callable] = None,
    ):
        import time

        self.num_requests = num_requests
        self.provider = provider
        self.num_cached = 0
        self.user_callback = status_callback
        self.status: dict[str, Any] = {
            "id": None,
            "status": "submitting",
            "progress": 0,
            "metadata": {"num_requests": num_requests},
            "start_time": time.time(),
        }
        self.live = None

    def update(
        self,
        batch_id: Optional[str],
        status: Optional[str],
        progress: Optional[int],
        metadata: Optional[dict],
    ):
        """Update the internal status and call the user's callback if provided."""
        if batch_id is not None:
            self.status["id"] = batch_id
        if status is not None:
            self.status["status"] = status
        if progress is not None:
            self.status["progress"] = progress
        if metadata is not None:
            self.status["metadata"] = metadata
            if "num_cached" in metadata:
                self.num_cached = metadata.get("num_cached")

        if self.user_callback:
            self.user_callback(batch_id, status, progress, metadata)

        logger.debug(
            "Batch progress update: batch_id=%s status=%s progress=%s",
            batch_id,
            status,
            progress,
        )
        if self.live:
            self.live.update(self.make_panel())

    def _format_elapsed(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            mins, secs = divmod(int(seconds), 60)
            return f"{mins}m {secs}s"
        hours, remainder = divmod(int(seconds), 3600)
        mins, secs = divmod(remainder, 60)
        return f"{hours}h {mins}m {secs}s"

    def make_panel(self) -> Any:
        """Create a rich Panel representing the current batch status."""
        import time

        from rich.console import Group
        from rich.panel import Panel
        from rich.spinner import Spinner
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(padding=(0, 4))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white", min_width=20)

        table.add_row("Batch ID:", self.status["id"] or "[dim]submitting...[/dim]")
        table.add_row("Provider:", f"[magenta]{self.provider}[/magenta]")
        table.add_row()

        num = self.status["metadata"].get("num_requests") or self.num_requests
        table.add_row("Requests:", f"[bold]{num}[/bold]")
        table.add_row("Cached:", f"[bold]{self.num_cached}[/bold]")

        status = (self.status["status"] or "unknown").lower()
        if status == "completed":
            styled_status = "[green]completed[/green]"
        elif status in ("failed", "cancelled", "expired"):
            styled_status = f"[red]{status}[/red]"
        else:
            styled_status = f"[yellow]{status}[/yellow]"
        table.add_row("Status:", styled_status)

        p = self.status["progress"]
        table.add_row("Progress:", f"[bold]{p}%[/bold]" if p is not None else "[dim]N/A[/dim]")

        elapsed = time.time() - self.status["start_time"]
        table.add_row("Elapsed:", f"[dim]{self._format_elapsed(elapsed)}[/dim]")

        show_spinner = status not in ("completed", "failed", "cancelled", "expired")
        content = Group(table, Text(""), Spinner("dots", text=" Processing...") if show_spinner else Text(""))

        return Panel(
            content,
            title="[bold blue]Batch Processing[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )

    def set_live(self, live):
        self.live = live


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
    logger.debug("submit_batch_job start: inputs=%d", len(inputs))
    provider, api_key = _get_batch_context(predictor)

    # Get the appropriate batch client and adapter from settings
    batch_client = get_batch_client(provider, api_key=api_key)
    batch_adapter = get_batch_adapter()

    # Format and submit
    batch_request = batch_adapter.format(predictor, inputs)
    logger.debug("submit_batch_job submitting request: requests=%d", len(batch_request["requests"]))
    batch_id = await batch_client.submit(batch_request)
    logger.debug("submit_batch_job submitted batch_id=%s", batch_id)
    return batch_id


async def abatch(
    predictor: dspy.Predict,
    inputs: list[dict],
    show_progress: bool = True,
    poll_interval: float = 30.0,
    max_poll_time: str = "24h",
    status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
) -> list[ABatchResult]:
    """
    Submit a batch of inputs and wait for completion.

    This function creates a single batch request for all inputs, formats it for the appropriate provider,
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
            print(res.prediction.answer)

        # Custom polling interval and max wait time
        results = await abatch(predictor, inputs, poll_interval=60.0, max_poll_time="1h")
        ```
    """
    logger.debug(
        "abatch start: inputs=%d show_progress=%s poll_interval=%s max_poll_time=%s",
        len(inputs),
        show_progress,
        poll_interval,
        max_poll_time,
    )
    provider, api_key = _get_batch_context(predictor)

    display = None
    if show_progress:
        try:
            display = BatchProgressDisplay(len(inputs), provider, status_callback)
        except ImportError:
            logger.debug("abatch progress display unavailable; rich not installed")

    async def run_batch():
        client = get_batch_client(
            provider,
            api_key=api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=display.update if display else status_callback,
        )
        adapter = get_batch_adapter()
        return await adapter(predictor, inputs, client, show_progress=False)

    if display:
        from rich.live import Live

        with Live(display.make_panel(), refresh_per_second=4) as live:
            display.set_live(live)
            task = asyncio.create_task(run_batch())
            while not task.done():
                live.update(display.make_panel())
                await asyncio.sleep(0.5)
            return await task

    return await run_batch()


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
