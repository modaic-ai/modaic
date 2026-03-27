import asyncio
import logging
from importlib import import_module
from typing import Any, Callable, Optional, Tuple

import dspy
from litellm import get_llm_provider
from rich.live import Live

from .adapters import BatchAdapter, BatchChatAdapter, BatchJSONAdapter, BatchXMLAdapter, BatchRequestContext, BATCH_ADAPTERS, get_batch_adapter
from .clients import BatchClient
from .types import ABatchResult, ABatchRow, BatchReponse, BatchRequest, BatchRequestItem, FailedPrediction, ResultItem

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

CLIENTS: dict[str, tuple[str, str]] = {
    "openai": ("modaic.batch.clients", "OpenAIBatchClient"),
    "anthropic": ("modaic.batch.clients", "AnthropicBatchClient"),
    "together_ai": ("modaic.batch.clients", "TogetherBatchClient"),
    "azure": ("modaic.batch.clients", "AzureBatchClient"),
    "fireworks_ai": ("modaic.batch.clients", "FireworksBatchClient"),
}


GroupedBatchInputs = list[tuple[dspy.Predict, list[dict[str, Any]]]]


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
    module_name, class_name = CLIENTS[provider]
    client_cls = getattr(import_module(module_name), class_name)
    return client_cls(
        api_key=api_key, poll_interval=poll_interval, max_poll_time=max_poll_time, status_callback=status_callback
    )


def _get_predictor_lm(predictor: dspy.Predict) -> dspy.LM:
    lm = getattr(predictor, "lm", None) or dspy.settings.lm
    if lm is None:
        raise ValueError(
            "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. "
            "e.g, `dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
        )
    return lm


def _get_batch_context(predictor: dspy.Predict) -> Tuple[str, Optional[str]]:
    """Helper to extract provider and API key from predictor or settings."""
    lm = _get_predictor_lm(predictor)
    model = lm.model
    _, provider, _, _ = get_llm_provider(model)
    api_key = getattr(lm, "kwargs", {}).get("api_key")
    return provider, api_key


def _flatten_grouped_inputs(inputs: GroupedBatchInputs) -> list[BatchRequestContext]:
    request_contexts: list[BatchRequestContext] = []
    for group_index, (predictor, predictor_inputs) in enumerate(inputs):
        for example_index, input_example in enumerate(predictor_inputs):
            request_index = len(request_contexts)
            request_contexts.append(
                BatchRequestContext(
                    predictor=predictor,
                    inputs=dict(input_example),
                    group_index=group_index,
                    example_index=example_index,
                    request_index=request_index,
                    request_id=f"request-{request_index}",
                )
            )
    return request_contexts


def _build_grouped_results(
    batch_id: str,
    inputs: GroupedBatchInputs,
    request_contexts: list[BatchRequestContext],
    rows: list[ABatchRow],
) -> list[tuple[dspy.Predict, ABatchResult]]:
    rows_by_group: list[list[tuple[int, ABatchRow]]] = [[] for _ in inputs]
    for request_context, row in zip(request_contexts, rows, strict=True):
        rows_by_group[request_context.group_index].append((request_context.example_index, row))

    grouped_results: list[tuple[dspy.Predict, ABatchResult]] = []
    for group_index, (predictor, _) in enumerate(inputs):
        group_rows = [row for _, row in sorted(rows_by_group[group_index], key=lambda item: item[0])]
        grouped_results.append((predictor, ABatchResult.from_rows(batch_id, group_rows, predict_index=group_index)))
    return grouped_results


def _resolve_grouped_batch_context(inputs: GroupedBatchInputs) -> tuple[str, Optional[str]]:
    providers: list[str] = []
    explicit_api_keys: set[str] = set()
    for predictor, _ in inputs:
        provider, api_key = _get_batch_context(predictor)
        providers.append(provider)
        if api_key is not None:
            explicit_api_keys.add(api_key)

    unique_providers = set(providers)
    if len(unique_providers) != 1:
        raise ValueError("All predictors passed to modaic.batch.abatch must use the same provider")

    if len(explicit_api_keys) > 1:
        raise ValueError("All predictors passed to modaic.batch.abatch must use the same API key or pass `client=`")

    return providers[0], next(iter(explicit_api_keys), None) if explicit_api_keys else None


def _validate_explicit_client(inputs: GroupedBatchInputs, client: BatchClient) -> None:
    provider = getattr(client, "provider", None)
    if provider in {"modal", "vllm"}:
        client_lm = getattr(client, "lm", None)
        client_model = getattr(client_lm, "model", None)
        for predictor, _ in inputs:
            predictor_lm = _get_predictor_lm(predictor)
            predictor_model = getattr(predictor_lm, "model", None)
            if not isinstance(predictor_model, str) or not predictor_model.startswith("huggingface/"):
                raise ValueError(
                    f"{provider} batch client requires all predictors to use `huggingface/...` models"
                )
            if isinstance(client_model, str) and predictor_model != client_model:
                raise ValueError(
                    f"{provider} batch client requires all predictors to use the same model as the client LM"
                )
        return

    if provider is None:
        return

    for predictor, _ in inputs:
        predictor_provider, _ = _get_batch_context(predictor)
        if predictor_provider != provider:
            raise ValueError(
                f"Explicit batch client provider '{provider}' does not match predictor provider '{predictor_provider}'"
            )


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

    def set_live(self, live: Live):
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
    batch_request = batch_adapter.format(_flatten_grouped_inputs([(predictor, inputs)]))
    logger.debug("submit_batch_job submitting request: requests=%d", len(batch_request["requests"]))
    batch_id = await batch_client.submit(batch_request)
    logger.debug("submit_batch_job submitted batch_id=%s", batch_id)
    return batch_id


async def abatch(
    inputs: GroupedBatchInputs,
    show_progress: bool = True,
    poll_interval: float = 30.0,
    max_poll_time: str = "24h",
    return_messages: bool = False,
    status_callback: Optional[Callable[[str, str, Optional[int], dict], None]] = None,
    client: Optional[BatchClient] = None,
) -> list[tuple[dspy.Predict, ABatchResult]]:
    """
    Submit one grouped batch request and wait for completion.

    This function flattens multiple predictor/input groups into a single batch request,
    submits that request, and returns one ABatchResult per predictor group.

    The BatchAdapter is automatically determined from dspy.settings.adapter.

    Args:
        inputs: A list of (predictor, input_examples) pairs to batch together.
        show_progress: If True, show a progress display while waiting (requires rich).
        poll_interval: Seconds between status checks (default: 30.0).
        max_poll_time: Maximum time to wait for completion as a string like "30s", "5m", or "24h" (default: "24h").
        return_messages: If True, attach request messages and raw assistant outputs to each returned prediction.

    Returns:
        A list of (predictor, ABatchResult) pairs in the same order as the input groups.

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
        grouped_results = await abatch([(predictor, inputs)])
        _, results = grouped_results[0]
        for res in results:
            print(res.prediction.answer)

        # Custom polling interval and max wait time
        grouped_results = await abatch([(predictor, inputs)], poll_interval=60.0, max_poll_time="1h")
        ```
    """
    if not inputs:
        return []

    request_contexts = _flatten_grouped_inputs(inputs)
    logger.debug(
        "abatch start: inputs=%d show_progress=%s poll_interval=%s max_poll_time=%s",
        len(request_contexts),
        show_progress,
        poll_interval,
        max_poll_time,
    )
    provider = None
    api_key = None
    if client is None:
        provider, api_key = _resolve_grouped_batch_context(inputs)
    else:
        _validate_explicit_client(inputs, client)

    if getattr(client, "provider", None) in {"modal", "vllm"}:
        show_progress = False

    display = None
    if show_progress:
        try:
            assert provider is not None
            display = BatchProgressDisplay(len(request_contexts), provider, status_callback)
        except ImportError:
            logger.debug("abatch progress display unavailable; rich not installed")

    async def run_batch():
        resolved_client = client or get_batch_client(
            provider,
            api_key=api_key,
            poll_interval=poll_interval,
            max_poll_time=max_poll_time,
            status_callback=display.update if display else status_callback,
        )
        adapter = get_batch_adapter()
        batch_id, rows = await adapter(
            request_contexts,
            resolved_client,
            show_progress=False,
            return_messages=return_messages,
        )
        return _build_grouped_results(batch_id, inputs, request_contexts, rows)

    if display:
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
