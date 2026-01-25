import asyncio
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
from .types import BatchRequest, BatchRequestItem, BatchResult, FailedPrediction, ResultItem

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
        results: list[ResultItem],
    ) -> list[dspy.Prediction | FailedPrediction]:
        """
        Parse ResultItems back into DSPy Predictions.

        Args:
            predictor: The predictor used for the batch request.
            inputs_list: The original list of inputs (must be in same order as results).
            results: The ResultItem list, already sorted to match inputs_list order.

        Returns:
            A list of dspy.Prediction objects or FailedPrediction for failures.
        """
        predictions: list[dspy.Prediction | FailedPrediction] = []

        for i, (inputs, result) in enumerate(zip(inputs_list, results, strict=True)):
            # Recompute the signature processing for this input
            lm, config, signature, demos, kwargs = predictor._forward_preprocess(**inputs)
            processed_signature = self.adapter._call_preprocess(lm, config, signature, inputs)

            # Build output in the format _call_postprocess expects
            output: dict[str, Any] = {"text": result["text"]}
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
    ) -> list[dspy.Prediction | FailedPrediction]:
        """
        Execute a batch job: format inputs, submit to client, and parse results.

        Args:
            predictor: The dspy.Predict instance to use for processing.
            inputs: List of input dictionaries for the predictor.
            batch_client: The BatchClient to use for submitting and polling.
            show_progress: Whether to show a progress display while waiting.

        Returns:
            A list of dspy.Prediction or FailedPrediction objects, one per input.

        Raises:
            TimeoutError: If the job doesn't complete within max_poll_time.
            RuntimeError: If the batch job fails.
        """
        # Format inputs into a BatchRequest
        batch_request = self.format(predictor, inputs)

        # Submit and wait for results
        batch_result = await batch_client.submit_and_wait(batch_request, show_progress=show_progress)

        # Convert raw results to ResultItems, sorted by custom_id
        result_items: list[ResultItem] = []
        sorted_raw_results = sorted(
            batch_result.results,
            key=lambda x: int(x.get("custom_id", "request-0").split("-")[-1]),
        )
        for raw_result in sorted_raw_results:
            result_items.append(batch_client.parse(raw_result))

        # Parse into predictions
        return self.parse(predictor, inputs, result_items)


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
    ) -> list[dspy.Prediction | FailedPrediction]:
        """
        Execute batch with ChatAdapter, retry failures with JSONAdapter.

        Args:
            predictor: The dspy.Predict instance to use for processing.
            inputs: List of input dictionaries for the predictor.
            batch_client: The BatchClient to use for submitting and polling.
            show_progress: Whether to show a progress display while waiting.

        Returns:
            A list of dspy.Prediction or FailedPrediction objects, one per input.
        """
        # First pass: run with ChatAdapter
        results = await super().__call__(predictor, inputs, batch_client, show_progress)

        # Collect failed predictions and their original indices
        failed_indices: list[int] = []
        failed_inputs: list[dict] = []
        for result in results:
            if isinstance(result, FailedPrediction):
                failed_indices.append(result.index)
                failed_inputs.append(inputs[result.index])

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
    status_callback: Optional[Callable[[str, Optional[int]], None]] = None,
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
    status_callback: Optional[Callable[[str, Optional[int]], None]] = None,
) -> list[dspy.Prediction | FailedPrediction]:
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
        A list of dspy.Prediction or FailedPrediction objects, one per input.

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
        predictions = await abatch(predictor, inputs)
        for pred in predictions:
            print(pred.answer)

        # Custom polling interval and max wait time
        predictions = await abatch(predictor, inputs, poll_interval=60.0, max_poll_time="1h")
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
    batch_client = get_batch_client(
        provider,
        api_key=api_key,
        poll_interval=poll_interval,
        max_poll_time=max_poll_time,
        status_callback=status_callback,
    )
    batch_adapter = get_batch_adapter()

    # Use the adapter to orchestrate the batch process
    return await batch_adapter(predictor, inputs, batch_client, show_progress=show_progress)


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


async def aget_batch_results(batch_id: str, provider: str, api_key: Optional[str] = None) -> BatchResult:
    """
    Get the results of a completed batch job.

    Args:
        batch_id: The batch ID to get results for.
        provider: The provider name (e.g., "openai", "together_ai").
        api_key: Optional API key for the provider.

    Returns:
        BatchResult containing the results.
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
