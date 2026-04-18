import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import dspy
import litellm
from litellm import get_llm_provider as _get_llm_provider

from .clients.base import BatchClient
from .types import ABatchResult, ABatchRow, BatchRequest, BatchRequestItem, FailedPrediction, ResultItem

logger = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def get_llm_provider(model: str):
    return _get_llm_provider(model)


# Monkey-patch litellm's model-info lookups with cached versions so that
# DSPy's _call_preprocess (which calls these on every row) doesn't repeat
# expensive uncached lookups that can even hit the network.
litellm.supports_function_calling = lru_cache(maxsize=64)(litellm.supports_function_calling)
litellm.supports_reasoning = lru_cache(maxsize=64)(litellm.supports_reasoning)


@dataclass(frozen=True)
class BatchRequestContext:
    predictor: dspy.Predict
    inputs: dict[str, Any]
    group_index: int
    example_index: int
    request_index: int
    request_id: str


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

    def format(self, request_contexts: list[BatchRequestContext]) -> BatchRequest:
        """
        Create a BatchRequest from a predictor and list of inputs.

        Args:
            predictor: The dspy.Predict instance to use.
            inputs_list: List of input dictionaries for the predictor.

        Returns:
            A BatchRequest ready to be submitted to a BatchClient.
        """
        logger.debug("Formatting batch request with %d inputs", len(request_contexts))
        requests = []

        # Track global model and lm_kwargs (None if they differ across requests)
        global_model = None
        global_lm_kwargs = None

        for request_context in request_contexts:
            predictor = request_context.predictor
            inputs = request_context.inputs
            lm, config, signature, demos, kwargs = predictor._forward_preprocess(**inputs)
            with dspy.settings.context(send_stream=None):
                processed_signature = self.adapter._call_preprocess(lm, config, signature, inputs)
                messages = self.adapter.format(processed_signature, demos, inputs)

            model_name = get_llm_provider(lm.model)[0]

            requests.append(
                BatchRequestItem(
                    id=request_context.request_id,
                    model=model_name,
                    messages=messages,
                    lm_kwargs=config,
                )
            )

            if request_context.request_index == 0:
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
        request_contexts: list[BatchRequestContext],
        results: list[ResultItem | FailedPrediction | None],
    ) -> list[dspy.Prediction | FailedPrediction]:
        """
        Parse ResultItems back into DSPy Predictions.

        Args:
            request_contexts: The original predictor/input metadata (must be in same order as results).
            results: The ResultItem list, already sorted to match inputs_list order.
                    May contain None for items that failed at the API level.

        Returns:
            A list of dspy.Prediction objects or FailedPrediction for failures.
        """
        predictions: list[dspy.Prediction | FailedPrediction] = []

        for request_context, result in zip(request_contexts, results, strict=True):
            predictor = request_context.predictor
            inputs = request_context.inputs
            example_index = request_context.example_index
            if result is None:
                predictions.append(FailedPrediction(error="API level failure or parse error", index=example_index))
                continue
            if isinstance(result, FailedPrediction):
                predictions.append(FailedPrediction(error=result.error, index=example_index))
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
            if result.get("reasoning_content") is not None:
                output["reasoning_content"] = result["reasoning_content"]

            # Parse using the adapter
            try:
                parsed_outputs = self.adapter._call_postprocess(processed_signature, signature, [output], lm, config)
                if parsed_outputs:
                    predictions.append(dspy.Prediction(**parsed_outputs[0]))
                else:
                    predictions.append(FailedPrediction(error="empty output", index=example_index))
            except Exception as e:
                predictions.append(FailedPrediction(error=str(e), index=example_index))

        return predictions

    async def _execute_rows(
        self,
        request_contexts: list[BatchRequestContext],
        batch_client: BatchClient,
        show_progress: bool = True,
        return_messages: bool = False,
    ) -> tuple[str, list[ABatchRow]]:
        """
        Execute a batch job: format inputs, submit to client, and parse results.

        Args:
            request_contexts: Flattened predictor/input metadata for the batch request.
            batch_client: The BatchClient to use for submitting and polling.
            show_progress: Whether to show a progress display while waiting.

        Returns:
            The provider batch ID and a list of ABatchRow objects, one per input.

        Raises:
            TimeoutError: If the job doesn't complete within max_poll_time.
            RuntimeError: If the batch job fails.
        """
        logger.debug(
            "BatchAdapter start: inputs=%d, show_progress=%s, adapter=%s",
            len(request_contexts),
            show_progress,
            type(self.adapter).__name__,
        )
        # Format inputs into a BatchRequest
        batch_request = self.format(request_contexts)
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
        result_items: list[ResultItem | FailedPrediction | None] = []
        for raw_result in sorted_raw_results:
            try:
                result_items.append(batch_client.parse(raw_result))
            except Exception as exc:
                # If a single item fails to parse, preserve the item-level error
                # so downstream callers can inspect the real failure reason.
                result_items.append(FailedPrediction(error=str(exc), index=-1))

        # Final safety check: ensure the result count matches the input count
        if len(result_items) != len(request_contexts):
            logger.error(
                "Batch result count mismatch: expected=%d got=%d successes=%d errors=%d",
                len(request_contexts),
                len(result_items),
                len(batch_result.results),
                len(batch_result.errors or []),
            )
            raise RuntimeError(
                f"Batch result count mismatch: expected {len(request_contexts)}, "
                f"got {len(result_items)} ({len(batch_result.results)} successes, "
                f"{len(batch_result.errors or [])} errors from API)"
            )

        # Parse into predictions
        request_messages = [list(req["messages"]) for req in batch_request["requests"]]
        predictions = self.parse(request_contexts, result_items)
        logger.debug("BatchAdapter parsed predictions: count=%d", len(predictions))

        if return_messages:
            for prediction, messages, result_item in zip(predictions, request_messages, result_items, strict=True):
                prediction._messages = list(messages)
                prediction_outputs = {
                    "text": None if result_item is None or isinstance(result_item, FailedPrediction) else result_item["text"],
                    "reasoning_content": (
                        None
                        if result_item is None or isinstance(result_item, FailedPrediction)
                        else result_item.get("reasoning_content")
                    ),
                }
                prediction._outputs = prediction_outputs

        # Map predictions back to ABatchResult with their formatted messages
        return batch_result.batch_id, [
            ABatchRow(
                prediction=pred,
                messages=list(req["messages"]),
                outputs={
                    "text": None if result_item is None or isinstance(result_item, FailedPrediction) else result_item["text"],
                    "reasoning_content": (
                        None
                        if result_item is None or isinstance(result_item, FailedPrediction)
                        else result_item.get("reasoning_content")
                    ),
                },
                example=dict(input_example),
            )
            for pred, req, result_item, input_example in zip(
                predictions,
                batch_request["requests"],
                result_items,
                [request_context.inputs for request_context in request_contexts],
                strict=True,
            )
        ]

    async def __call__(
        self,
        request_contexts: list[BatchRequestContext],
        batch_client: BatchClient,
        show_progress: bool = True,
        return_messages: bool = False,
    ) -> tuple[str, list[ABatchRow]]:
        async with batch_client.start():
            return await self._execute_rows(
                request_contexts,
                batch_client,
                show_progress=show_progress,
                return_messages=return_messages,
            )


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
        request_contexts: list[BatchRequestContext],
        batch_client: BatchClient,
        show_progress: bool = True,
        return_messages: bool = False,
    ) -> tuple[str, list[ABatchRow]]:
        """
        Execute batch with ChatAdapter, retry failures with JSONAdapter.

        Args:
            request_contexts: Flattened predictor/input metadata for the batch request.
            batch_client: The BatchClient to use for submitting and polling.
            show_progress: Whether to show a progress display while waiting.

        Returns:
            The provider batch ID and one ABatchRow per flattened input.
        """
        async with batch_client.start():
            # First pass: run with ChatAdapter
            batch_id, rows = await super()._execute_rows(
                request_contexts,
                batch_client,
                show_progress,
                return_messages=return_messages,
            )
            logger.debug("BatchChatAdapter first pass complete: results=%d", len(rows))

            # Collect failed predictions and their original indices
            failed_row_indices: list[int] = []
            failed_request_contexts: list[BatchRequestContext] = []
            for row_index, (request_context, result) in enumerate(zip(request_contexts, rows, strict=True)):
                pred = result.prediction
                if isinstance(pred, FailedPrediction):
                    failed_row_indices.append(row_index)
                    failed_request_contexts.append(request_context)

            # If no failures, return as-is
            if not failed_request_contexts:
                logger.info("BatchChatAdapter completed with no failures returning rows: results=%d", len(rows))
                return batch_id, rows

            logger.info(
                "BatchChatAdapter retrying failures with JSONAdapter: failures=%d",
                len(failed_request_contexts),
            )

            # Second pass: retry failures with JSONAdapter using the same client start() scope.
            json_adapter = BatchJSONAdapter()
            _, retry_rows = await json_adapter._execute_rows(
                failed_request_contexts,
                batch_client,
                show_progress=show_progress,
                return_messages=return_messages,
            )

            # Merge retry results back into the original results list
            for original_row_index, retry_result in zip(failed_row_indices, retry_rows, strict=True):
                rows[original_row_index] = retry_result

            failed = sum(1 for result in rows if isinstance(result.prediction, FailedPrediction))
            logger.info("BatchChatAdapter retry merge complete: failed=%d total=%d", failed, len(rows))
            return batch_id, rows


class BatchXMLAdapter(BatchAdapter):
    """
    BatchAdapter that uses XMLAdapter with fallback to JSONAdapter for failures.

    On first pass, uses XMLAdapter to parse results. Any failures are collected
    and retried using BatchJSONAdapter. The final results combine successes from
    both passes.
    """

    def __init__(self):
        self.adapter = dspy.XMLAdapter()

    def parse(
        self,
        request_contexts: list[BatchRequestContext],
        results: list[ResultItem | FailedPrediction | None],
    ) -> list[dspy.Prediction | FailedPrediction]:
        """Parse ResultItems using XMLAdapter (no ChatAdapter delimiter fixup needed)."""
        predictions: list[dspy.Prediction | FailedPrediction] = []

        for request_context, result in zip(request_contexts, results, strict=True):
            predictor = request_context.predictor
            inputs = request_context.inputs
            example_index = request_context.example_index
            if result is None:
                predictions.append(FailedPrediction(error="API level failure or parse error", index=example_index))
                continue
            if isinstance(result, FailedPrediction):
                predictions.append(FailedPrediction(error=result.error, index=example_index))
                continue

            # Recompute the signature processing for this input
            lm, config, signature, demos, kwargs = predictor._forward_preprocess(**inputs)
            processed_signature = self.adapter._call_preprocess(lm, config, signature, inputs)

            # XML text does not need the ChatAdapter delimiter regex fixup
            output: dict[str, Any] = {"text": result["text"]}
            if "logprobs" in result:
                output["logprobs"] = result["logprobs"]
            if "tool_calls" in result:
                output["tool_calls"] = result["tool_calls"]
            if result.get("reasoning_content") is not None:
                output["reasoning_content"] = result["reasoning_content"]

            # Parse using the adapter
            try:
                parsed_outputs = self.adapter._call_postprocess(processed_signature, signature, [output], lm, config)
                if parsed_outputs:
                    predictions.append(dspy.Prediction(**parsed_outputs[0]))
                else:
                    predictions.append(FailedPrediction(error="empty output", index=example_index))
            except Exception as e:
                predictions.append(FailedPrediction(error=str(e), index=example_index))

        return predictions

    async def __call__(
        self,
        request_contexts: list[BatchRequestContext],
        batch_client: BatchClient,
        show_progress: bool = True,
        return_messages: bool = False,
    ) -> tuple[str, list[ABatchRow]]:
        """
        Execute batch with XMLAdapter, retry failures with JSONAdapter.

        Args:
            request_contexts: Flattened predictor/input metadata for the batch request.
            batch_client: The BatchClient to use for submitting and polling.
            show_progress: Whether to show a progress display while waiting.

        Returns:
            The provider batch ID and one ABatchRow per flattened input.
        """
        async with batch_client.start():
            # First pass: run with XMLAdapter
            batch_id, rows = await super()._execute_rows(
                request_contexts,
                batch_client,
                show_progress,
                return_messages=return_messages,
            )
            logger.debug("BatchXMLAdapter first pass complete: results=%d", len(rows))

            # Collect failed predictions and their original indices
            failed_row_indices: list[int] = []
            failed_request_contexts: list[BatchRequestContext] = []
            for row_index, (request_context, result) in enumerate(zip(request_contexts, rows, strict=True)):
                pred = result.prediction
                if isinstance(pred, FailedPrediction):
                    failed_row_indices.append(row_index)
                    failed_request_contexts.append(request_context)

            # If no failures, return as-is
            if not failed_request_contexts:
                logger.info("BatchXMLAdapter completed with no failures returning rows: results=%d", len(rows))
                return batch_id, rows

            logger.info(
                "BatchXMLAdapter retrying failures with JSONAdapter: failures=%d",
                len(failed_request_contexts),
            )

            # Second pass: retry failures with JSONAdapter using the same client start() scope.
            json_adapter = BatchJSONAdapter()
            _, retry_rows = await json_adapter._execute_rows(
                failed_request_contexts,
                batch_client,
                show_progress=show_progress,
                return_messages=return_messages,
            )

            # Merge retry results back into the original results list
            for original_row_index, retry_result in zip(failed_row_indices, retry_rows, strict=True):
                rows[original_row_index] = retry_result

            failed = sum(1 for result in rows if isinstance(result.prediction, FailedPrediction))
            logger.info("BatchXMLAdapter retry merge complete: failed=%d total=%d", failed, len(rows))
            return batch_id, rows


# Mapping from dspy Adapter types to BatchAdapter classes
BATCH_ADAPTERS: dict[type[dspy.Adapter], type[BatchAdapter]] = {
    dspy.ChatAdapter: BatchChatAdapter,
    dspy.JSONAdapter: BatchJSONAdapter,
    dspy.XMLAdapter: BatchXMLAdapter,
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
