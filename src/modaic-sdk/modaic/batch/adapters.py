from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import dspy
import litellm
from litellm import get_llm_provider as _get_llm_provider

from .types import BatchRequestItem, FailedPrediction, ResultItem

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
    """Translates dspy predictors/inputs into BatchRequestItems and predictions back out.

    Subclasses override `adapter` and optionally `parse_output` (for XML, which skips
    the ChatAdapter delimiter fixup) and `fallback_for_failures` (for Chat/XML → JSON retry).
    """

    adapter: dspy.Adapter

    def format(self, contexts: list[BatchRequestContext]) -> list[BatchRequestItem]:
        items: list[BatchRequestItem] = []
        for ctx in contexts:
            lm, config, signature, demos, _ = ctx.predictor._forward_preprocess(**ctx.inputs)
            with dspy.settings.context(send_stream=None):
                processed = self.adapter._call_preprocess(lm, config, signature, ctx.inputs)
                messages = self.adapter.format(processed, demos, ctx.inputs)
            model_name = get_llm_provider(lm.model)[0]
            lm_stored = {
                k: v for k, v in getattr(lm, "kwargs", {}).items()
                if v is not None and k not in ("api_key", "api_base", "api_version")
            }
            merged_kwargs = {**lm_stored, **config}
            items.append(
                BatchRequestItem(
                    id=ctx.request_id,
                    model=model_name,
                    messages=messages,
                    lm_kwargs=merged_kwargs,
                )
            )
        return items

    def parse(
        self,
        contexts: list[BatchRequestContext],
        results: list[ResultItem | FailedPrediction | None],
    ) -> list[dspy.Prediction | FailedPrediction]:
        predictions: list[dspy.Prediction | FailedPrediction] = []
        for ctx, result in zip(contexts, results, strict=True):
            if result is None:
                predictions.append(FailedPrediction(error="API level failure or parse error", index=ctx.example_index))
                continue
            if isinstance(result, FailedPrediction):
                predictions.append(FailedPrediction(error=result.error, index=ctx.example_index))
                continue

            lm, config, signature, _, _ = ctx.predictor._forward_preprocess(**ctx.inputs)
            processed = self.adapter._call_preprocess(lm, config, signature, ctx.inputs)
            output = self._build_output(result)
            try:
                parsed = self.adapter._call_postprocess(processed, signature, [output], lm, config)
                if parsed:
                    predictions.append(dspy.Prediction(**parsed[0]))
                else:
                    predictions.append(FailedPrediction(error="empty output", index=ctx.example_index))
            except Exception as e:
                predictions.append(FailedPrediction(error=str(e), index=ctx.example_index))
        return predictions

    def _build_output(self, result: ResultItem) -> dict[str, Any]:
        """Build the dict passed to adapter._call_postprocess. Override for format-specific fixups."""
        text = re.sub(r"([^\n])(\[\[\s*##)", r"\1\n\2", result["text"])
        return self._attach_extras({"text": text}, result)

    @staticmethod
    def _attach_extras(output: dict[str, Any], result: ResultItem) -> dict[str, Any]:
        if "logprobs" in result:
            output["logprobs"] = result["logprobs"]
        if "tool_calls" in result:
            output["tool_calls"] = result["tool_calls"]
        if result.get("reasoning_content") is not None:
            output["reasoning_content"] = result["reasoning_content"]
        return output

    def fallback_for_failures(self) -> Optional[BatchAdapter]:
        """Optional secondary adapter to retry failed predictions with. Return None to disable."""
        return None


class BatchJSONAdapter(BatchAdapter):
    def __init__(self):
        self.adapter = dspy.JSONAdapter()


class BatchChatAdapter(BatchAdapter):
    def __init__(self):
        self.adapter = dspy.ChatAdapter()

    def fallback_for_failures(self) -> Optional[BatchAdapter]:
        return BatchJSONAdapter()


class BatchXMLAdapter(BatchAdapter):
    def __init__(self):
        self.adapter = dspy.XMLAdapter()

    def _build_output(self, result: ResultItem) -> dict[str, Any]:
        return self._attach_extras({"text": result["text"]}, result)

    def fallback_for_failures(self) -> Optional[BatchAdapter]:
        return BatchJSONAdapter()


BATCH_ADAPTERS: dict[type[dspy.Adapter], type[BatchAdapter]] = {
    dspy.ChatAdapter: BatchChatAdapter,
    dspy.JSONAdapter: BatchJSONAdapter,
    dspy.XMLAdapter: BatchXMLAdapter,
}


def get_batch_adapter(dspy_adapter: Optional[dspy.Adapter] = None) -> BatchAdapter:
    adapter = dspy_adapter or dspy.settings.adapter or dspy.ChatAdapter()
    adapter_type = type(adapter)
    if adapter_type not in BATCH_ADAPTERS:
        raise NotImplementedError(
            f"No BatchAdapter implemented for {adapter_type.__name__}. "
            f"Supported adapters: {[a.__name__ for a in BATCH_ADAPTERS.keys()]}"
        )
    return BATCH_ADAPTERS[adapter_type]()
