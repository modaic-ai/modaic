from dataclasses import dataclass
from typing import Any, Optional, TypedDict

import dspy


class Message(TypedDict):
    role: str
    content: str | list[dict]


class BatchRequestItem(TypedDict):
    id: str
    model: str
    messages: list[Message]
    lm_kwargs: dict


@dataclass
class FailedPrediction:
    """Represents a failed prediction that can be retried with a different adapter."""

    error: str
    index: int  # Original index in the inputs list


class ABatchResult(TypedDict):
    prediction: dspy.Prediction | FailedPrediction
    messages: list[Message]


class BatchRequest(TypedDict):
    """
    Common batch request format for all providers.

    Args:
        requests: List of batch request items.
        model: Model to use for the batch request if all requests have the same model. None means all requests have different models.
        lm_kwargs: LM kwargs to use for the batch request if all requests have the same LM kwargs. None means all requests have different LM kwargs.
    """

    requests: list[BatchRequestItem]
    model: Optional[str]
    lm_kwargs: Optional[dict]


class ResultItem(TypedDict, total=False):
    """
    Result item from a batch completion.

    The `text` field is required. Optional fields include `logprobs` and `tool_calls`
    which may be present depending on the request configuration.
    """

    text: str  # Required: the completion text
    logprobs: Optional[dict]  # Optional: log probabilities if requested
    tool_calls: Optional[list[dict]]  # Optional: tool calls if function calling was used


# Make text required
ResultItem.__required_keys__ = frozenset({"text"})
ResultItem.__optional_keys__ = frozenset({"logprobs", "tool_calls"})


@dataclass
class BatchReponse:
    """Result from a completed batch job (raw API response)."""

    batch_id: str
    status: str
    results: list[dict[str, Any]]  # Raw results from the batch API
    errors: Optional[list[dict]] = None
