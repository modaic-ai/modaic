from modaic.batch.batch import (
    BatchJobHandle,
    abatch,
    acancel_batch,
    aget_batch_results,
    aget_batch_status,
    get_batch_adapter,
    get_batch_client,
    submit_batch_job,
    supports_abatch,
)
from modaic.batch.enqueued_limits import EnqueuedLimits
from modaic.batch.lmdb_cache import LmdbLMCache
from modaic.batch.progress import BatchProgressDisplay, ShardEvent
from modaic.batch.runner import BatchJobRunner
from modaic.batch.types import (
    ABatchResult,
    ABatchRow,
    BatchReponse,
    BatchRequestItem,
    BatchResponse,
    FailedPrediction,
    RawResults,
    ShardOutcome,
)

__all__ = [
    "ABatchResult",
    "ABatchRow",
    "BatchJobHandle",
    "BatchJobRunner",
    "BatchProgressDisplay",
    "BatchReponse",
    "BatchRequestItem",
    "BatchResponse",
    "EnqueuedLimits",
    "FailedPrediction",
    "LmdbLMCache",
    "RawResults",
    "ShardEvent",
    "ShardOutcome",
    "VLLMBatchClient",
    "abatch",
    "acancel_batch",
    "aget_batch_results",
    "aget_batch_status",
    "get_batch_adapter",
    "get_batch_client",
    "submit_batch_job",
    "supports_abatch",
]


def __getattr__(name: str):
    if name == "VLLMBatchClient":
        from modaic.batch.clients.vllm import VLLMBatchClient
        return VLLMBatchClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
