from modaic.batch.batch import (
    BatchJobHandle,
    ShardInfo,
    abatch,
    acancel_batch,
    aget_batch_results,
    aget_batch_status,
    get_batch_adapter,
    get_batch_client,
    plan_shards,
    plan_shards_pre_rendered,
    submit_batch_job,
    submit_shard,
    submit_shard_pre_rendered,
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
    "ShardInfo",
    "ShardOutcome",
    "VLLMBatchClient",
    "abatch",
    "acancel_batch",
    "aget_batch_results",
    "aget_batch_status",
    "get_batch_adapter",
    "get_batch_client",
    "plan_shards",
    "plan_shards_pre_rendered",
    "submit_batch_job",
    "submit_shard",
    "submit_shard_pre_rendered",
    "supports_abatch",
]


def __getattr__(name: str):
    if name == "VLLMBatchClient":
        from modaic.batch.clients.vllm import VLLMBatchClient
        return VLLMBatchClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
