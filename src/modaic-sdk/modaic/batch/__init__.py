from modaic.batch.batch import abatch, acancel_batch, aget_batch_results, submit_batch_job, supports_abatch
from modaic.batch.lmdb_cache import LmdbLMCache
from modaic.batch.types import ABatchResult, ABatchRow, FailedPrediction

__all__ = [
    "abatch",
    "acancel_batch",
    "aget_batch_results",
    "submit_batch_job",
    "supports_abatch",
    "ABatchResult",
    "ABatchRow",
    "FailedPrediction",
    "LmdbLMCache",
    "VLLMBatchClient",
]


def __getattr__(name: str):
    if name == "VLLMBatchClient":
        from modaic.batch.clients.vllm import VLLMBatchClient

        return VLLMBatchClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
