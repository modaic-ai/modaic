from importlib import import_module

__all__ = [
    "abatch",
    "acancel_batch",
    "aget_batch_results",
    "submit_batch_job",
    "supports_abatch",
    "FailedPrediction",
    "ModalBatchClient",
]

_LAZY_IMPORTS = {
    "abatch": ("modaic.batch.batch", "abatch"),
    "acancel_batch": ("modaic.batch.batch", "acancel_batch"),
    "aget_batch_results": ("modaic.batch.batch", "aget_batch_results"),
    "submit_batch_job": ("modaic.batch.batch", "submit_batch_job"),
    "supports_abatch": ("modaic.batch.batch", "supports_abatch"),
    "FailedPrediction": ("modaic.batch.types", "FailedPrediction"),
    "ModalBatchClient": ("modaic.batch.modal_client", "ModalBatchClient"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
