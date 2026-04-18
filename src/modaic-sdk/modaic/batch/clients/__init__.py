"""Batch provider clients.

Concrete implementations live in submodules (for example ``modaic.batch.clients.openai``).
This package does not import them at load time so optional dependencies stay lazy.
``get_batch_client`` in :mod:`modaic.batch.batch` imports only the submodule for
the selected provider.

For backwards compatibility, attribute access on ``modaic.batch.clients`` resolves
names lazily via :func:`__getattr__`.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AnthropicBatchClient",
    "AzureBatchClient",
    "BatchClient",
    "FireworksBatchClient",
    "OpenAIBatchClient",
    "TogetherBatchClient",
    "VertexAIBatchClient",
    "VLLMBatchClient",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "BatchClient": (".base", "BatchClient"),
    "OpenAIBatchClient": (".openai", "OpenAIBatchClient"),
    "AzureBatchClient": (".azure", "AzureBatchClient"),
    "TogetherBatchClient": (".together", "TogetherBatchClient"),
    "FireworksBatchClient": (".fireworks", "FireworksBatchClient"),
    "AnthropicBatchClient": (".anthropic", "AnthropicBatchClient"),
    "VertexAIBatchClient": (".vertex", "VertexAIBatchClient"),
    "VLLMBatchClient": (".vllm", "VLLMBatchClient"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submod, attr = _EXPORTS[name]
    module = import_module(submod, __name__)
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)
