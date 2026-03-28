from importlib import import_module

from .base import BatchClient

__all__ = [
    "BatchClient",
    "OpenAIBatchClient",
    "AzureBatchClient",
    "TogetherBatchClient",
    "FireworksBatchClient",
    "AnthropicBatchClient",
    "VertexAIBatchClient",
    "VLLMBatchClient",
    "VLLMBatchClient2",
]

_LAZY_IMPORTS = {
    "OpenAIBatchClient": ("modaic.batch.clients.openai", "OpenAIBatchClient"),
    "AzureBatchClient": ("modaic.batch.clients.azure", "AzureBatchClient"),
    "TogetherBatchClient": ("modaic.batch.clients.together", "TogetherBatchClient"),
    "FireworksBatchClient": ("modaic.batch.clients.fireworks", "FireworksBatchClient"),
    "AnthropicBatchClient": ("modaic.batch.clients.anthropic", "AnthropicBatchClient"),
    "VertexAIBatchClient": ("modaic.batch.clients.vertex", "VertexAIBatchClient"),
    "VLLMBatchClient": ("modaic.batch.clients.vllm", "VLLMBatchClient"),
    "VLLMBatchClient2": ("modaic.batch.clients.vllm", "VLLMBatchClient2"),
}


def __getattr__(name: str):
    if name == "BatchClient":
        return BatchClient
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
