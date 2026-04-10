from importlib import import_module

__all__ = [
    "AutoProgram",
    "PrecompiledProgram",
    "AutoAgent",
    "PrecompiledAgent",
    "AutoConfig",
    "Retriever",
    "Indexer",
    "PrecompiledConfig",
    "configure",
    "track",
    "SerializableSignature",
    "Arbiter",
    "ModaicClient",
    "get_modaic_client",
    "SafeLM",
    "Predict",
    "PredictConfig",
    "configure_modaic_client",
    "exceptions",
    "settings",
    "Enum",
]

_LAZY_IMPORTS = {
    "Arbiter": ("modaic_client", "Arbiter"),
    "ModaicClient": ("modaic_client", "ModaicClient"),
    "configure": ("modaic_client", "configure"),
    "configure_modaic_client": ("modaic_client", "configure_modaic_client"),
    "exceptions": ("modaic_client", "exceptions"),
    "get_modaic_client": ("modaic_client", "get_modaic_client"),
    "settings": ("modaic_client", "settings"),
    "track": ("modaic_client", "track"),
    "AutoAgent": ("modaic.auto", "AutoAgent"),
    "AutoConfig": ("modaic.auto", "AutoConfig"),
    "AutoProgram": ("modaic.auto", "AutoProgram"),
    "Indexer": ("modaic.precompiled", "Indexer"),
    "PrecompiledAgent": ("modaic.precompiled", "PrecompiledAgent"),
    "PrecompiledConfig": ("modaic.precompiled", "PrecompiledConfig"),
    "PrecompiledProgram": ("modaic.precompiled", "PrecompiledProgram"),
    "Retriever": ("modaic.precompiled", "Retriever"),
    "Predict": ("modaic.programs", "Predict"),
    "PredictConfig": ("modaic.programs", "PredictConfig"),
    "SafeLM": ("modaic.safe_lm", "SafeLM"),
    "SerializableSignature": ("modaic.serializers", "SerializableSignature"),
    "Enum": ("modaic.types", "Enum"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
