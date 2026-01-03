from .auto import AutoAgent, AutoConfig, AutoProgram, AutoRetriever
from .observability import Trackable, configure, track, track_modaic_obj
from .precompiled import Indexer, PrecompiledAgent, PrecompiledConfig, PrecompiledProgram, Retriever
from .programs import Predict, PredictConfig  # noqa: F401
from .serializers import SerializableLM, SerializableSignature

__all__ = [
    # New preferred names
    "AutoProgram",
    "PrecompiledProgram",
    # Deprecated names (kept for backward compatibility)
    "AutoAgent",
    "PrecompiledAgent",
    # Other exports
    "AutoConfig",
    "AutoRetriever",
    "Retriever",
    "Indexer",
    "PrecompiledConfig",
    "configure",
    "track",
    "Trackable",
    "track_modaic_obj",
    "SerializableSignature",
    "SerializableLM",
]
_configured = False
