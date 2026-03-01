from modaic_client import (  # noqa: F401
    Arbiter,
    ModaicClient,
    configure,
    configure_modaic_client,
    exceptions,
    get_modaic_client,
    settings,
    track,
)

from .auto import AutoAgent, AutoConfig, AutoProgram, AutoRetriever
from .precompiled import Indexer, PrecompiledAgent, PrecompiledConfig, PrecompiledProgram, Retriever
from .programs import Predict, PredictConfig  # noqa: F401
from .safe_lm import SafeLM
from .serializers import SerializableSignature

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
    "SerializableSignature",
    "Arbiter",
    "ModaicClient",
    "get_modaic_client",
    "SafeLM",
]
