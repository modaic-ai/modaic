from .auto import AutoAgent, AutoConfig, AutoProgram, AutoRetriever
from .observability import Trackable, configure, track, track_modaic_obj
from .precompiled import Indexer, PrecompiledAgent, PrecompiledConfig, PrecompiledProgram, Retriever

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
]
_configured = False
