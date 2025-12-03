from .auto import AutoAgent, AutoConfig, AutoProgram, AutoRetriever
from .indexing import Embedder
from .observability import Trackable, configure, track, track_modaic_obj
from .precompiled import Indexer, PrecompiledAgent, PrecompiledConfig, PrecompiledProgram, Retriever
from .query_language import AND, OR, Condition, Prop, Value, parse_modaic_filter

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
    "Embedder",
    "configure",
    "track",
    "Trackable",
    "track_modaic_obj",
    "AND",
    "OR",
    "Prop",
    "Value",
    "parse_modaic_filter",
    "Condition",
]
_configured = False
