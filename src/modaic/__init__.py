from .auto_agent import AutoAgent, AutoConfig, AutoRetriever
from .indexing import Embedder
from .observability import Trackable, configure, track, track_modaic_obj
from .precompiled import Indexer, PrecompiledAgent, PrecompiledConfig, Retriever
from .query_language import AND, OR, Condition, Prop, Value, parse_modaic_filter

__all__ = [
    "AutoAgent",
    "AutoConfig",
    "AutoRetriever",
    "Retriever",
    "Indexer",
    "PrecompiledAgent",
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
]
_configured = False


def _auto_configure():
    global _configured
    if not _configured:
        print("auto configuring")
        configure()
        _configured = True


_auto_configure()
