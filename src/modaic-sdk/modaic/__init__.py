import modaic_client as client
from modaic_client import (
    Arbiter,
    ModaicClient,
    configure,
    configure_modaic_client,
    exceptions,
    get_modaic_client,
    settings,
    track,
)

from modaic import _litellm_provider  # noqa: F401  (registers the `modaic/` litellm provider)
from modaic.auto import AutoAgent, AutoConfig, AutoProgram
from modaic.precompiled import Indexer, PrecompiledAgent, PrecompiledConfig, PrecompiledProgram, Retriever
from modaic.programs import Predict, PredictConfig
from modaic.safe_lm import SafeLM
from modaic.serializers import SerializableSignature
from modaic.types import Enum

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
