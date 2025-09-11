import pathlib
from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional, Type

from modaic.context.base import Context
from modaic.precompiled_agent import PrecompiledConfig

from ..observability import Trackable, track_modaic_obj
from ..precompiled_agent import Retriever


class Indexer(Retriever):
    @abstractmethod
    def ingest(self, contexts: List[Context], **kwargs):
        pass
