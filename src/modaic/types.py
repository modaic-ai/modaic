# from abc import ABC, abstractmethod
from typing import List
from .context.types import Context


class Indexer:
    def __init__(self, *args, **kwargs):
        pass

    # @abstractmethod
    def ingest(self, contexts: List[Context], *args, **kwargs):
        pass
