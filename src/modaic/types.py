from abc import ABC, abstractmethod
from typing import List
from .context.types import Context


class Indexer(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def add(self, context: Context, *args, **kwargs):
        pass
    
    def add_all(self, contexts: List[Context], *args, **kwargs):
        for context in contexts:
            self.add(context)
    
    @abstractmethod
    def delete(self, *args, **kwargs):
        pass
    
    def delete_all(self, *args, **kwargs):
        pass