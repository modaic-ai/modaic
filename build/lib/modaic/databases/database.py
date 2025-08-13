from dataclasses import dataclass
from ..context.base import Context, ContextSchema
from abc import ABC, abstractmethod


@dataclass
class ContextDatabaseConfig:
    pass


class ContextDatabase(ABC):
    """
    A database that can store context objects.
    """

    def __init__(self, config: ContextDatabaseConfig, **kwargs):
        self.config = config

    @abstractmethod
    def add_item(self, item: Context):
        pass

    @abstractmethod
    def get_item(self, item: Context):
        pass

    @abstractmethod
    def delete_item(self, item: Context):
        pass

    @abstractmethod
    def update_item(self, item: Context):
        pass


@dataclass
class RAGDatabaseConfig:
    pass


class RAGDatabase(ABC):
    """
    A database used for RAG
    """

    def __init__(self, config: RAGDatabaseConfig, **kwargs):
        self.config = config

    @abstractmethod
    def search(self, *args, **kwargs) -> ContextSchema:
        pass
