from abc import ABC, abstractmethod
from typing import List, Any
from ..context.base import Context, Source, SourceType
import os
import pickle
import uuid


class ContextStorage(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def add(self, contexts: List[Context], **kwargs):
        pass

    @abstractmethod
    def get(self, *args, **kwargs) -> Any:
        pass


class PickleContextStorage(ContextStorage):
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def add(self, contexts: List[Context], **kwargs) -> List[Context]:
        for context in contexts:
            context_id = uuid.uuid4()
            file_name = f"{context_id}.pkl"
            with open(os.path.join(self.directory, file_name), "wb") as f:
                pickle.dump(context, f)
            new_source = Source(origin=file_name, type=SourceType.LOCAL_PATH)
            context.set_source(new_source)
            context.metadata["context_id"] = context_id
        return contexts

    def get(self, source: Source) -> Context:
        with open(os.path.join(self.directory, source.origin), "rb") as f:
            return pickle.load(f)
