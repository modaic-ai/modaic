import os
import pickle
import uuid
from typing import List

from ..context.base import Context


class ContextPickleStore:
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def add(self, contexts: List[Context], **kwargs) -> List[Context]:
        for context in contexts:
            context_id = uuid.uuid4()
            file_name = f"{context_id}.pkl"
            with open(os.path.join(self.directory, file_name), "wb") as f:
                pickle.dump(context, f)
            context.metadata["context_id"] = context_id
        return contexts

    def get(self, source: str) -> Context:
        with open(os.path.join(self.directory, source), "rb") as f:
            return pickle.load(f)
