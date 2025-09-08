import pathlib
from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional, Type

from modaic.context.base import Context
from modaic.precompiled_agent import PrecompiledConfig

from ..precompiled_agent import _push_to_hub


class Retriever(ABC):
    config_class: ClassVar[Type[PrecompiledConfig]]

    def __init__(self, config: PrecompiledConfig, **kwargs):
        self.config = config
        assert isinstance(config, self.config_class), f"Config must be an instance of {self.config_class.__name__}"

    @abstractmethod
    def retrieve(self, query: str, **kwargs):
        pass

    def save_precompiled(self, path: str) -> None:
        """
        Saves the indexer configuration to the given path.

        Args:
          path: The path to save the indexer configuration and auto classes mapping.
        """
        path_obj = pathlib.Path(path)
        extra_auto_classes = {"AutoRetriever": self}
        self.config.save_precompiled(path_obj, extra_auto_classes)

    def push_to_hub(
        self,
        repo_path: str,
        access_token: Optional[str] = None,
        commit_message="(no commit message)",
    ) -> None:
        """
        Pushes the indexer and the config to the given repo_path.

        Args:
            repo_path: The path on Modaic hub to save the agent and config to.
            access_token: Your Modaic access token.
            commit_message: The commit message to use when pushing to the hub.
        """
        _push_to_hub(self, repo_path, access_token, commit_message)


class Indexer(Retriever):
    @abstractmethod
    def ingest(self, contexts: List[Context], **kwargs):
        pass
