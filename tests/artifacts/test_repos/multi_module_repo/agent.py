from typing import TYPE_CHECKING

from modaic import PrecompiledAgent

if TYPE_CHECKING:
    from multi_module_repo.retriever import ExampleRetriever
from multi_module_repo.api import some_api_func
from pydantic import Field

from modaic import PrecompiledConfig


class AgentWRetreiverConfig(PrecompiledConfig):
    num_fetch: int
    lm: str = "openai/gpt-4o-mini"
    embedder: str = "openai/text-embedding-3-small"
    clients: dict = Field(default_factory=lambda: {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]})


class AgentWRetreiver(PrecompiledAgent):
    config: AgentWRetreiverConfig

    def __init__(self, config: AgentWRetreiverConfig, retriever: "ExampleRetriever", **kwargs):
        super().__init__(config, retriever=retriever, **kwargs)
        self.lm = self.config.lm
        self.clients = self.config.clients

    def forward(self, query: str) -> str:
        return self.retriever.retrieve(query)
