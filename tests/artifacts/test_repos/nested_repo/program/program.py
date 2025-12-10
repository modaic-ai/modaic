from modaic import PrecompiledProgram

from .config import AgentWRetreiverConfig
from .retriever import ExampleRetriever
from .tools.google.google_search import search_google  # noqa: F401
from .utils.used import random_util  # noqa: F401


class AgentWRetreiver(PrecompiledProgram):
    config: AgentWRetreiverConfig

    def __init__(self, config: AgentWRetreiverConfig, retriever: ExampleRetriever, **kwargs):
        super().__init__(config, retriever=retriever, **kwargs)
        self.lm = self.config.lm
        self.clients = self.config.clients

    def forward(self, query: str) -> str:
        return self.retriever.retrieve(query)
