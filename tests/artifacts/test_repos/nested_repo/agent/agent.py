# from nested_repo.top import top_level_function
from modaic import PrecompiledAgent
from nested_repo.agent.config import AgentWRetreiverConfig
from nested_repo.agent.retriever import ExampleRetriever
from nested_repo.agent.tools.google.google_search import search_google
from nested_repo.agent.utils.used import random_util


class AgentWRetreiver(PrecompiledAgent):
    config: AgentWRetreiverConfig

    def __init__(self, config: AgentWRetreiverConfig, retriever: ExampleRetriever, **kwargs):
        super().__init__(config, retriever=retriever, **kwargs)
        self.lm = self.config.lm
        self.clients = self.config.clients

    def forward(self, query: str) -> str:
        return self.retriever.retrieve(query)
