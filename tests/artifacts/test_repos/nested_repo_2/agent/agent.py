from modaic import PrecompiledAgent
from nested_repo_2.agent.config import AgentWRetreiverConfig
from nested_repo_2.agent.retriever import ExampleRetriever
from nested_repo_2.agent.tools.used_tool import use_this_tool
from nested_repo_2.agent.utils.used_util import helpful_util


class AgentWRetreiver(PrecompiledAgent):
    config: AgentWRetreiverConfig

    def __init__(self, config: AgentWRetreiverConfig, retriever: ExampleRetriever, **kwargs):
        super().__init__(config, retriever=retriever, **kwargs)
        self.lm = self.config.lm
        self.clients = self.config.clients

    def forward(self, query: str) -> str:
        return self.retriever.retrieve(query)
