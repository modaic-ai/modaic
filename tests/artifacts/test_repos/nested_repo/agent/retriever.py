from modaic import Retriever
from nested_repo.agent.config import AgentWRetreiverConfig
from nested_repo.agent.tools.jira.jira_api_tools import call_jira_api
from nested_repo.agent.utils.used import random_util


class ExampleRetriever(Retriever):
    config: AgentWRetreiverConfig

    def __init__(self, config: AgentWRetreiverConfig, needed_param: str, **kwargs):
        super().__init__(config, **kwargs)
        self.embedder_name = config.embedder
        self.needed_param = needed_param

    def retrieve(self, query: str) -> str:
        return f"Retrieved {self.config.num_fetch} results for {query}"
