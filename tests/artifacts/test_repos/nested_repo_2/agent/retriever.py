from agent.config import AgentWRetreiverConfig
from api.used_api import use_this_api

from modaic import Retriever


class ExampleRetriever(Retriever):
    config: AgentWRetreiverConfig

    def __init__(self, config: AgentWRetreiverConfig, needed_param: str, **kwargs):
        super().__init__(config, **kwargs)
        self.embedder_name = config.embedder
        self.needed_param = needed_param

    def retrieve(self, query: str) -> str:
        return f"Retrieved {self.config.num_fetch} results for {query}"
