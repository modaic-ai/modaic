import sys

from modaic import PrecompiledAgent

from .config import AgentWRetreiverConfig
from .retriever import ExampleRetriever
from .tools.google.google_search import search_google  # noqa: F401
from .utils.used import random_util  # noqa: F401


class AgentWRetreiver(PrecompiledAgent):
    config: AgentWRetreiverConfig

    def __init__(self, config: AgentWRetreiverConfig, retriever: ExampleRetriever, **kwargs):
        super().__init__(config, retriever=retriever, **kwargs)
        self.lm = self.config.lm
        self.clients = self.config.clients

    def forward(self, query: str) -> str:
        return self.retriever.retrieve(query)


if __name__ == "__main__":
    username = sys.argv[1]  # ← first arg after script name (username)
    config = AgentWRetreiverConfig(num_fetch=1)
    retriever = ExampleRetriever(config, needed_param="hi")
    agent = AgentWRetreiver(config, retriever=retriever)
    repo_path = f"{username}/nested_repo_3"
    agent.push_to_hub(repo_path, with_code=True)
