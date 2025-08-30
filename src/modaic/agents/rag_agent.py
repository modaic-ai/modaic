from modaic import PrecompiledConfig, PrecompiledAgent, Indexer
from .registry import builtin_agent, builtin_indexer, builtin_config
from typing import List
from modaic.context import ContextSchema

agent_name = "basic-rag"


@builtin_config(agent_name)
class RAGAgentConfig(PrecompiledConfig):
    def __init__(self):
        pass

    def forward(self, query: str):
        return "hello"


@builtin_indexer(agent_name)
class RAGIndexer(Indexer):
    def ingest(self, config: RAGAgentConfig, contexts: List[ContextSchema]):
        return "hello"


@builtin_agent(agent_name)
class RAGAgent(PrecompiledAgent):
    def __init__(self, config: RAGAgentConfig, indexer: RAGIndexer):
        super().__init__(config)
        self.indexer = indexer

    def forward(self, query: str):
        return "hello"
