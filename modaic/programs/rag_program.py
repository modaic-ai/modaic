from typing import Any

from modaic import Indexer, PrecompiledConfig, PrecompiledProgram

from .registry import builtin_config, builtin_indexer, builtin_program

program_name = "basic-rag"


@builtin_config(program_name)
class RAGProgramConfig(PrecompiledConfig):
    def __init__(self):
        pass

    def forward(self, query: str) -> str:
        return "hello"


@builtin_indexer(program_name)
class RAGIndexer(Indexer):
    def __init__(self, config: RAGProgramConfig):
        super().__init__(config)

    def index(self, contents: Any):
        pass


@builtin_program(program_name)
class RAGProgram(PrecompiledProgram):
    def __init__(self, config: RAGProgramConfig, indexer: RAGIndexer):
        super().__init__(config)
        self.indexer = indexer

    def forward(self, query: str) -> str:
        return "hello"
