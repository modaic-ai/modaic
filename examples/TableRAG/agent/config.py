from modaic.precompiled_agent import PrecompiledConfig
from dataclasses import dataclass


@dataclass
class TableRAGConfig(PrecompiledConfig):
    k_recall: int = 50
    k_rerank: int = 5
