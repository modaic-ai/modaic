from dataclasses import dataclass

from modaic.precompiled_agent import PrecompiledConfig


class TableRAGConfig(PrecompiledConfig):
    k_recall: int = 50
    k_rerank: int = 5
