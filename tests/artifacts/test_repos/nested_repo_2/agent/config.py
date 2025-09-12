from pydantic import Field

from modaic import PrecompiledConfig


class AgentWRetreiverConfig(PrecompiledConfig):
    num_fetch: int
    lm: str = "openai/gpt-4o-mini"
    embedder: str = "openai/text-embedding-3-small"
    clients: dict = Field(default_factory=lambda: {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]})
