from dataclasses import dataclass
from modaic import PrecompiledConfig, PrecompiledAgent, Indexer
import example_utils.utils
import example_utils.nested_example_utils.more

# import utils
# import torage.context_store
import test_module_resolution
import my_module1


@dataclass
class MyConfig(PrecompiledConfig):
    agent_type = "MyAgent"
    indexer_type = "MyIndexer"

    hello: str = "world"
    bye: int = 123


class MyAgent(PrecompiledAgent):
    config_class = MyConfig

    def __init__(self, config: MyConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(self, **kwargs):
        return "hello"


class MyIndexer(Indexer):
    config_class = MyConfig

    def __init__(self, config: MyConfig, **kwargs):
        super().__init__(config, **kwargs)

    def ingest(self, **kwargs):
        return "hello"

    def retrieve(self, **kwargs):
        return "hello"


cfg = MyConfig(hello="hello", bye=456)
indexer = MyIndexer(cfg)
agent = MyAgent(cfg, indexer=indexer)

agent.push_to_hub("modaic-ai/hub_test")
