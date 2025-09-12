import json
import os
import shutil
from pathlib import Path
from typing import Literal

import dspy
import pytest
import requests
from pydantic import Field

from modaic.hub import get_user_info
from modaic.precompiled import Indexer, PrecompiledAgent, PrecompiledConfig, Retriever
from tests.testing_utils import delete_agent_repo

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")
MODAIC_API_URL = os.getenv("MODAIC_API_URL") or "https://api.modaic.dev"


class Summarize(dspy.Signature):
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField(desc="Answer to the question, based on the passage")


class ExampleConfig(PrecompiledConfig):
    output_type: Literal["bool", "str"]
    lm: str = "openai/gpt-4o-mini"
    number: int = 1


class ExampleAgent(PrecompiledAgent):
    config: ExampleConfig

    def __init__(self, config: ExampleConfig, runtime_param: str, **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(Summarize)
        self.predictor.lm = dspy.LM("openai/gpt-4o-mini")
        self.runtime_param = runtime_param

    def forward(self, question: str, context: str) -> str:
        return self.predictor(question=question, context=context)


class AgentWRetreiverConfig(PrecompiledConfig):
    num_fetch: int
    lm: str = "openai/gpt-4o-mini"
    embedder: str = "openai/text-embedding-3-small"
    clients: dict = Field(default_factory=lambda: {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]})


class ExampleRetriever(Retriever):
    config: AgentWRetreiverConfig

    def __init__(self, config: AgentWRetreiverConfig, needed_param: str, **kwargs):
        super().__init__(config, **kwargs)
        self.embedder_name = config.embedder
        self.needed_param = needed_param

    def retrieve(self, query: str) -> str:
        return f"Retrieved {self.config.num_fetch} results for {query}"


class AgentWRetreiver(PrecompiledAgent):
    config: AgentWRetreiverConfig

    def __init__(self, config: AgentWRetreiverConfig, retriever: ExampleRetriever, **kwargs):
        super().__init__(config, retriever=retriever, **kwargs)
        self.lm = self.config.lm
        self.clients = self.config.clients

    def forward(self, query: str) -> str:
        return self.retriever.retrieve(query)


@pytest.fixture
def clean_folder():
    shutil.rmtree("tests/artifacts/temp/test_precompiled", ignore_errors=True)
    os.makedirs("tests/artifacts/temp/test_precompiled")
    return Path("tests/artifacts/temp/test_precompiled")


@pytest.fixture
def clean_modaic_cache():
    shutil.rmtree(os.environ["MODAIC_CACHE"], ignore_errors=True)
    return


@pytest.fixture
def hub_repo(clean_modaic_cache):
    if not MODAIC_TOKEN:
        pytest.skip("Skipping because MODAIC_TOKEN is not set")

    username = get_user_info(MODAIC_TOKEN)["login"]
    # delete the repo
    delete_agent_repo(username=username, agent_name="no-code-repo")

    return f"{username}/no-code-repo"


def test_init_subclass():
    with pytest.raises(ValueError):

        class BadAgent(PrecompiledAgent):
            def __init__(self, config: PrecompiledConfig, **kwargs):
                super().__init__(config, **kwargs)

            def forward(self, query: str) -> str:
                return "imma bad boy"

    with pytest.raises(ValueError):

        class BadRetriever(Retriever):
            def __init__(self, config: PrecompiledConfig, **kwargs):
                super().__init__(config, **kwargs)

            def retrieve(self, query: str) -> str:
                return "imma bad girl"

    with pytest.raises(ValueError):

        class BadIndexer(Indexer):
            def __init__(self, config: PrecompiledConfig, **kwargs):
                super().__init__(config, **kwargs)

            def ingest(self, contexts) -> None:
                return "imma bad indexer"

            def retrieve(self, query: str) -> str:
                return "imma bad indexer"

    # Still abstract, since no retrieve method is implemented
    class JustPassinThrough(Retriever):
        def __init__(self, config: PrecompiledConfig, **kwargs):
            super().__init__(config, **kwargs)

    # Still abstract, since no ingest method is implemented
    class JustPassinThroughIndexer(Indexer):
        def __init__(self, config: PrecompiledConfig, **kwargs):
            super().__init__(config, **kwargs)

        def retrieve(self, query: str) -> str:
            return "imma just pass through"


def test_precompiled_config_local(clean_folder):
    ExampleConfig(output_type="str").save_precompiled(clean_folder)
    assert os.path.exists(clean_folder / "config.json")
    assert len(os.listdir(clean_folder)) == 1
    loaded_config = ExampleConfig.from_precompiled(clean_folder)
    assert loaded_config.output_type == "str"
    assert loaded_config.lm == "openai/gpt-4o-mini"
    assert loaded_config.number == 1

    loaded_config = ExampleConfig.from_precompiled(clean_folder, lm="openai/gpt-4o", number=2)
    assert loaded_config.output_type == "str"
    assert loaded_config.lm == "openai/gpt-4o"
    assert loaded_config.number == 2


def test_precompiled_agent_local(clean_folder):
    ExampleAgent(ExampleConfig(output_type="str"), runtime_param="Hello").save_precompiled(clean_folder)
    assert os.path.exists(clean_folder / "config.json")
    assert os.path.exists(clean_folder / "agent.json")
    assert len(os.listdir(clean_folder)) == 2
    loaded_agent = ExampleAgent.from_precompiled(clean_folder, runtime_param="Hello")
    assert loaded_agent.runtime_param == "Hello"
    assert loaded_agent.config.output_type == "str"
    assert loaded_agent.config.lm == "openai/gpt-4o-mini"
    assert loaded_agent.config.number == 1

    loaded_agent = ExampleAgent.from_precompiled(
        clean_folder, runtime_param="wassuh", config_options={"lm": "openai/gpt-4o", "number": 2}
    )
    assert loaded_agent.config.output_type == "str"
    assert loaded_agent.config.lm == "openai/gpt-4o"
    assert loaded_agent.config.number == 2
    assert loaded_agent.runtime_param == "wassuh"


def test_precompiled_retriever_local(clean_folder):
    # Test retriever by itself
    ExampleRetriever(AgentWRetreiverConfig(num_fetch=10), needed_param="Hello").save_precompiled(clean_folder)
    assert os.path.exists(clean_folder / "config.json")
    assert len(os.listdir(clean_folder)) == 1
    loaded_retriever = ExampleRetriever.from_precompiled(clean_folder, needed_param="Goodbye")
    assert loaded_retriever.config.num_fetch == 10
    assert loaded_retriever.needed_param == "Goodbye"
    assert loaded_retriever.config.embedder == loaded_retriever.embedder_name == "openai/text-embedding-3-small"
    assert loaded_retriever.config.lm == "openai/gpt-4o-mini"

    loaded_retriever = ExampleRetriever.from_precompiled(
        clean_folder, needed_param="wassuhhhh", config_options={"num_fetch": 20}
    )
    assert loaded_retriever.config.num_fetch == 20
    assert loaded_retriever.needed_param == "wassuhhhh"
    assert loaded_retriever.config.embedder == loaded_retriever.embedder_name == "openai/text-embedding-3-small"
    assert loaded_retriever.config.lm == "openai/gpt-4o-mini"


def test_precompiled_agent_with_retriever_local(clean_folder):
    # Test agent with retriever
    config = AgentWRetreiverConfig(num_fetch=10)
    retriever = ExampleRetriever(config, needed_param="param required")
    agent = AgentWRetreiver(config, retriever)
    agent.save_precompiled(clean_folder)
    assert os.path.exists(clean_folder / "config.json")
    assert os.path.exists(clean_folder / "agent.json")
    assert len(os.listdir(clean_folder)) == 2
    loaded_retriever = ExampleRetriever.from_precompiled(clean_folder, needed_param="param required")
    loaded_agent = AgentWRetreiver.from_precompiled(clean_folder, retriever=loaded_retriever)
    assert loaded_retriever.config.num_fetch == loaded_agent.config.num_fetch == 10
    assert loaded_retriever.config.lm == loaded_agent.config.lm == "openai/gpt-4o-mini"
    assert loaded_retriever.config.embedder == loaded_agent.config.embedder == "openai/text-embedding-3-small"
    assert (
        loaded_retriever.config.clients
        == loaded_agent.config.clients
        == {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]}
    )
    assert loaded_retriever.needed_param == "param required"
    assert loaded_agent("my query") == "Retrieved 10 results for my query"

    config_options = {"num_fetch": 20}
    loaded_retriever = ExampleRetriever.from_precompiled(
        clean_folder, needed_param="param required2", config_options=config_options
    )
    loaded_agent = AgentWRetreiver.from_precompiled(
        clean_folder, retriever=loaded_retriever, config_options=config_options
    )
    assert loaded_retriever.config.num_fetch == loaded_agent.config.num_fetch == 20
    assert loaded_retriever.config.lm == loaded_agent.config.lm == "openai/gpt-4o-mini"
    assert loaded_retriever.config.embedder == loaded_agent.config.embedder == "openai/text-embedding-3-small"
    assert (
        loaded_retriever.config.clients
        == loaded_agent.config.clients
        == {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]}
    )
    assert loaded_retriever.needed_param == "param required2"
    assert loaded_retriever.retrieve("my query") == "Retrieved 20 results for my query"


# the following test only test with_code=True, with_code=False tests are done in test_auto_agent.py


def test_precompiled_agent_hub(hub_repo):
    ExampleAgent(ExampleConfig(output_type="str"), runtime_param="Hello").push_to_hub(hub_repo, with_code=False)
    repo_dir = Path(os.environ["MODAIC_CACHE"]) / "agents" / hub_repo

    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "agent.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 4
    loaded_agent = ExampleAgent.from_precompiled(
        hub_repo, runtime_param="wassuh", config_options={"lm": "openai/gpt-4o"}
    )
    assert loaded_agent.runtime_param == "wassuh"
    assert loaded_agent.config.lm == "openai/gpt-4o"
    assert loaded_agent.config.output_type == "str"
    assert loaded_agent.config.number == 1
    loaded_agent.push_to_hub(hub_repo, with_code=False)

    loaded_agent2 = ExampleAgent.from_precompiled(hub_repo, runtime_param="wassuh2", config_options={"number": 2})
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "agent.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 4
    assert loaded_agent2.runtime_param == "wassuh2"
    assert loaded_agent2.config.number == 2
    assert loaded_agent2.config.lm == "openai/gpt-4o"
    assert loaded_agent2.config.output_type == "str"
    loaded_agent2.push_to_hub(hub_repo, with_code=False)
    # now test with removing the local cache
    shutil.rmtree(repo_dir)
    loaded_agent3 = ExampleAgent.from_precompiled(
        hub_repo, runtime_param="wassuh3", config_options={"output_type": "bool"}
    )
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "agent.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 4
    assert loaded_agent3.runtime_param == "wassuh3"
    assert loaded_agent3.config.output_type == "bool"
    assert loaded_agent3.config.lm == "openai/gpt-4o"
    assert loaded_agent3.config.number == 2


def test_precompiled_retriever_hub(hub_repo):
    clients = {"openai": ["sama"]}
    ExampleRetriever(AgentWRetreiverConfig(num_fetch=10, clients=clients), needed_param="Hello").push_to_hub(
        hub_repo, with_code=False
    )
    repo_dir = Path(os.environ["MODAIC_CACHE"]) / "agents" / hub_repo
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 3
    loaded_retriever = ExampleRetriever.from_precompiled(
        hub_repo, needed_param="Goodbye", config_options={"num_fetch": 20}
    )
    assert loaded_retriever.config.num_fetch == 20
    assert loaded_retriever.needed_param == "Goodbye"
    assert loaded_retriever.config.embedder == loaded_retriever.embedder_name == "openai/text-embedding-3-small"
    assert loaded_retriever.config.lm == "openai/gpt-4o-mini"
    assert loaded_retriever.config.clients == clients
    assert loaded_retriever.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_retriever.push_to_hub(hub_repo, with_code=False)
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 3

    loaded_retriever2 = ExampleRetriever.from_precompiled(
        hub_repo, needed_param="Goodbye2", config_options={"lm": "openai/gpt-4o"}
    )
    assert loaded_retriever2.config.lm == "openai/gpt-4o"
    assert loaded_retriever2.needed_param == "Goodbye2"
    assert loaded_retriever2.config.num_fetch == 20
    assert loaded_retriever2.config.embedder == loaded_retriever2.embedder_name == "openai/text-embedding-3-small"
    assert loaded_retriever2.config.clients == clients
    assert loaded_retriever2.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_retriever2.push_to_hub(hub_repo, with_code=False)
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 3

    shutil.rmtree(repo_dir)
    loaded_retriever3 = ExampleRetriever.from_precompiled(
        hub_repo, needed_param="Goodbye3", config_options={"clients": {"openai": ["sama2"]}}
    )
    assert loaded_retriever3.config.clients == {"openai": ["sama2"]}
    assert loaded_retriever3.needed_param == "Goodbye3"
    assert loaded_retriever3.config.num_fetch == 20
    assert loaded_retriever3.config.embedder == loaded_retriever3.embedder_name == "openai/text-embedding-3-small"
    assert loaded_retriever3.config.lm == "openai/gpt-4o"
    assert loaded_retriever3.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_retriever3.push_to_hub(hub_repo, with_code=False)


def test_precompiled_agent_with_retriever_hub(hub_repo):
    clients = {"openai": ["sama"]}
    config = AgentWRetreiverConfig(num_fetch=10, clients=clients)
    retriever = ExampleRetriever(config, needed_param="Hello")
    agent = AgentWRetreiver(config, retriever)
    agent.push_to_hub(hub_repo, with_code=False)
    repo_dir = Path(os.environ["MODAIC_CACHE"]) / "agents" / hub_repo
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "agent.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 4

    config_options = {"num_fetch": 20}
    loaded_retriever = ExampleRetriever.from_precompiled(
        hub_repo, needed_param="Goodbye", config_options=config_options
    )
    loaded_agent = AgentWRetreiver.from_precompiled(hub_repo, retriever=loaded_retriever, config_options=config_options)
    assert loaded_retriever.config.num_fetch == loaded_agent.config.num_fetch == 20
    assert loaded_retriever.config.clients == loaded_agent.config.clients == clients
    assert loaded_retriever.config.lm == loaded_agent.config.lm == "openai/gpt-4o-mini"
    assert (
        loaded_retriever.config.embedder
        == loaded_agent.config.embedder
        == loaded_retriever.embedder_name
        == "openai/text-embedding-3-small"
    )
    assert loaded_retriever.needed_param == "Goodbye"
    assert loaded_retriever.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_agent.push_to_hub(hub_repo, with_code=False)
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "agent.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 4

    config_options = {"lm": "openai/gpt-4o"}
    loaded_retriever2 = ExampleRetriever.from_precompiled(
        hub_repo, needed_param="Goodbye2", config_options=config_options
    )
    loaded_agent2 = AgentWRetreiver.from_precompiled(
        hub_repo, retriever=loaded_retriever2, config_options=config_options
    )
    assert loaded_retriever2.config.lm == loaded_agent2.config.lm == "openai/gpt-4o"
    assert loaded_retriever2.config.num_fetch == loaded_agent2.config.num_fetch == 20
    assert loaded_retriever2.config.clients == loaded_agent2.config.clients == clients
    assert loaded_retriever2.needed_param == "Goodbye2"
    assert loaded_retriever2.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_agent2.push_to_hub(hub_repo, with_code=False)
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "agent.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 4

    shutil.rmtree(repo_dir)
    config_options = {"clients": {"openai": ["sama3"]}}
    loaded_retriever3 = ExampleRetriever.from_precompiled(
        hub_repo, needed_param="Goodbye3", config_options=config_options
    )
    loaded_agent3 = AgentWRetreiver.from_precompiled(
        hub_repo, retriever=loaded_retriever3, config_options=config_options
    )
    assert loaded_retriever3.config.clients == loaded_agent3.config.clients == {"openai": ["sama3"]}
    assert loaded_retriever3.config.num_fetch == loaded_agent3.config.num_fetch == 20
    assert loaded_retriever3.needed_param == "Goodbye3"
    assert loaded_retriever3.config.lm == loaded_agent3.config.lm == "openai/gpt-4o"
    assert loaded_retriever3.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_agent3.push_to_hub(hub_repo, with_code=False)


def test_unauthorized_push_to_hub():
    pass
