import json
import os
import shutil
from pathlib import Path
from typing import List, Literal, Type

import dspy
import pytest
from pydantic import Field

from modaic.hub import PROGRAM_CACHE, MODAIC_CACHE, get_user_info
from modaic.precompiled import Indexer, PrecompiledProgram, PrecompiledConfig, Retriever
from tests.testing_utils import delete_program_repo

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")
MODAIC_API_URL = os.getenv("MODAIC_API_URL") or "https://api.modaic.dev"


class Summarize(dspy.Signature):
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Answer to the question, based on the passage")


class ExampleConfig(PrecompiledConfig):
    output_type: Literal["bool", "str"]
    lm: str = "openai/gpt-4o-mini"
    number: int = 1


class ExampleProgram(PrecompiledProgram):
    config: ExampleConfig

    def __init__(self, config: ExampleConfig, runtime_param: str, **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(Summarize)
        self.predictor.lm = dspy.LM("openai/gpt-4o-mini")
        self.runtime_param = runtime_param

    def forward(self, question: str, context: str) -> str:
        return self.predictor(question=question, context=context)


class ProgramWRetreiverConfig(PrecompiledConfig):
    num_fetch: int
    lm: str = "openai/gpt-4o-mini"
    embedder: str = "openai/text-embedding-3-small"
    clients: dict = Field(default_factory=lambda: {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]})


class ExampleRetriever(Retriever):
    config: ProgramWRetreiverConfig

    def __init__(self, config: ProgramWRetreiverConfig, needed_param: str, **kwargs):
        super().__init__(config, **kwargs)
        self.embedder_name = config.embedder
        self.needed_param = needed_param

    def retrieve(self, query: str) -> str:
        return f"Retrieved {self.config.num_fetch} results for {query}"


class ProgramWRetreiver(PrecompiledProgram):
    config: ProgramWRetreiverConfig

    def __init__(self, config: ProgramWRetreiverConfig, retriever: ExampleRetriever, **kwargs):
        super().__init__(config, retriever=retriever, **kwargs)
        self.lm = self.config.lm
        self.clients = self.config.clients

    def forward(self, query: str) -> str:
        return self.retriever.retrieve(query)


@pytest.fixture
def clean_folder() -> Path:
    shutil.rmtree("tests/artifacts/temp/test_precompiled", ignore_errors=True)
    os.makedirs("tests/artifacts/temp/test_precompiled")
    return Path("tests/artifacts/temp/test_precompiled")


@pytest.fixture
def clean_modaic_cache() -> Path:
    shutil.rmtree(MODAIC_CACHE, ignore_errors=True)
    return MODAIC_CACHE


@pytest.fixture
def hub_repo(clean_modaic_cache: Path) -> str:
    if not MODAIC_TOKEN:
        pytest.skip("Skipping because MODAIC_TOKEN is not set")

    username = get_user_info(MODAIC_TOKEN)["login"]
    # delete the repo
    delete_program_repo(username=username, program_name="no-code-repo")

    return f"{username}/no-code-repo"


# TODO: add run on __call__ to tests
def test_init_subclass():
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


def test_precompiled_config_local(clean_folder: Path):
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


def test_precompiled_program_local(clean_folder: Path):
    ExampleProgram(ExampleConfig(output_type="str"), runtime_param="Hello").save_precompiled(clean_folder)
    assert os.path.exists(clean_folder / "config.json")
    assert os.path.exists(clean_folder / "program.json")
    assert len(os.listdir(clean_folder)) == 2
    loaded_program = ExampleProgram.from_precompiled(clean_folder, runtime_param="Hello")
    assert loaded_program.runtime_param == "Hello"
    assert loaded_program.config.output_type == "str"
    assert loaded_program.config.lm == "openai/gpt-4o-mini"
    assert loaded_program.config.number == 1

    loaded_program = ExampleProgram.from_precompiled(
        clean_folder, runtime_param="wassuh", config={"lm": "openai/gpt-4o", "number": 2}
    )
    assert loaded_program.config.output_type == "str"
    assert loaded_program.config.lm == "openai/gpt-4o"
    assert loaded_program.config.number == 2
    assert loaded_program.runtime_param == "wassuh"
    loaded_program(question="what is the meaning of life?", context="The meaning of life is 42")


def test_precompiled_retriever_local(clean_folder: Path):
    # Test retriever by itself
    ExampleRetriever(ProgramWRetreiverConfig(num_fetch=10), needed_param="Hello").save_precompiled(clean_folder)
    assert os.path.exists(clean_folder / "config.json")
    assert len(os.listdir(clean_folder)) == 1
    loaded_retriever = ExampleRetriever.from_precompiled(clean_folder, needed_param="Goodbye")
    assert loaded_retriever.config.num_fetch == 10
    assert loaded_retriever.needed_param == "Goodbye"
    assert loaded_retriever.config.embedder == loaded_retriever.embedder_name == "openai/text-embedding-3-small"
    assert loaded_retriever.config.lm == "openai/gpt-4o-mini"

    loaded_retriever = ExampleRetriever.from_precompiled(
        clean_folder, needed_param="wassuhhhh", config={"num_fetch": 20}
    )
    assert loaded_retriever.config.num_fetch == 20
    assert loaded_retriever.needed_param == "wassuhhhh"
    assert loaded_retriever.config.embedder == loaded_retriever.embedder_name == "openai/text-embedding-3-small"
    assert loaded_retriever.config.lm == "openai/gpt-4o-mini"


def test_precompiled_program_with_retriever_local(clean_folder: Path):
    # Test program with retriever
    config = ProgramWRetreiverConfig(num_fetch=10)
    retriever = ExampleRetriever(config, needed_param="param required")
    program = ProgramWRetreiver(config, retriever)
    program.save_precompiled(clean_folder)
    assert os.path.exists(clean_folder / "config.json")
    assert os.path.exists(clean_folder / "program.json")
    assert len(os.listdir(clean_folder)) == 2
    loaded_retriever = ExampleRetriever.from_precompiled(clean_folder, needed_param="param required")
    loaded_program = ProgramWRetreiver.from_precompiled(clean_folder, retriever=loaded_retriever)
    assert loaded_retriever.config.num_fetch == loaded_program.config.num_fetch == 10
    assert loaded_retriever.config.lm == loaded_program.config.lm == "openai/gpt-4o-mini"
    assert loaded_retriever.config.embedder == loaded_program.config.embedder == "openai/text-embedding-3-small"
    assert (
        loaded_retriever.config.clients
        == loaded_program.config.clients
        == {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]}
    )
    assert loaded_retriever.needed_param == "param required"
    assert loaded_program("my query") == "Retrieved 10 results for my query"

    config = {"num_fetch": 20}
    loaded_retriever = ExampleRetriever.from_precompiled(clean_folder, needed_param="param required2", config=config)
    loaded_program = ProgramWRetreiver.from_precompiled(clean_folder, retriever=loaded_retriever, config=config)
    assert loaded_retriever.config.num_fetch == loaded_program.config.num_fetch == 20
    assert loaded_retriever.config.lm == loaded_program.config.lm == "openai/gpt-4o-mini"
    assert loaded_retriever.config.embedder == loaded_program.config.embedder == "openai/text-embedding-3-small"
    assert (
        loaded_retriever.config.clients
        == loaded_program.config.clients
        == {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]}
    )
    assert loaded_retriever.needed_param == "param required2"
    assert loaded_retriever.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_program(query="my query")


# the following test only test with_code=True, with_code=False tests are done in test_auto.py


def test_precompiled_program_hub(hub_repo: str):
    ExampleProgram(ExampleConfig(output_type="str"), runtime_param="Hello").push_to_hub(hub_repo, with_code=False)
    temp_dir = Path(MODAIC_CACHE) / "temp" / hub_repo
    repo_dir = Path(PROGRAM_CACHE) / hub_repo

    assert os.path.exists(temp_dir / "config.json")
    assert os.path.exists(temp_dir / "program.json")
    assert os.path.exists(temp_dir / "README.md")
    assert os.path.exists(temp_dir / ".git")
    assert len(os.listdir(temp_dir)) == 4
    loaded_program = ExampleProgram.from_precompiled(hub_repo, runtime_param="wassuh", config={"lm": "openai/gpt-4o"})
    assert loaded_program.runtime_param == "wassuh"
    assert loaded_program.config.lm == "openai/gpt-4o"
    assert loaded_program.config.output_type == "str"
    assert loaded_program.config.number == 1
    loaded_program.push_to_hub(hub_repo, with_code=False)

    loaded_program2 = ExampleProgram.from_precompiled(hub_repo, runtime_param="wassuh2", config={"number": 2})
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "program.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 4
    assert loaded_program2.runtime_param == "wassuh2"
    assert loaded_program2.config.number == 2
    assert loaded_program2.config.lm == "openai/gpt-4o"
    assert loaded_program2.config.output_type == "str"
    loaded_program2.push_to_hub(hub_repo, with_code=False)
    # now test with removing the local cache
    shutil.rmtree(repo_dir)
    loaded_program3 = ExampleProgram.from_precompiled(hub_repo, runtime_param="wassuh3", config={"output_type": "bool"})
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "program.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 4
    assert loaded_program3.runtime_param == "wassuh3"
    assert loaded_program3.config.output_type == "bool"
    assert loaded_program3.config.lm == "openai/gpt-4o"
    assert loaded_program3.config.number == 2


def test_precompiled_retriever_hub(hub_repo: str):
    clients = {"openai": ["sama"]}
    ExampleRetriever(ProgramWRetreiverConfig(num_fetch=10, clients=clients), needed_param="Hello").push_to_hub(
        hub_repo, with_code=False
    )
    temp_dir = Path(MODAIC_CACHE) / "temp" / hub_repo
    repo_dir = Path(PROGRAM_CACHE) / hub_repo
    assert os.path.exists(temp_dir / "config.json")
    assert os.path.exists(temp_dir / "README.md")
    assert os.path.exists(temp_dir / ".git")
    assert len(os.listdir(temp_dir)) == 3
    loaded_retriever = ExampleRetriever.from_precompiled(hub_repo, needed_param="Goodbye", config={"num_fetch": 20})
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
        hub_repo, needed_param="Goodbye2", config={"lm": "openai/gpt-4o"}
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
        hub_repo, needed_param="Goodbye3", config={"clients": {"openai": ["sama2"]}}
    )
    assert loaded_retriever3.config.clients == {"openai": ["sama2"]}
    assert loaded_retriever3.needed_param == "Goodbye3"
    assert loaded_retriever3.config.num_fetch == 20
    assert loaded_retriever3.config.embedder == loaded_retriever3.embedder_name == "openai/text-embedding-3-small"
    assert loaded_retriever3.config.lm == "openai/gpt-4o"
    assert loaded_retriever3.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_retriever3.push_to_hub(hub_repo, with_code=False)


def test_precompiled_program_with_retriever_hub(hub_repo: str):
    clients = {"openai": ["sama"]}
    config = ProgramWRetreiverConfig(num_fetch=10, clients=clients)
    retriever = ExampleRetriever(config, needed_param="Hello")
    program = ProgramWRetreiver(config, retriever)
    program.push_to_hub(hub_repo, with_code=False)
    temp_dir = Path(MODAIC_CACHE) / "temp" / hub_repo
    repo_dir = Path(PROGRAM_CACHE) / hub_repo
    assert os.path.exists(temp_dir / "config.json")
    assert os.path.exists(temp_dir / "program.json")
    assert os.path.exists(temp_dir / "README.md")
    assert os.path.exists(temp_dir / ".git")
    assert len(os.listdir(temp_dir)) == 4

    config = {"num_fetch": 20}
    loaded_retriever = ExampleRetriever.from_precompiled(hub_repo, needed_param="Goodbye", config=config)
    loaded_program = ProgramWRetreiver.from_precompiled(hub_repo, retriever=loaded_retriever, config=config)
    assert loaded_retriever.config.num_fetch == loaded_program.config.num_fetch == 20
    assert loaded_retriever.config.clients == loaded_program.config.clients == clients
    assert loaded_retriever.config.lm == loaded_program.config.lm == "openai/gpt-4o-mini"
    assert (
        loaded_retriever.config.embedder
        == loaded_program.config.embedder
        == loaded_retriever.embedder_name
        == "openai/text-embedding-3-small"
    )
    assert loaded_retriever.needed_param == "Goodbye"
    assert loaded_retriever.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_program.push_to_hub(hub_repo, with_code=False)
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "program.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 4

    config = {"lm": "openai/gpt-4o"}
    loaded_retriever2 = ExampleRetriever.from_precompiled(hub_repo, needed_param="Goodbye2", config=config)
    loaded_program2 = ProgramWRetreiver.from_precompiled(hub_repo, retriever=loaded_retriever2, config=config)
    assert loaded_retriever2.config.lm == loaded_program2.config.lm == "openai/gpt-4o"
    assert loaded_retriever2.config.num_fetch == loaded_program2.config.num_fetch == 20
    assert loaded_retriever2.config.clients == loaded_program2.config.clients == clients
    assert loaded_retriever2.needed_param == "Goodbye2"
    assert loaded_retriever2.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_program2.push_to_hub(hub_repo, with_code=False)
    assert os.path.exists(repo_dir / "config.json")
    assert os.path.exists(repo_dir / "program.json")
    assert os.path.exists(repo_dir / "README.md")
    assert os.path.exists(repo_dir / ".git")
    assert len(os.listdir(repo_dir)) == 4

    shutil.rmtree(repo_dir)
    config = {"clients": {"openai": ["sama3"]}}
    loaded_retriever3 = ExampleRetriever.from_precompiled(hub_repo, needed_param="Goodbye3", config=config)
    loaded_program3 = ProgramWRetreiver.from_precompiled(hub_repo, retriever=loaded_retriever3, config=config)
    assert loaded_retriever3.config.clients == loaded_program3.config.clients == {"openai": ["sama3"]}
    assert loaded_retriever3.config.num_fetch == loaded_program3.config.num_fetch == 20
    assert loaded_retriever3.needed_param == "Goodbye3"
    assert loaded_retriever3.config.lm == loaded_program3.config.lm == "openai/gpt-4o"
    assert loaded_retriever3.retrieve("my query") == "Retrieved 20 results for my query"
    loaded_program3.push_to_hub(hub_repo, with_code=False)


class InnerSecretProgram(dspy.Module):
    def __init__(self):
        self.predictor = dspy.Predict(Summarize)
        self.predictor.set_lm(lm=dspy.LM("openai/gpt-4o-mini", api_key="sk-proj-1234567890", hf_token="hf_1234567890"))

    def forward(self, query: str) -> str:
        return self.predictor(query=query)


class SecretProgramConfig(PrecompiledConfig):
    pass


class SecretProgram(PrecompiledProgram):
    config: SecretProgramConfig

    def __init__(self, config: SecretProgramConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(Summarize)
        self.predictor.set_lm(lm=dspy.LM("openai/gpt-4o-mini", api_key="sk-proj-1234567890"))
        self.inner = InnerSecretProgram()

    def forward(self, query: str) -> str:
        return self.inner(query=query)


def test_precompiled_program_with_secret(clean_folder: Path):
    SecretProgram(SecretProgramConfig()).save_precompiled(clean_folder)
    with open(clean_folder / "program.json", "r") as f:
        program_state = json.load(f)
    assert program_state["inner.predictor"]["lm"]["api_key"] == "********"
    assert program_state["inner.predictor"]["lm"]["hf_token"] == "********"
    assert program_state["predictor"]["lm"]["api_key"] == "********"
    loaded_program = SecretProgram.from_precompiled(clean_folder, api_key="set-api-key", hf_token="set-hf-token")
    assert loaded_program.inner.predictor.lm.kwargs["api_key"] == "set-api-key"
    assert loaded_program.inner.predictor.lm.kwargs["hf_token"] == "set-hf-token"
    assert loaded_program.predictor.lm.kwargs["api_key"] == "set-api-key"


def test_unauthorized_push_to_hub():
    pass


class NoConfigProgram(PrecompiledProgram):
    def __init__(self, runtime_param: str, **kwargs):
        super().__init__(None, **kwargs)
        self.predictor = dspy.Predict(Summarize)
        self.predictor.lm = dspy.LM("openai/gpt-4o-mini")
        self.runtime_param = runtime_param

    def forward(self, question: str, context: str) -> str:
        return self.predictor(question=question, context=context)


class DefaultConfigProgram(PrecompiledProgram):
    def __init__(self, config: PrecompiledConfig = None, runtime_param: str = "wassuh", **kwargs):
        super().__init__(config, **kwargs)
        self.predictor = dspy.Predict(Summarize)
        self.predictor.lm = dspy.LM("openai/gpt-4o-mini")
        self.runtime_param = runtime_param

    def forward(self, question: str, context: str) -> str:
        return self.predictor(question=question, context=context)


NO_CONFIG_PROGRAM_CLASSES = [NoConfigProgram, DefaultConfigProgram]


@pytest.mark.parametrize("ProgramCls", NO_CONFIG_PROGRAM_CLASSES)
def test_no_config_local(clean_folder: Path, ProgramCls: Type[PrecompiledProgram]):
    ProgramCls(runtime_param="Hello").save_precompiled(clean_folder)
    assert os.path.exists(clean_folder / "config.json")
    assert os.path.exists(clean_folder / "program.json")
    assert len(os.listdir(clean_folder)) == 2
    loaded_program = ProgramCls.from_precompiled(clean_folder, runtime_param="Hello")
    assert loaded_program.runtime_param == "Hello"

    loaded_program(question="what is the meaning of life?", context="The meaning of life is 42")


@pytest.mark.parametrize("ProgramCls", NO_CONFIG_PROGRAM_CLASSES)
def test_no_config_hub(hub_repo: str, ProgramCls: Type[PrecompiledProgram]):
    ProgramCls(runtime_param="Hello").push_to_hub(hub_repo, with_code=False)
    temp_dir = Path(MODAIC_CACHE) / "temp" / hub_repo

    assert os.path.exists(temp_dir / "config.json")
    assert os.path.exists(temp_dir / "program.json")
    assert os.path.exists(temp_dir / "README.md")
    assert os.path.exists(temp_dir / ".git")
    assert len(os.listdir(temp_dir)) == 4
    loaded_program = ProgramCls.from_precompiled(hub_repo, runtime_param="wassuhh")
    assert loaded_program.runtime_param == "wassuhh"


class NoConfigRetriever(Retriever):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def retrieve(self, query: str) -> str:
        return f"Retrieved 10 results for {query}"


class NoConfigWhRetrieverProgram(PrecompiledProgram):
    retriever: NoConfigRetriever

    def __init__(self, runtime_param: str, retriever: NoConfigRetriever, **kwargs):
        super().__init__(retriever=retriever, **kwargs)
        self.predictor = dspy.Predict(Summarize)
        self.predictor.lm = dspy.LM("openai/gpt-4o-mini")
        self.runtime_param = runtime_param

    def forward(self, question: str) -> dspy.Prediction:
        return self.predictor(question=question, context=self.retriever.retrieve(question))


def test_no_config_w_retriever_local(clean_folder: Path):
    retriever = NoConfigRetriever()
    NoConfigWhRetrieverProgram(runtime_param="Hello", retriever=retriever).save_precompiled(clean_folder)
    assert os.path.exists(clean_folder / "config.json")
    assert os.path.exists(clean_folder / "program.json")
    assert len(os.listdir(clean_folder)) == 2
    loaded_program = NoConfigWhRetrieverProgram.from_precompiled(
        clean_folder, runtime_param="Hello", retriever=retriever
    )
    assert loaded_program.runtime_param == "Hello"
    loaded_program(question="what is the meaning of life?")


def test_no_config_w_retriever_hub(hub_repo: str):
    retriever = NoConfigRetriever()
    NoConfigWhRetrieverProgram(runtime_param="Hello", retriever=retriever).push_to_hub(hub_repo, with_code=False)
    temp_dir = Path(MODAIC_CACHE) / "temp" / hub_repo

    assert os.path.exists(temp_dir / "config.json")
    assert os.path.exists(temp_dir / "program.json")
    assert os.path.exists(temp_dir / "README.md")
    assert os.path.exists(temp_dir / ".git")
    assert len(os.listdir(temp_dir)) == 4
    loaded_program = NoConfigWhRetrieverProgram.from_precompiled(hub_repo, runtime_param="wassuhh", retriever=retriever)
    assert loaded_program.runtime_param == "wassuhh"
