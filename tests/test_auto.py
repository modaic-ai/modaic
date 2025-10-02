import os
import pathlib
import shutil
import subprocess
from pathlib import Path

import pytest

from modaic import AutoAgent, AutoConfig, AutoRetriever
from modaic.hub import MODAIC_CACHE, get_user_info
from tests.testing_utils import delete_agent_repo

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")
INSTALL_TEST_REPO_DEPS = os.getenv("INSTALL_TEST_REPO_DEPS", "True").lower() == "true"
USERNAME = get_user_info(os.environ["MODAIC_TOKEN"])["login"]


def get_cached_agent_dir(repo_name: str) -> Path:
    return MODAIC_CACHE / "agents" / repo_name


def clean_modaic_cache() -> None:
    """Remove the MODAIC cache directory if it exists.

    Params:
        None

    Returns:
        None
    """
    shutil.rmtree(MODAIC_CACHE, ignore_errors=True)


def prepare_repo(repo_name: str) -> None:
    """Clean cache and ensure remote hub repo is deleted before test run.

    Params:
        repo_name (str): The name of the test repository in artifacts/test_repos.

    Returns:
        None
    """
    clean_modaic_cache()
    if not MODAIC_TOKEN:
        pytest.skip("Skipping because MODAIC_TOKEN is not set")
    delete_agent_repo(username=USERNAME, agent_name=repo_name)


def run_script(repo_name: str, run_path: str = "compile.py", module_mode: bool = False) -> None:
    """Run the repository's compile script inside its own uv environment.

    Params:
        repo_name (str): The name of the test repository directory to compile.

    Returns:
        None
    """
    repo_dir = pathlib.Path("tests/artifacts/test_repos") / repo_name
    if INSTALL_TEST_REPO_DEPS:
        subprocess.run(["uv", "sync"], cwd=repo_dir, check=True)
        # Ensure the root package is available in the subproject env
    if module_mode:
        subprocess.run(["uv", "run", "-m", run_path, USERNAME], cwd=repo_dir, check=True)
    else:
        subprocess.run(["uv", "run", run_path, USERNAME], cwd=repo_dir, check=True)
    # clean cache
    shutil.rmtree("tests/artifacts/temp/modaic_cache", ignore_errors=True)


def test_simple_repo() -> None:
    prepare_repo("simple_repo")
    run_script("simple_repo", run_path="agent.py")
    clean_modaic_cache()
    config = AutoConfig.from_precompiled(f"{USERNAME}/simple_repo")
    print("config", config)
    assert config.lm == "openai/gpt-4o"
    assert config.output_type == "str"
    assert config.number == 1
    cache_dir = get_cached_agent_dir(f"{USERNAME}/simple_repo")
    assert os.path.exists(cache_dir / "config.json")
    assert os.path.exists(cache_dir / "agent.json")
    assert os.path.exists(cache_dir / "auto_classes.json")
    assert os.path.exists(cache_dir / "README.md")
    assert os.path.exists(cache_dir / "agent.py")
    assert os.path.exists(cache_dir / "pyproject.toml")
    clean_modaic_cache()
    agent = AutoAgent.from_precompiled(f"{USERNAME}/simple_repo", runtime_param="Hello")
    assert agent.config.lm == "openai/gpt-4o"
    assert agent.config.output_type == "str"
    assert agent.config.number == 1
    assert agent.runtime_param == "Hello"
    clean_modaic_cache()
    agent = AutoAgent.from_precompiled(
        f"{USERNAME}/simple_repo", runtime_param="Hello", config_options={"lm": "openai/gpt-4o-mini"}
    )
    assert agent.config.lm == "openai/gpt-4o-mini"
    assert agent.config.output_type == "str"
    assert agent.config.number == 1
    assert agent.runtime_param == "Hello"
    # TODO: test third party deps installation


def test_simple_repo_with_compile():
    prepare_repo("simple_repo_with_compile")
    run_script("simple_repo_with_compile", run_path="compile.py")
    clean_modaic_cache()
    config = AutoConfig.from_precompiled(f"{USERNAME}/simple_repo_with_compile")
    assert config.lm == "openai/gpt-4o"
    assert config.output_type == "str"
    assert config.number == 1
    cache_dir = get_cached_agent_dir(f"{USERNAME}/simple_repo_with_compile")
    assert os.path.exists(cache_dir / "config.json")
    assert os.path.exists(cache_dir / "agent.json")
    assert os.path.exists(cache_dir / "auto_classes.json")
    assert os.path.exists(cache_dir / "README.md")
    assert os.path.exists(cache_dir / "agent" / "agent.py")
    assert os.path.exists(cache_dir / "agent" / "mod.py")
    assert os.path.exists(cache_dir / "pyproject.toml")
    clean_modaic_cache()
    agent = AutoAgent.from_precompiled(f"{USERNAME}/simple_repo_with_compile", runtime_param="Hello")
    assert agent.config.lm == "openai/gpt-4o"
    assert agent.config.output_type == "str"
    assert agent.config.number == 1
    assert agent.runtime_param == "Hello"
    clean_modaic_cache()
    agent = AutoAgent.from_precompiled(
        f"{USERNAME}/simple_repo_with_compile", runtime_param="Hello", config_options={"lm": "openai/gpt-4o-mini"}
    )
    assert agent.config.lm == "openai/gpt-4o-mini"
    assert agent.config.output_type == "str"
    assert agent.config.number == 1
    assert agent.runtime_param == "Hello"
    # TODO: test third party deps installation


@pytest.mark.parametrize("repo_name", ["nested_repo", "nested_repo_2", "nested_repo_3"])
def test_nested_repo(repo_name):
    prepare_repo(repo_name)
    if repo_name == "nested_repo":
        run_script(repo_name, run_path="agent.compile", module_mode=True)
    elif repo_name == "nested_repo_2":
        run_script(repo_name, run_path="compile.py")
    else:
        run_script(repo_name, run_path="agent.agent", module_mode=True)
    clean_modaic_cache()
    config = AutoConfig.from_precompiled(f"{USERNAME}/{repo_name}", clients={"get_replaced": "noob"})
    assert config.num_fetch == 1
    assert config.lm == "openai/gpt-4o-mini"
    assert config.embedder == "openai/text-embedding-3-small"
    assert config.clients == {"get_replaced": "noob"}
    clean_modaic_cache()
    retriever = AutoRetriever.from_precompiled(f"{USERNAME}/{repo_name}", needed_param="hello")
    agent = AutoAgent.from_precompiled(f"{USERNAME}/{repo_name}", retriever=retriever)
    assert agent.config.num_fetch == 1
    assert agent.config.lm == "openai/gpt-4o-mini"
    assert agent.config.embedder == "openai/text-embedding-3-small"
    assert agent.config.clients == {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]}
    assert retriever.needed_param == "hello"
    assert agent.forward("my query") == "Retrieved 1 results for my query"
    clean_modaic_cache()
    config_options = {"lm": "openai/gpt-4o"}
    retriever = AutoRetriever.from_precompiled(
        f"{USERNAME}/{repo_name}", needed_param="hello", config_options=config_options
    )
    agent = AutoAgent.from_precompiled(f"{USERNAME}/{repo_name}", retriever=retriever, config_options=config_options)
    assert agent.config.num_fetch == 1
    assert agent.config.lm == "openai/gpt-4o"
    assert agent.config.embedder == "openai/text-embedding-3-small"
    assert agent.config.clients == {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]}
    assert retriever.needed_param == "hello"
    assert agent.forward("my query") == "Retrieved 1 results for my query"


def test_auth():
    pass
