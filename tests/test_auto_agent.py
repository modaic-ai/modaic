import os
import pathlib
import shutil
import subprocess
from pathlib import Path

import pytest

from modaic import AutoAgent, AutoConfig, AutoRetriever
from modaic.hub import get_user_info
from tests.testing_utils import delete_agent_repo

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")


@pytest.fixture
def username() -> str:
    return get_user_info(os.environ["MODAIC_TOKEN"])["login"]


def get_cached_agent_dir(repo_name: str) -> str:
    return Path(os.environ["MODAIC_CACHE"]) / "agents" / repo_name


def clean_modaic_cache() -> None:
    """Remove the MODAIC cache directory if it exists.

    Params:
        None

    Returns:
        None
    """
    shutil.rmtree(os.environ["MODAIC_CACHE"], ignore_errors=True)


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
    delete_agent_repo(username="hub_tests", agent_name=repo_name)


def run_script(username: str, repo_name: str, run_path: str = "compile.py") -> None:
    """Run the repository's compile script inside its own uv environment.

    Params:
        repo_name (str): The name of the test repository directory to compile.

    Returns:
        None
    """
    env = os.environ.copy()
    env.update(
        {
            "MODAIC_CACHE": "../../temps/modaic_cache",
        }
    )
    repo_dir = pathlib.Path("tests/artifacts/test_repos") / repo_name
    subprocess.run(["uv", "sync"], cwd=repo_dir, check=True, env=env)
    # Ensure the root package is available in the subproject env
    subprocess.run(["uv", "pip", "install", "-e", "../../../../"], cwd=repo_dir, check=True, env=env)
    subprocess.run(["uv", "run", "python", run_path, username], cwd=repo_dir, check=True, env=env)
    # clean cache
    shutil.rmtree("tests/artifacts/temp/modaic_cache", ignore_errors=True)


def test_simple_repo(username: str) -> None:
    prepare_repo("simple_repo")
    run_script(username, "simple_repo", run_path="agent.py")
    config = AutoConfig.from_precompiled(f"{username}/simple_repo")
    assert config.lm == "openai/gpt-4o"
    assert config.output_type == "str"
    assert config.number == 1
    clean_modaic_cache()
    agent = AutoAgent.from_precompiled("hub_tests/simple_repo", runtime_param="Hello")
    assert agent.config.lm == "openai/gpt-4o"
    assert agent.config.output_type == "str"
    assert agent.config.number == 1
    assert agent.runtime_param == "Hello"
    clean_modaic_cache()
    agent = AutoAgent.from_precompiled(
        "hub_tests/simple_repo", runtime_param="Hello", config_options={"lm": "openai/gpt-4o-mini"}
    )
    assert agent.config.lm == "openai/gpt-4o-mini"
    assert agent.config.output_type == "str"
    assert agent.config.number == 1
    assert agent.runtime_param == "Hello"
    # TODO: test third party deps installation


def test_simple_repo_with_compile(username: str):
    pass
