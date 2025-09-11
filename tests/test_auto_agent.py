import os
import pathlib
import shutil
import subprocess

import pytest

from tests.testing_utils import delete_agent_repo

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")


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


def run_compile(repo_name: str) -> None:
    """Run the repository's compile script inside its own uv environment.

    Params:
        repo_name (str): The name of the test repository directory to compile.

    Returns:
        None
    """
    repo_dir = pathlib.Path("tests/artifacts/test_repos") / repo_name
    subprocess.run(["uv", "sync"], cwd=repo_dir, check=True)
    subprocess.run(["uv", "run", "python", "compile.py"], cwd=repo_dir, check=True)


def test_simple_repo() -> None:
    prepare_repo("simple_repo")
    run_compile("simple_repo")
