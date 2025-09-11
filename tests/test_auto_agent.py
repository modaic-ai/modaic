import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from modaic.auto_agent import AutoAgent
from tests.testing_utils import delete_agent_repo

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")


def clean_modaic_cache():
    shutil.rmtree(os.environ["MODAIC_CACHE"], ignore_errors=True)


def prepare_repo(repo_name):
    clean_modaic_cache()
    if not MODAIC_TOKEN:
        pytest.skip("Skipping because MODAIC_TOKEN is not set")
    # delete the repo
    delete_agent_repo(username="hub_tests", agent_name=repo_name)


def run_compile(repo_name):
    repo_dir = pathlib.Path("tests/artifacts/test_repos") / repo_name
    subprocess.run(["uv", "sync"], cwd=repo_dir, check=True)
    subprocess.run(
        [sys.executable, "compile.py"],
        cwd=repo_dir,
        check=True,
    )


def test_simple_repo():
    prepare_repo("simple_repo")
    run_compile("simple_repo")
