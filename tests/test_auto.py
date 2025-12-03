import os
import pathlib
import shutil
import subprocess
from pathlib import Path
from typing import Union

import pytest
import tomlkit as tomlk

from modaic import AutoProgram, AutoConfig, AutoRetriever
from modaic.hub import MODAIC_CACHE, get_user_info
from tests.testing_utils import delete_program_repo

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")
INSTALL_TEST_REPO_DEPS = os.getenv("INSTALL_TEST_REPO_DEPS", "True").lower() == "true"
USERNAME = get_user_info(os.environ["MODAIC_TOKEN"])["login"]


def get_cached_program_dir(repo_name: str) -> Path:
    return MODAIC_CACHE / "programs" / repo_name


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
    delete_program_repo(username=USERNAME, program_name=repo_name)


def run_script(repo_name: str, run_path: str = "compile.py") -> None:
    """Run the repository's compile script inside its own uv environment.

    Params:
        repo_name (str): The name of the test repository directory to compile.

    Returns:
        None
    """
    env = os.environ.copy()
    env.update(
        {
            "MODAIC_CACHE": "../../temp/modaic_cache",
        }
    )
    repo_dir = pathlib.Path("tests/artifacts/test_repos") / repo_name
    if INSTALL_TEST_REPO_DEPS:
        subprocess.run(["uv", "sync"], cwd=repo_dir, check=True, env=env)
        # Ensure the root package is available in the subproject env
    # Run as file
    if run_path.endswith(".py"):
        subprocess.run(["uv", "run", run_path, USERNAME], cwd=repo_dir, check=True, env=env)
    # Run as module
    else:
        subprocess.run(["uv", "run", "-m", run_path, USERNAME], cwd=repo_dir, check=True, env=env)
    # clean cache
    shutil.rmtree("tests/artifacts/temp/modaic_cache", ignore_errors=True)


# recursive dict/list of dicts/lists of strs representing a folder structure
FolderLayout = dict[str, Union[str, "FolderLayout"]] | list[Union[str, "FolderLayout"]]


def assert_expected_files(cache_dir: Path, extra_expected_files: FolderLayout):
    default_expected = ["program.json", "auto_classes.json", "config.json", "pyproject.toml", "README.md", ".git"]
    if isinstance(extra_expected_files, list):
        expected = extra_expected_files + default_expected
    elif isinstance(extra_expected_files, dict):
        expected = [extra_expected_files] + default_expected
    else:
        raise ValueError(f"Invalid folder layout: {extra_expected_files}")
    assert_folder_layout(cache_dir, expected)


def assert_top_level_names(dir: Path, expected_files: FolderLayout | str, root: bool = True):
    if isinstance(expected_files, list):
        expected_names = []
        for obj in expected_files:
            if isinstance(obj, str):
                expected_names.append(obj)
            elif isinstance(obj, dict):
                expected_names.extend(list(obj.keys()))
            else:
                raise ValueError(f"Invalid folder layout: {expected_files}")
    elif isinstance(expected_files, dict):
        expected_names = list(expected_files.keys())
    elif isinstance(expected_files, str):
        expected_names = [expected_files]
    else:
        raise ValueError(f"Invalid folder layout: {expected_files}")
    expected_names = expected_names if root else expected_names + ["__init__.py"]
    missing = set(expected_names) - set(os.listdir(dir))
    assert missing == set(), f"Missing files, in {dir}, {missing}"
    unexpected = set(os.listdir(dir)) - set(expected_names)
    assert unexpected.issubset(set(["__pycache__", "__init__.py"])), (
        f"Unexpected files in {dir}, {unexpected - set(['__pycache__', '__init__.py'])}"
    )


def assert_folder_layout(
    dir: Path, expected_files: FolderLayout | str, root: bool = True, assert_top_level: bool = True
):
    """
    Asserts that the files in the directory match the expected folder structure.
    Checking that only expected files are included. Will raise assertion error if unexpected files are included.
    Args:
        dir: The directory to assert the files in.
        expected_files: The expected folder structure.

    Raises:
        Assertion error if expected file not found in path or if unexpected file found in path
    """
    # dir is a single file folder
    if isinstance(expected_files, str):
        assert_top_level_names(dir, expected_files, root)
    # dir is a folder containg multiples files or subfolders
    elif isinstance(expected_files, list):
        assert_top_level_names(dir, expected_files, root)
        for file in expected_files:
            if isinstance(file, dict):
                assert_folder_layout(dir, file, root=False, assert_top_level=False)
            elif not isinstance(file, str):
                raise ValueError(f"Invalid folder layout: {expected_files}")
    # dir contains subfolders, however don't check top level because we don't know if this is the entirety of dir or a subset
    elif isinstance(expected_files, dict):
        for key, value in expected_files.items():
            assert_folder_layout(dir / key, value, root=False)
    else:
        raise ValueError(f"Invalid folder layout: {expected_files}")


def assert_dependencies(cache_dir: Path, extra_expected_dependencies: list[str]):
    expected_dependencies = extra_expected_dependencies + ["dspy", "modaic"]

    pyproject_path = cache_dir / "pyproject.toml"
    doc = tomlk.parse(pyproject_path.read_text(encoding="utf-8"))
    actual_dependencies = doc.get("project", {}).get("dependencies", [])

    missing = set(expected_dependencies) - set(actual_dependencies)
    assert missing == set(), f"Missing dependencies, {missing}"
    unexpected = set(actual_dependencies) - set(expected_dependencies)
    assert unexpected == set(), f"Unexpected dependencies, {unexpected}"


def test_simple_repo() -> None:
    prepare_repo("simple_repo")
    run_script("simple_repo", run_path="program.py")
    clean_modaic_cache()
    config = AutoConfig.from_precompiled(f"{USERNAME}/simple_repo")
    assert config.lm == "openai/gpt-4o"
    assert config.output_type == "str"
    assert config.number == 1
    cache_dir = get_cached_program_dir(f"{USERNAME}/simple_repo")
    assert_expected_files(cache_dir, ["program.py"])
    assert_dependencies(cache_dir, ["dspy", "modaic", "praw"])

    clean_modaic_cache()
    program = AutoProgram.from_precompiled(f"{USERNAME}/simple_repo", runtime_param="Hello")
    assert program.config.lm == "openai/gpt-4o"
    assert program.config.output_type == "str"
    assert program.config.number == 1
    assert program.runtime_param == "Hello"
    clean_modaic_cache()
    program = AutoProgram.from_precompiled(
        f"{USERNAME}/simple_repo", runtime_param="Hello", config={"lm": "openai/gpt-4o-mini"}
    )
    assert program.config.lm == "openai/gpt-4o-mini"
    assert program.config.output_type == "str"
    assert program.config.number == 1
    assert program.runtime_param == "Hello"
    # TODO: test third party deps installation


simple_repo_with_compile_extra_files = [{"program": ["program.py", "mod.py"]}, "compile.py", "include_me_too.txt"]


def test_simple_repo_with_compile():
    prepare_repo("simple_repo_with_compile")
    run_script("simple_repo_with_compile", run_path="compile.py")
    clean_modaic_cache()
    config = AutoConfig.from_precompiled(f"{USERNAME}/simple_repo_with_compile")
    assert config.lm == "openai/gpt-4o"
    assert config.output_type == "str"
    assert config.number == 1
    cache_dir = get_cached_program_dir(f"{USERNAME}/simple_repo_with_compile")
    assert os.path.exists(cache_dir / "config.json")
    assert os.path.exists(cache_dir / "program.json")
    assert os.path.exists(cache_dir / "auto_classes.json")
    assert os.path.exists(cache_dir / "README.md")
    assert os.path.exists(cache_dir / "program" / "program.py")
    assert os.path.exists(cache_dir / "program" / "mod.py")
    assert os.path.exists(cache_dir / "pyproject.toml")
    assert os.path.exists(cache_dir / "include_me_too.txt")
    extra_files = [{"program": ["program.py", "mod.py"]}, "compile.py", "include_me_too.txt"]
    assert_expected_files(cache_dir, extra_files)
    assert_dependencies(cache_dir, ["dspy", "modaic"])
    clean_modaic_cache()
    program = AutoProgram.from_precompiled(f"{USERNAME}/simple_repo_with_compile", runtime_param="Hello")
    assert program.config.lm == "openai/gpt-4o"
    assert program.config.output_type == "str"
    assert program.config.number == 1
    assert program.runtime_param == "Hello"
    clean_modaic_cache()
    program = AutoProgram.from_precompiled(
        f"{USERNAME}/simple_repo_with_compile", runtime_param="Hello", config={"lm": "openai/gpt-4o-mini"}
    )
    assert program.config.lm == "openai/gpt-4o-mini"
    assert program.config.output_type == "str"
    assert program.config.number == 1
    assert program.runtime_param == "Hello"
    # TODO: test third party deps installation


nested_repo_extra_files = {
    "program": [
        {
            "tools": {"google": "google_search.py", "jira": "jira_api_tools.py"},
            "utils": ["second_degree_import.py", "used.py"],
        },
        "program.py",
        "compile.py",
        "config.py",
        "retriever.py",
    ]
}
nested_repo_2_extra_files = [
    {
        "program": [
            {
                "tools": {"google": "google_search.py", "jira": "jira_api_tools.py"},
                "utils": [
                    "second_degree_import.py",
                    "unused_but_included.py",
                    "used.py",
                ],
            },
            "program.py",
            "config.py",
            "retriever.py",
        ]
    },
    {"unused_but_included_folder": [".env", "folder_content1.py", "folder_content2.txt"]},
    "compile.py",
]
nested_repo_3_extra_files = {
    "program": [
        {
            "tools": [{"google": "google_search.py", "jira": "jira_api_tools.py"}, "unused_but_included2.py"],
            "utils": ["second_degree_import.py", "unused_but_included.py", "used.py"],
        },
        "program.py",
        "config.py",
        "retriever.py",
    ],
}


@pytest.mark.parametrize(
    "repo_name, run_path, extra_expected_files, extra_expected_dependencies",
    [
        (
            "nested_repo",
            "program.compile",
            nested_repo_extra_files,
            [],
        ),
        (
            "nested_repo_2",
            "compile.py",
            nested_repo_2_extra_files,
            ["dspy", "modaic", "praw", "sagemaker"],
        ),
        (
            "nested_repo_3",
            "program.program",
            nested_repo_3_extra_files,
            ["dspy", "modaic"],
        ),
    ],
)
def test_nested_repo(
    repo_name: str, run_path: str, extra_expected_files: FolderLayout, extra_expected_dependencies: list[str]
):
    prepare_repo(repo_name)
    run_script(repo_name, run_path=run_path)
    clean_modaic_cache()
    config = AutoConfig.from_precompiled(f"{USERNAME}/{repo_name}", clients={"get_replaced": "noob"})
    assert config.num_fetch == 1
    assert config.lm == "openai/gpt-4o-mini"
    assert config.embedder == "openai/text-embedding-3-small"
    assert config.clients == {"get_replaced": "noob"}

    cache_dir = get_cached_program_dir(f"{USERNAME}/{repo_name}")
    assert_expected_files(cache_dir, extra_expected_files)
    assert_dependencies(cache_dir, extra_expected_dependencies)

    clean_modaic_cache()
    retriever = AutoRetriever.from_precompiled(f"{USERNAME}/{repo_name}", needed_param="hello")
    program = AutoProgram.from_precompiled(f"{USERNAME}/{repo_name}", retriever=retriever)
    assert program.config.num_fetch == 1
    assert program.config.lm == "openai/gpt-4o-mini"
    assert program.config.embedder == "openai/text-embedding-3-small"
    assert program.config.clients == {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]}
    assert retriever.needed_param == "hello"
    assert program.forward("my query") == "Retrieved 1 results for my query"
    clean_modaic_cache()
    config = {"lm": "openai/gpt-4o"}
    retriever = AutoRetriever.from_precompiled(f"{USERNAME}/{repo_name}", needed_param="hello", config=config)
    program = AutoProgram.from_precompiled(f"{USERNAME}/{repo_name}", retriever=retriever, config=config)
    assert program.config.num_fetch == 1
    assert program.config.lm == "openai/gpt-4o"
    assert program.config.embedder == "openai/text-embedding-3-small"
    assert program.config.clients == {"mit": ["csail", "mit-media-lab"], "berkeley": ["bear"]}
    assert retriever.needed_param == "hello"
    assert program.forward("my query") == "Retrieved 1 results for my query"


def test_auth():
    pass
