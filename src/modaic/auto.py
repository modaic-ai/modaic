import importlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Optional, Type, TypedDict

from .hub import AGENTS_CACHE, load_repo
from .precompiled import PrecompiledAgent, PrecompiledConfig, Retriever, is_local_path

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")


class RegisteredRepo(TypedDict, total=False):
    AutoConfig: Type[PrecompiledConfig]
    AutoAgent: Type[PrecompiledAgent]
    AutoRetriever: Type[Retriever]


_REGISTRY: dict[str, RegisteredRepo] = {}


def register(
    name: str,
    auto_type: Literal["AutoConfig", "AutoAgent", "AutoRetriever"],
    cls: Type[PrecompiledConfig | PrecompiledAgent | Retriever],
):
    if name in _REGISTRY:
        _REGISTRY[name][auto_type] = cls
    else:
        _REGISTRY[name] = {auto_type: cls}


# TODO: Cleanup code still using parent_mdoule
@lru_cache
def _load_dynamic_class(
    repo_dir: Path, class_path: str, hub_path: str = None
) -> Type[PrecompiledConfig | PrecompiledAgent | Retriever]:
    """
    Load a class from a given repository directory and fully qualified class path.

    Args:
      repo_dir: Absolute path to a local repository directory containing the code.
      class_path: Dotted path to the target class (e.g., "pkg.module.Class").
      hub_path: The path to the repo on modaic hub (if its a hub repo) *Must be specified if its a hub repo*

    Returns:
      The resolved class object.
    """
    if hub_path is None:
        print("local repo")
        # Local folder case
        repo_dir_str = str(repo_dir)
        if repo_dir_str not in sys.path:
            sys.path.insert(0, repo_dir_str)
        full_path = f"{class_path}"
    else:
        print("hub repo")
        # loaded hub repo case
        agents_cache_str = str(AGENTS_CACHE)
        if agents_cache_str not in sys.path:
            print("inserting", agents_cache_str, "into sys.path")
            sys.path.insert(0, agents_cache_str)
        parent_module = hub_path.replace("/", ".")
        full_path = f"{parent_module}.{class_path}"

    module_name, _, attr = full_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


class AutoConfig:
    """
    Config loader for precompiled agents and retrievers.
    """

    @staticmethod
    def from_precompiled(repo_path: str, **kwargs) -> PrecompiledConfig:
        local = is_local_path(repo_path)
        repo_dir = load_repo(repo_path, local)
        AutoConfig._from_precompiled(repo_dir, hub_path=repo_path if not local else None, **kwargs)

    @staticmethod
    def _from_precompiled(repo_dir: Path, hub_path: str = None, **kwargs) -> PrecompiledConfig:
        """
        Load a config for an agent or retriever from a precompiled repo.

        Args:
          repo_dir: The path to the repo directory. the loaded local repository directory.
          hub_path: The path to the repo on modaic hub (if its a hub repo) *Must be specified if its a hub repo*

        Returns:
          A config object constructed via the resolved config class.
        """

        cfg_path = repo_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Failed to load AutoConfig, config.json not found in {hub_path or str(repo_dir)}")
        with open(cfg_path, "r") as fp:
            cfg = json.load(fp)

        ConfigClass = _load_auto_class(repo_dir, "AutoConfig", hub_path=hub_path)  # noqa: N806
        return ConfigClass(**{**cfg, **kwargs})


class AutoAgent:
    """
    Dynamic loader for precompiled agents hosted on a hub or local path.
    """

    @staticmethod
    def from_precompiled(
        repo_path: str,
        *,
        config_options: Optional[dict] = None,
        project: Optional[str] = None,
        **kw,
    ) -> PrecompiledAgent:
        """
        Load a compiled agent from the given identifier.

        Args:
          repo_path: Hub path ("user/repo") or local directory.
          project: Optional project name. If not provided and repo_path is a hub path, defaults to the repo name.
          **kw: Additional keyword arguments forwarded to the Agent constructor.

        Returns:
          An instantiated Agent subclass.
        """
        # TODO: fast lookups via registry
        local = is_local_path(repo_path)
        repo_dir = load_repo(repo_path, local)
        hub_path = repo_path if not local else None

        if config_options is None:
            config_options = {}

        cfg = AutoConfig._from_precompiled(repo_dir, hub_path=hub_path, **config_options)
        AgentClass = _load_auto_class(repo_dir, "AutoAgent", hub_path=hub_path)  # noqa: N806

        # automatically configure repo and project from repo_path if not provided
        # TODO: redundant checks in if statement. Investigate removing.
        if not local and "/" in repo_path and not repo_path.startswith("/"):
            parts = repo_path.split("/")
            if len(parts) >= 2:
                kw.setdefault("repo", repo_path)
                # Use explicit project parameter if provided, otherwise default to repo name
                if project is not None:
                    kw.setdefault("project", f"{repo_path}-{project}")
                else:
                    kw.setdefault("project", repo_path)
                kw.setdefault("trace", True)

        return AgentClass(config=cfg, **kw)


class AutoRetriever:
    """
    Dynamic loader for precompiled retrievers hosted on a hub or local path.
    """

    @staticmethod
    def from_precompiled(
        repo_path: str,
        *,
        config_options: Optional[dict] = None,
        project: Optional[str] = None,
        **kw,
    ) -> Retriever:
        """
        Load a compiled retriever from the given identifier.

        Args:
          repo_path: hub path ("user/repo"), or local directory.
          project: Optional project name. If not provided and repo_path is a hub path, defaults to the repo name.
          **kw: Additional keyword arguments forwarded to the Retriever constructor.

        Returns:
          An instantiated Retriever subclass.
        """
        local = is_local_path(repo_path)
        repo_dir = load_repo(repo_path, local)
        hub_path = repo_path if not local else None

        if config_options is None:
            config_options = {}

        cfg = AutoConfig._from_precompiled(repo_dir, hub_path=hub_path, **config_options)
        RetrieverClass = _load_auto_class(repo_dir, "AutoRetriever", hub_path=hub_path)  # noqa: N806

        # automatically configure repo and project from repo_path if not provided
        # TODO: redundant checks in if statement. Investigate removing.
        if not local and "/" in repo_path and not repo_path.startswith("/"):
            parts = repo_path.split("/")
            if len(parts) >= 2:
                kw.setdefault("repo", repo_path)
                if project is not None:
                    kw.setdefault("project", f"{repo_path}-{project}")
                else:
                    kw.setdefault("project", repo_path)
                kw.setdefault("trace", True)

        return RetrieverClass(config=cfg, **kw)


def _load_auto_class(
    repo_dir: Path,
    auto_name: Literal["AutoConfig", "AutoAgent", "AutoRetriever"],
    hub_path: str = None,
) -> Type[PrecompiledConfig | PrecompiledAgent | Retriever]:
    """
    Load a class from the auto_classes.json file.

    Args:
        repo_dir: The path to the repo directory. the loaded local repository directory.
        auto_name: The name of the auto class to load. (AutoConfig, AutoAgent, AutoRetriever)
        hub_path: The path to the repo on modaic hub (if its a hub repo) *Must be specified if its a hub repo*
    """
    # determine if the repo was loaded from local or hub
    local = hub_path is None
    auto_classes_path = repo_dir / "auto_classes.json"

    if not auto_classes_path.exists():
        raise FileNotFoundError(
            f"Failed to load {auto_name}, auto_classes.json not found in {hub_path or str(repo_dir)}, if this is your repo, make sure you push_to_hub() with `with_code=True`"
        )

    with open(auto_classes_path, "r") as fp:
        auto_classes = json.load(fp)

    if not (auto_class_path := auto_classes.get(auto_name)):
        raise KeyError(
            f"{auto_name} not found in {hub_path or str(repo_dir)}/auto_classes.json. Please check that the auto_classes.json file is correct."
        ) from None

    repo_dir = repo_dir.parent.parent if not local else repo_dir
    LoadedClass = _load_dynamic_class(repo_dir, auto_class_path, hub_path=hub_path)  # noqa: N806
    return LoadedClass


def builtin_agent(name: str) -> Callable[[Type], Type]:
    def _wrap(cls: Type) -> Type:
        register(name, "AutoAgent", cls)
        return cls

    return _wrap


def builtin_indexer(name: str) -> Callable[[Type], Type]:
    def _wrap(cls: Type) -> Type:
        register(name, "AutoRetriever", cls)
        return cls

    return _wrap


def builtin_config(name: str) -> Callable[[Type], Type]:
    def _wrap(cls: Type) -> Type:
        register(name, "AutoConfig", cls)
        return cls

    return _wrap
