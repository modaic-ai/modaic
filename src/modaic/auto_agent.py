# from abc import ABC, abstractmethod
import json
import importlib
import sys
from functools import lru_cache
import os
from pathlib import Path
import git
from .utils import compute_cache_dir
from .hub import MODAIC_HUB_URL

MODAIC_CACHE = compute_cache_dir()


_REGISTRY = {}  # maps model_type string -> (ConfigCls, ModelCls)


def register(model_type: str, config_cls, model_cls):
    _REGISTRY[model_type] = (config_cls, model_cls)


@lru_cache
def _load_dynamic_class(repo_dir: str, class_path: str):
    """
    Load a class from a given repository directory and fully qualified class path.

    Params:
      repo_dir: Absolute path to a local repository directory containing the code.
      class_path: Dotted path to the target class (e.g., "pkg.module.Class").

    Returns:
      The resolved class object.
    """
    print(repo_dir)
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    module_name, _, attr = class_path.rpartition(".")
    print(module_name)
    module = importlib.import_module(module_name)
    print(module)
    return getattr(module, attr)


class AutoConfig:
    """
    Config loader for precompiled agents and indexers.
    """

    @staticmethod
    def from_precompiled(repo_path: str, *, local: bool = False):
        """
        Load a config for an agent or indexer from a precompiled repo.

        Params:
          repo_path: Hub path ("user/repo") or a local directory.
          local: If True, treat repo_path as a local directory and do not fetch.

        Returns:
          A config object constructed via the resolved config class.
        """
        repo_dir = load_repo(repo_path, local)
        cfg_path = os.path.join(repo_dir, "config.json")
        with open(cfg_path, "r") as fp:
            cfg = json.load(fp)

        model_type = cfg.get("agent_type")

        if model_type in _REGISTRY:
            ConfigClass, _ = _REGISTRY[model_type]
            return ConfigClass.from_dict(cfg)

        auto_classes_path = os.path.join(repo_dir, "auto_classes.json")
        with open(auto_classes_path, "r") as fp:
            auto_classes = json.load(fp)

        try:
            dyn_path = auto_classes["AutoConfig"]
        except KeyError:
            raise ValueError(
                f"AutoConfig not found in {auto_classes_path}. Please check that the auto_classes.json file is correct."
            )
        DynConfig = _load_dynamic_class(repo_dir, dyn_path)
        return DynConfig.from_dict(cfg)


class AutoAgent:
    """
    Dynamic loader for precompiled agents hosted on a hub or local path.
    """

    @staticmethod
    def from_precompiled(repo_path: str, *, local: bool = False, **kw):
        """
        Load a compiled agent from the given identifier.

        Params:
          repo_path: Hub path ("user/repo") or local directory.
          local: If True, treat repo_path as local and do not fetch/update from hub.
          **kw: Additional keyword arguments forwarded to the Agent constructor.

        Returns:
          An instantiated Agent subclass.
        """
        repo_dir = load_repo(repo_path, local)

        cfg_path = os.path.join(repo_dir, "config.json")
        with open(cfg_path, "r") as fp:
            _ = json.load(fp)

        cfg = AutoConfig.from_precompiled(repo_dir, local=True)
        model_type = cfg.agent_type

        if model_type in _REGISTRY:
            _, AgentClass = _REGISTRY[model_type]
        else:
            auto_classes_path = os.path.join(repo_dir, "auto_classes.json")
            with open(auto_classes_path, "r") as fp:
                auto_classes = json.load(fp)
            try:
                dyn_path = auto_classes["AutoAgent"]
            except KeyError:
                raise ValueError(
                    f"AutoAgent not found in {auto_classes_path}. Please check that the auto_classes.json file is correct."
                )
            AgentClass = _load_dynamic_class(repo_dir, dyn_path)

        return AgentClass(config=cfg, **kw)


class AutoIndexer:
    """
    Dynamic loader for precompiled indexers hosted on a hub or local path.
    """

    @staticmethod
    def from_precompiled(repo_path: str, *, local: bool = False, **kw):
        """
        Load a compiled indexer from the given identifier.

        Params:
          repo_path: hub path ("user/repo"), or local directory.
          local: If True, treat repo_path as local and do not fetch/update from hub.
          **kw: Additional keyword arguments forwarded to the Indexer constructor.

        Returns:
          An instantiated Indexer subclass.
        """
        repo_dir = load_repo(repo_path, local)

        cfg_path = os.path.join(repo_dir, "config.json")
        with open(cfg_path, "r") as fp:
            cfg_dict = json.load(fp)

        cfg = AutoConfig.from_precompiled(repo_dir, local=True)
        indexer_type = cfg_dict.get("indexer_type")

        auto_classes_path = os.path.join(repo_dir, "auto_classes.json")
        with open(auto_classes_path, "r") as fp:
            auto_classes = json.load(fp)

        if indexer_type and indexer_type in _REGISTRY:
            _, IndexerClass = _REGISTRY[indexer_type]
        else:
            try:
                dyn_path = auto_classes["AutoIndexer"]
            except KeyError:
                raise ValueError(
                    f"AutoIndexer not found in {auto_classes_path}. Please check that the auto_classes.json file is correct."
                )
            IndexerClass = _load_dynamic_class(repo_dir, dyn_path)

        return IndexerClass(config=cfg, **kw)


def git_snapshot(repo_path: str, *, rev: str = "main") -> Path:
    """
    Ensure a local cached checkout of a hub repository and return its path.

    Params:
      repo_path: Hub path ("user/repo").
      rev: Branch, tag, or full commit SHA to checkout; defaults to "main".

    Returns:
      Absolute path to the local cached repository under MODAIC_CACHE/repo_path.
    """
    # If a local folder path is provided, just return it
    repo_dir = Path(MODAIC_CACHE) / repo_path
    try:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)

        remote_url = f"{MODAIC_HUB_URL.rstrip('/')}" + f"/{repo_path}.git"

        if not repo_dir.exists():
            git.Repo.clone_from(remote_url, repo_dir, branch=rev)
            return repo_dir

        # Repo exists → update
        repo = git.Repo(repo_dir)
        if "origin" not in [r.name for r in repo.remotes]:
            repo.create_remote("origin", remote_url)
        else:
            repo.remotes.origin.set_url(remote_url)

        repo.remotes.origin.fetch()
        target = rev
        # Create/switch branch to track origin/target and hard reset to it
        repo.git.switch("-C", target, f"origin/{target}")
        repo.git.reset("--hard", f"origin/{target}")
        return repo_dir
    except Exception as e:
        repo_dir.rmdir()
        raise e


def load_repo(repo_path: str, is_local: bool = False) -> Path:
    if is_local:
        path = Path(repo_path)
        if not path.exists():
            raise FileNotFoundError(f"Local repo path {repo_path} does not exist")
        return path
    else:
        return git_snapshot(repo_path)
