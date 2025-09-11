import importlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Type

from .hub import load_repo
from .precompiled_agent import PrecompiledAgent, PrecompiledConfig
from .retrievers import Retriever

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")


_REGISTRY = {}  # maps model_type string -> (ConfigCls, ModelCls)


def register(model_type: str, config_cls: Type[PrecompiledConfig], model_cls: Type[PrecompiledAgent]):
    _REGISTRY[model_type] = (config_cls, model_cls)


@lru_cache
def _load_dynamic_class(
    repo_dir: str, class_path: str, parent_module: Optional[str] = None
) -> Type[PrecompiledConfig | PrecompiledAgent | Retriever]:
    """
    Load a class from a given repository directory and fully qualified class path.

    Args:
      repo_dir: Absolute path to a local repository directory containing the code.
      class_path: Dotted path to the target class (e.g., "pkg.module.Class").
      parent_module: Optional dotted module prefix (e.g., "swagginty.TableRAG"). If provided,
                     class_path is treated as relative to this module and only the agents cache
                     root is added to sys.path.

    Returns:
      The resolved class object.
    """

    repo_path = Path(repo_dir)

    repo_dir_str = str(repo_path)
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)
    full_path = (
        f"{parent_module}.{class_path}"
        if parent_module and not class_path.startswith(parent_module + ".")
        else class_path
    )

    module_name, _, attr = full_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


class AutoConfig:
    """
    Config loader for precompiled agents and indexers.
    """

    @staticmethod
    def from_precompiled(
        repo_path: str, *, local: bool = False, parent_module: Optional[str] = None, **kwargs
    ) -> PrecompiledConfig:
        """
        Load a config for an agent or indexer from a precompiled repo.

        Args:
          repo_path: Hub path ("user/repo") or a local directory.
          local: If True, treat repo_path as a local directory and do not fetch.
          parent_module: Optional dotted module prefix (e.g., "swagginty.TableRAG") to use to import classes from repo_path. If provided, overides default parent_module behavior.

        Returns:
          A config object constructed via the resolved config class.
        """
        repo_dir = load_repo(repo_path, local)
        cfg_path = os.path.join(repo_dir, "config.json")
        with open(cfg_path, "r") as fp:
            cfg = json.load(fp)

        auto_classes_path = os.path.join(repo_dir, "auto_classes.json")
        with open(auto_classes_path, "r") as fp:
            auto_classes = json.load(fp)

        try:
            dyn_path = auto_classes["AutoConfig"]
        except KeyError:
            raise KeyError(
                f"AutoConfig not found in {auto_classes_path}. Please check that the auto_classes.json file is correct."
            ) from None

        if parent_module is None and not local:
            parent_module = str(repo_path).replace("/", ".")

        repo_dir = repo_dir.parent.parent if not local else repo_dir
        DynConfig: Type[PrecompiledConfig] = _load_dynamic_class(repo_dir, dyn_path, parent_module=parent_module)  # noqa: N806
        return DynConfig(**cfg, **kwargs)


class AutoAgent:
    """
    Dynamic loader for precompiled agents hosted on a hub or local path.
    """

    @staticmethod
    def from_precompiled(
        repo_path: str,
        *,
        config_options: Optional[dict] = None,
        local: bool = False,
        parent_module: Optional[str] = None,
        project: Optional[str] = None,
        **kw,
    ) -> PrecompiledAgent:
        """
        Load a compiled agent from the given identifier.

        Args:
          repo_path: Hub path ("user/repo") or local directory.
          local: If True, treat repo_path as local and do not fetch/update from hub.
          parent_module: Optional dotted module prefix (e.g., "swagginty.TableRAG") to use to import classes from repo_path. If provided, overides default parent_module behavior.
          project: Optional project name. If not provided and repo_path is a hub path, defaults to the repo name.
          **kw: Additional keyword arguments forwarded to the Agent constructor.

        Returns:
          An instantiated Agent subclass.
        """
        repo_dir = load_repo(repo_path, local)

        cfg_path = os.path.join(repo_dir, "config.json")
        with open(cfg_path, "r") as fp:
            _ = json.load(fp)
        if config_options is None:
            config_options = {}

        auto_classes_path = os.path.join(repo_dir, "auto_classes.json")
        with open(auto_classes_path, "r") as fp:
            auto_classes = json.load(fp)
        if not (auto_agent_path := auto_classes.get("AutoAgent")):
            raise KeyError(
                f"AutoAgent not found in {auto_classes_path}. Please check that the auto_classes.json file is correct."
            ) from None

        cfg = AutoConfig.from_precompiled(repo_dir, local=True, parent_module=parent_module, **config_options)

        if auto_agent_path in _REGISTRY:
            _, AgentClass = _REGISTRY[auto_agent_path]  # noqa: N806
        else:
            if parent_module is None and not local:
                parent_module = str(repo_path).replace("/", ".")

            repo_dir = repo_dir.parent.parent if not local else repo_dir
            AgentClass = _load_dynamic_class(repo_dir, auto_agent_path, parent_module=parent_module)  # noqa: N806

        # automatically configure repo and project from repo_path if not provided
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
    Dynamic loader for precompiled indexers hosted on a hub or local path.
    """

    @staticmethod
    def from_precompiled(
        repo_path: str,
        *,
        config_options: Optional[dict] = None,
        local: bool = False,
        parent_module: Optional[str] = None,
        project: Optional[str] = None,
        **kw,
    ) -> Retriever:
        """
        Load a compiled indexer from the given identifier.

        Args:
          repo_path: hub path ("user/repo"), or local directory.
          local: If True, treat repo_path as local and do not fetch/update from hub.
          parent_module: Optional dotted module prefix (e.g., "swagginty.TableRAG") to use to import classes from repo_path. If provided, overides default parent_module behavior.
          project: Optional project name. If not provided and repo_path is a hub path, defaults to the repo name.
          **kw: Additional keyword arguments forwarded to the Indexer constructor.

        Returns:
          An instantiated Indexer subclass.
        """
        repo_dir = load_repo(repo_path, local)

        if config_options is None:
            config_options = {}

        cfg = AutoConfig.from_precompiled(repo_dir, local=True, parent_module=parent_module, **config_options)

        auto_classes_path = os.path.join(repo_dir, "auto_classes.json")
        with open(auto_classes_path, "r") as fp:
            auto_classes = json.load(fp)

        if not (auto_retriever_path := auto_classes.get("AutoRetriever")):
            raise KeyError(
                f"AutoRetriever not found in {auto_classes_path}. Please check that the auto_classes.json file is correct."
            ) from None

        if auto_retriever_path in _REGISTRY:
            _, IndexerClass = _REGISTRY[auto_retriever_path]  # noqa: N806
        else:
            if parent_module is None and not local:
                parent_module = str(repo_path).replace("/", ".")

            repo_dir = repo_dir.parent.parent if not local else repo_dir
            IndexerClass = _load_dynamic_class(repo_dir, auto_retriever_path, parent_module=parent_module)  # noqa: N806

        # automatically configure repo and project from repo_path if not provided
        if not local and "/" in repo_path and not repo_path.startswith("/"):
            parts = repo_path.split("/")
            if len(parts) >= 2:
                kw.setdefault("repo", repo_path)
                if project is not None:
                    kw.setdefault("project", f"{repo_path}-{project}")
                else:
                    kw.setdefault("project", repo_path)
                kw.setdefault("trace", True)

        return IndexerClass(config=cfg, **kw)
