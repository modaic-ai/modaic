import importlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Type

from .hub import load_repo
from .precompiled import PrecompiledAgent, PrecompiledConfig, Retriever, is_local_path

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
    print(f"repo_dir_str: {repo_dir_str}")
    print(f"sys.path: {sys.path}")
    if repo_dir_str not in sys.path:
        # print(f"Inserting {repo_dir_str} into sys.path")
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
    Config loader for precompiled agents and retrievers.
    """

    @staticmethod
    def from_precompiled(repo_path: str, *, parent_module: Optional[str] = None, **kwargs) -> PrecompiledConfig:
        """
        Load a config for an agent or retriever from a precompiled repo.

        Args:
          repo_path: Hub path ("user/repo") or a local directory.
          parent_module: Optional dotted module prefix (e.g., "swagginty.TableRAG") to use to import classes from repo_path. If provided, overides default parent_module behavior.

        Returns:
          A config object constructed via the resolved config class.
        """
        local = is_local_path(repo_path)
        repo_dir = load_repo(repo_path, local)

        cfg_path = repo_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Failed to load AutoConfig, config.json not found in {repo_path}")
        with open(cfg_path, "r") as fp:
            cfg = json.load(fp)

        ConfigClass = _load_auto_class(repo_path, repo_dir, "AutoConfig", parent_module=parent_module)  # noqa: N806
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
        parent_module: Optional[str] = None,
        project: Optional[str] = None,
        **kw,
    ) -> PrecompiledAgent:
        """
        Load a compiled agent from the given identifier.

        Args:
          repo_path: Hub path ("user/repo") or local directory.
          parent_module: Optional dotted module prefix (e.g., "swagginty.TableRAG") to use to import classes from repo_path. If provided, overides default parent_module behavior.
          project: Optional project name. If not provided and repo_path is a hub path, defaults to the repo name.
          **kw: Additional keyword arguments forwarded to the Agent constructor.

        Returns:
          An instantiated Agent subclass.
        """
        local = is_local_path(repo_path)
        repo_dir = load_repo(repo_path, local)

        if config_options is None:
            config_options = {}

        cfg = AutoConfig.from_precompiled(repo_dir, local=True, parent_module=parent_module, **config_options)
        AgentClass = _load_auto_class(repo_path, repo_dir, "AutoAgent", parent_module=parent_module)  # noqa: N806

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
    Dynamic loader for precompiled retrievers hosted on a hub or local path.
    """

    @staticmethod
    def from_precompiled(
        repo_path: str,
        *,
        config_options: Optional[dict] = None,
        parent_module: Optional[str] = None,
        project: Optional[str] = None,
        **kw,
    ) -> Retriever:
        """
        Load a compiled retriever from the given identifier.

        Args:
          repo_path: hub path ("user/repo"), or local directory.
          parent_module: Optional dotted module prefix (e.g., "swagginty.TableRAG") to use to import classes from repo_path. If provided, overides default parent_module behavior.
          project: Optional project name. If not provided and repo_path is a hub path, defaults to the repo name.
          **kw: Additional keyword arguments forwarded to the Retriever constructor.

        Returns:
          An instantiated Retriever subclass.
        """
        local = is_local_path(repo_path)
        repo_dir = load_repo(repo_path, local)

        if config_options is None:
            config_options = {}

        cfg = AutoConfig.from_precompiled(repo_dir, local=True, parent_module=parent_module, **config_options)
        RetrieverClass = _load_auto_class(repo_path, repo_dir, "AutoRetriever", parent_module=parent_module)  # noqa: N806

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

        return RetrieverClass(config=cfg, **kw)


def _load_auto_class(
    repo_path: str,
    repo_dir: Path,
    auto_name: Literal["AutoConfig", "AutoAgent", "AutoRetriever"],
    parent_module: Optional[str] = None,
) -> Type[PrecompiledConfig | PrecompiledAgent | Retriever]:
    """
    Load a class from the auto_classes.json file.

    Args:
        repo_path: The path to the repo. (local or hub path)
        repo_dir: The path to the repo directory. the loaded local repository directory.
        auto_name: The name of the auto class to load. (AutoConfig, AutoAgent, AutoRetriever)
        parent_module: The parent module to use to import the class.
    """
    # determine if the repo was loaded from local or hub
    local = is_local_path(repo_path)
    auto_classes_path = repo_dir / "auto_classes.json"

    if not auto_classes_path.exists():
        raise FileNotFoundError(
            f"Failed to load {auto_name}, auto_classes.json not found in {repo_path}, if this is your repo, make sure you push_to_hub() with `with_code=True`"
        )

    with open(auto_classes_path, "r") as fp:
        auto_classes = json.load(fp)

    if not (auto_class_path := auto_classes.get(auto_name)):
        raise KeyError(
            f"{auto_name} not found in {repo_path}/auto_classes.json. Please check that the auto_classes.json file is correct."
        ) from None

    if auto_class_path in _REGISTRY:
        _, LoadedClass = _REGISTRY[auto_class_path]  # noqa: N806
    else:
        if parent_module is None and not local:
            parent_module = str(repo_path).replace("/", ".")

        repo_dir = repo_dir.parent.parent if not local else repo_dir
        LoadedClass = _load_dynamic_class(repo_dir, auto_class_path, parent_module=parent_module)  # noqa: N806
    return LoadedClass
