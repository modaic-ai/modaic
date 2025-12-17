import importlib
import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Optional, Type, TypedDict

from .constants import PROGRAMS_CACHE
from .hub import load_repo
from .precompiled import PrecompiledConfig, PrecompiledProgram, Retriever, is_local_path


class RegisteredRepo(TypedDict, total=False):
    AutoConfig: Type[PrecompiledConfig]
    AutoProgram: Type[PrecompiledProgram]
    AutoRetriever: Type[Retriever]


_REGISTRY: dict[str, RegisteredRepo] = {}


def register(
    name: str,
    auto_type: Literal["AutoConfig", "AutoProgram", "AutoRetriever"],
    cls: Type[PrecompiledConfig | PrecompiledProgram | Retriever],
):
    if name in _REGISTRY:
        _REGISTRY[name][auto_type] = cls
    else:
        _REGISTRY[name] = {auto_type: cls}


# TODO: Cleanup code still using parent_mdoule
@lru_cache
def _load_dynamic_class(
    repo_dir: Path, class_path: str, hub_path: str = None, rev: str = "main"
) -> Type[PrecompiledConfig | PrecompiledProgram | Retriever]:
    """
    Load a class from a given repository directory and fully qualified class path.

    Args:
      repo_dir: Absolute path to a local repository directory containing the code.
      class_path: Dotted path to the target class (e.g., "pkg.program.Class").
      hub_path: The path to the repo on modaic hub (if its a hub repo) *Must be specified if its a hub repo*

    Returns:
      The resolved class object.
    """
    if class_path == "modaic.PrecompiledConfig":
        return PrecompiledConfig
    if hub_path is None:
        # Local folder case
        repo_dir_str = str(repo_dir)
        if repo_dir_str not in sys.path:
            sys.path.insert(0, repo_dir_str)
        full_path = f"{class_path}"
    else:
        # loaded hub repo case
        programs_cache_str = str(PROGRAMS_CACHE)
        if programs_cache_str not in sys.path:
            sys.path.insert(0, programs_cache_str)
        parent_module = hub_path.replace("/", ".")
        parent_module = f"{parent_module}.{rev}"
        full_path = f"{parent_module}.{class_path}"

    module_name, _, attr = full_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


class AutoConfig:
    """
    Config loader for precompiled programs and retrievers.
    """

    @staticmethod
    def from_precompiled(repo_path: str, rev: str = "main", **kwargs) -> PrecompiledConfig:
        local = is_local_path(repo_path)
        repo_dir, _ = load_repo(repo_path, local, rev=rev)
        return AutoConfig._from_precompiled(repo_dir, hub_path=repo_path if not local else None, **kwargs)

    @staticmethod
    def _from_precompiled(repo_dir: Path, hub_path: str = None, rev: str = "main", **kwargs) -> PrecompiledConfig:
        """
        Load a config for an program or retriever from a precompiled repo.

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

        ConfigClass = _load_auto_class(repo_dir, "AutoConfig", hub_path=hub_path, rev=rev)  # noqa: N806
        return ConfigClass(**{**cfg, **kwargs})


class AutoProgram:
    """
    Dynamic loader for precompiled programs hosted on a hub or local path.
    """

    @staticmethod
    def from_precompiled(
        repo_path: str,
        *,
        config: Optional[dict] = None,
        rev: str = "main",
        **kw,
    ) -> PrecompiledProgram:
        """
        Load a compiled program from the given identifier.

        Args:
          repo_path: Hub path ("user/repo") or local directory.
          **kw: Additional keyword arguments forwarded to the Program constructor.

        Returns:
          An instantiated Program subclass.
        """
        # TODO: fast lookups via registry
        local = is_local_path(repo_path)
        repo_dir, source_commit = load_repo(repo_path, local, rev=rev)
        hub_path = repo_path if not local else None

        if config is None:
            config = {}

        cfg = AutoConfig._from_precompiled(repo_dir, hub_path=hub_path, rev=rev, **config)
        # Support new (AutoProgram) and legacy (AutoAgent) naming in auto_classes.json
        try:
            ProgramClass = _load_auto_class(repo_dir, "AutoProgram", hub_path=hub_path, rev=rev)  # noqa: N806
        except KeyError:
            # Fall back to legacy AutoAgent for backward compatibility
            ProgramClass = _load_auto_class(repo_dir, "AutoAgent", hub_path=hub_path, rev=rev)  # noqa: N806

        # automatically configure repo and project from repo_path if not provided
        # TODO: redundant checks in if statement. Investigate removing.
        program = ProgramClass(config=cfg, **kw)
        program._source = repo_dir
        program._source_commit = source_commit
        return program


class AutoRetriever:
    """
    Dynamic loader for precompiled retrievers hosted on a hub or local path.
    """

    @staticmethod
    def from_precompiled(
        repo_path: str,
        *,
        config: Optional[dict] = None,
        rev: str = "main",
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
        repo_dir, source_commit = load_repo(repo_path, local, rev=rev)
        hub_path = repo_path if not local else None

        if config is None:
            config = {}

        cfg = AutoConfig._from_precompiled(repo_dir, hub_path=hub_path, rev=rev, **config)
        RetrieverClass = _load_auto_class(repo_dir, "AutoRetriever", hub_path=hub_path, rev=rev)  # noqa: N806

        retriever = RetrieverClass(config=cfg, **kw)
        retriever._source = repo_dir
        retriever._source_commit = source_commit
        # automatically configure repo and project from repo_path if not provided
        return retriever


def _load_auto_class(
    repo_dir: Path,
    auto_name: Literal["AutoConfig", "AutoProgram", "AutoAgent", "AutoRetriever"],
    hub_path: str = None,
    rev: str = "main",
) -> Type[PrecompiledConfig | PrecompiledProgram | Retriever]:
    """
    Load a class from the auto_classes.json file.

    Args:
        repo_dir: The path to the repo directory. the loaded local repository directory.
        auto_name: The name of the auto class to load. (AutoConfig, AutoProgram, AutoAgent (deprecated), AutoRetriever)
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
    LoadedClass = _load_dynamic_class(repo_dir, auto_class_path, hub_path=hub_path, rev=rev)  # noqa: N806
    return LoadedClass


def builtin_program(name: str) -> Callable[[Type], Type]:
    """Decorator to register a builtin program."""

    def _wrap(cls: Type) -> Type:
        register(name, "AutoProgram", cls)
        return cls

    return _wrap


def builtin_agent(name: str) -> Callable[[Type], Type]:
    """Deprecated: Use builtin_program instead."""
    warnings.warn(
        "builtin_agent is deprecated and will be removed in a future version. "
        "Please use builtin_program instead for better parity with DSPy.",
        DeprecationWarning,
        stacklevel=2,
    )

    def _wrap(cls: Type) -> Type:
        register(name, "AutoProgram", cls)
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


# Deprecated alias for backward compatibility
class AutoAgent(AutoProgram):
    """
    Deprecated: Use AutoProgram instead.

    Dynamic loader for precompiled programs hosted on a hub or local path.
    """

    @staticmethod
    def from_precompiled(
        repo_path: str,
        *,
        config: Optional[dict] = None,
        **kw,
    ) -> PrecompiledProgram:
        """Load a compiled program from the given identifier.

        .. deprecated::
            Use :class:`AutoProgram` instead. AutoAgent is deprecated for better parity with DSPy.
        """
        warnings.warn(
            "AutoAgent is deprecated and will be removed in a future version. "
            "Please use AutoProgram instead for better parity with DSPy.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AutoProgram.from_precompiled(repo_path, config=config, **kw)
