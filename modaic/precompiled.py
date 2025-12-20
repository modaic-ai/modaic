import inspect
import json
import os
import pathlib
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
    Type,
    TypeVar,
)

import dspy
from pydantic import BaseModel

from modaic.observability import Trackable, track_modaic_obj

from .exceptions import MissingSecretError
from .hub import Commit, load_repo, sync_and_push

C = TypeVar("C", bound="PrecompiledConfig")
A = TypeVar("A", bound="PrecompiledProgram")
R = TypeVar("R", bound="Retriever")


class PrecompiledConfig(BaseModel):
    model: Optional[str] = None

    def save_precompiled(
        self,
        path: str | Path,
    ) -> None:
        """
        Saves the config to a config.json file in the given local folder.
        Also saves the auto_classes.json with AutoConfig and any other auto classes passed to _extra_auto_classes

        Args:
            path: The local folder to save the config to.
        """
        # NOTE: since we don't allow PrecompiledConfig.push_to_hub(), when _extra_auto_classes is None we will assume that we don't need to save the auto_classes.json
        self._save_precompiled(path)

    def _save_precompiled(self, path: Path, extra_auto_classes: Optional[Dict[str, object]] = None) -> None:
        """
        Saves the config to a config.json file in the given local folder.
        Also saves the auto_classes.json with AutoConfig and any other auto classes passed to _extra_auto_classes

        Args:
            path: The local folder to save the config to.
            extra_auto_classes: An argument used internally to add extra auto classes to program repo
        """
        from .module_utils import _module_path

        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        if extra_auto_classes is None:
            return

        auto_classes = {"AutoConfig": self}
        auto_classes.update(extra_auto_classes)

        auto_classes_paths = {k: _module_path(cls) for k, cls in auto_classes.items()}

        with open(path / "auto_classes.json", "w") as f:
            json.dump(auto_classes_paths, f, indent=2)

    @classmethod
    def from_precompiled(cls: Type[C], path: str | Path, rev: str = "main", **kwargs) -> C:
        """
        Loads the config from a config.json file in the given path. The path can be a local directory or a repo on Modaic Hub.

        Args:
            path: The path to load the config from. Can be a local directory or a repo on Modaic Hub.
            **kwargs: Additional keyword arguments used to override the default config.

        Returns:
            An instance of the PrecompiledConfig class.
        """
        local = is_local_path(path)
        local_dir, _ = load_repo(path, local, rev=rev)
        # TODO load repos from the hub if not local
        path = local_dir / "config.json"
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**{**config_dict, **kwargs})

    @classmethod
    def from_dict(cls: Type[C], dict: Dict, **kwargs) -> C:
        """
        Loads the config from a dictionary.

        Args:
            dict: A dictionary containing the config.
            **kwargs: Additional keyword arguments used to override the default config.

        Returns:
            An instance of the PrecompiledConfig class.
        """
        instance = cls(**{**dict, **kwargs})
        return instance

    @classmethod
    def from_json(cls: Type[C], path: str, **kwargs) -> C:
        """
        Loads the config from a json file.

        Args:
            path: The path to load the config from.
            **kwargs: Additional keyword arguments used to override the default config.

        Returns:
            An instance of the PrecompiledConfig class.
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(**{**config_dict, **kwargs})

    def to_dict(self) -> Dict:
        """
        Converts the config to a dictionary.
        """
        return self.model_dump()

    def to_json(self) -> str:
        """
        Converts the config to a json string.
        """
        return self.model_dump_json()


# Use a metaclass to enforce super().__init__() with config
class PrecompiledProgram(dspy.Module):
    """
    Bases: `dspy.Module`

    PrecompiledProgram supports observability tracking through DSPy callbacks.
    """

    config: PrecompiledConfig
    retriever: Optional["Retriever"]
    _source: Path = None
    _source_commit: Optional[Commit] = None

    def __init__(
        self,
        config: Optional[PrecompiledConfig | dict] = None,
        *,
        retriever: Optional["Retriever"] = None,
        **kwargs,
    ):
        if config is None:
            config = self.__annotations__.get("config", PrecompiledConfig)()
        elif isinstance(config, dict):
            config = self.__annotations__.get("config", PrecompiledConfig)(**config)
        elif type(config) is not self.__annotations__.get("config", PrecompiledConfig):
            raise ValueError(
                f"config must be an instance of {self.__class__.__name__}'s config class ({self.__annotations__.get('config', PrecompiledConfig)}). Sublasses are not allowed."
            )
        self.config = config  # type: ignore
        # create DSPy callback for observability if tracing is enabled

        # initialize DSPy Module with callbacks
        super().__init__()
        self.retriever = retriever
        # TODO: throw a warning if the config of the retriever has different values than the config of the program

    def forward(self, **kwargs) -> str:
        """
        Forward pass for the program.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Forward pass result.
        """
        raise NotImplementedError(
            "Forward pass for PrecompiledProgram is not implemented. You must implement a forward method in your subclass."
        )

    def save_precompiled(self, path: str, _with_auto_classes: bool = False) -> None:
        """
        Saves the program.json and the config.json to the given local folder.

        Args:
            path: The local folder to save the program and config to. Must be a local path.
            _with_auto_classes: Internally used argument used to configure whether to save the auto classes mapping.
        """
        path = pathlib.Path(path)
        extra_auto_classes = None
        if _with_auto_classes:
            extra_auto_classes = {"AutoProgram": self}
            if self.retriever is not None:
                extra_auto_classes["AutoRetriever"] = self.retriever
        self.config._save_precompiled(path, extra_auto_classes)
        self.save(path / "program.json")
        _clean_secrets(path / "program.json")

    @classmethod
    def from_precompiled(
        cls: Type[A],
        path: str | Path,
        config: Optional[PrecompiledConfig | dict] = None,
        api_key: Optional[str | dict[str, str]] = None,
        hf_token: Optional[str | dict[str, str]] = None,
        rev: str = "main",
        **kwargs,
    ) -> A:
        """
        Loads the program and the config from the given path.

        Args:
            path: The path to load the program and config from. Can be a local path or a path on Modaic Hub.
            config: A dictionary containg key-value pairs used to override the default config.
            api_key: Your API key.
            hf_token: Your Hugging Face token.
            **kwargs: Additional keyword arguments forwarded to the PrecompiledProgram's constructor.

        Returns:
            An instance of the PrecompiledProgram class.
        """

        if cls is PrecompiledProgram:
            raise ValueError("from_precompiled() can only be used on a subclass of PrecompiledProgram.")

        ConfigClass: Type[PrecompiledConfig] = cls.__annotations__.get("config", PrecompiledConfig)  # noqa: N806
        local = is_local_path(path)
        local_dir, source_commit = load_repo(path, local, rev=rev)
        config = config or {}
        config = ConfigClass.from_precompiled(local_dir, **config)

        # Check if the program takes in a config parameter
        sig = inspect.signature(cls.__init__)
        if "config" in sig.parameters:
            program = cls(config=config, **kwargs)
        else:
            program = cls(**kwargs)
        # Support new (program.json) and legacy (agent.json) naming
        program_state_path = local_dir / "program.json"
        agent_state_path = local_dir / "agent.json"
        state_path = (
            program_state_path if program_state_path.exists() else agent_state_path
        )  # TODO: deprecate agent.json in next major release

        if state_path.exists():
            secrets = {"api_key": api_key, "hf_token": hf_token}
            state = _get_state_with_secrets(state_path, secrets)
            program.load_state(state)

        # We set _source_commit to track the commit hash.
        # _source is intentionally not set here because its initialized from PrecompiledProgram and not AutoProgram.
        program._source_commit = source_commit
        return program

    def push_to_hub(
        self,
        repo_path: str,
        access_token: Optional[str] = None,
        commit_message: str = "(no commit message)",
        with_code: Optional[bool] = None,
        private: bool = False,
        branch: str = "main",
        tag: str = None,
    ) -> Commit:
        """
        Pushes the program and the config to the given repo_path.

        Args:
            repo_path: The path on Modaic hub to save the program and config to.
            access_token: Your Modaic access token.
            commit_message: The commit message to use when pushing to the hub.
            with_code: Whether to save the code along with the program.json and config.json.
                - Defaults to True if the Program was loaded via AutoProgram, otherwise defaults to False
        """
        # Default to with_code=True if self._source is provided, otherwise default to false
        if with_code is None:
            with_code = self._source is not None

        return sync_and_push(
            self,
            repo_path,
            access_token=access_token,
            commit_message=commit_message,
            private=private,
            branch=branch,
            tag=tag,
            with_code=with_code,
        )


class Retriever(ABC, Trackable):
    config: PrecompiledConfig
    _source: Optional[Path] = None
    _source_commit: Optional[Commit] = None

    def __init__(self, config: Optional[PrecompiledConfig | dict] = None, **kwargs):
        ABC.__init__(self)
        Trackable.__init__(self, **kwargs)
        if config is None:
            config = self.__annotations__.get("config", PrecompiledConfig)()
        elif isinstance(config, dict):
            config = self.__annotations__.get("config", PrecompiledConfig)(**config)
        elif type(config) is not self.__annotations__.get("config", PrecompiledConfig):
            raise ValueError(
                f"config must be an instance of {self.__class__.__name__}'s config class ({self.__annotations__.get('config', PrecompiledConfig)}). Sublasses are not allowed."
            )
        self.config = config  # type: ignore

    @track_modaic_obj
    @abstractmethod
    def retrieve(self, query: str, **kwargs):
        pass

    @classmethod
    def from_precompiled(
        cls: Type[R], path: str | Path, config: Optional[dict] = None, rev: str = "main", **kwargs
    ) -> R:
        """
        Loads the retriever and the config from the given path.
        """
        if cls is Retriever:
            raise ValueError("from_precompiled() can only be used on a subclass of Retriever.")

        ConfigClass: Type[PrecompiledConfig] = cls.__annotations__["config"]  # noqa: N806
        local = is_local_path(path)
        local_dir, source_commit = load_repo(path, local, rev=rev)
        config = config or {}
        config = ConfigClass.from_precompiled(local_dir, **config)
        sig = inspect.signature(cls.__init__)
        if "config" in sig.parameters:
            retriever = cls(config=config, **kwargs)
        else:
            retriever = cls(**kwargs)

        # We set _source_commit to track the commit hash.
        # _source is intentionally not set here because its initialized from Retriever and not AutoRetriever.
        retriever._source_commit = source_commit
        return retriever

    def save_precompiled(self, path: str | Path, _with_auto_classes: bool = False) -> None:
        """
        Saves the retriever configuration to the given path.

        Args:
          path: The path to save the retriever configuration and auto classes mapping.
          _with_auto_classes: Internal argument used to configure whether to save the auto classes mapping.
        """
        path_obj = pathlib.Path(path)
        extra_auto_classes = None
        if _with_auto_classes:
            extra_auto_classes = {"AutoRetriever": self}
        self.config._save_precompiled(path_obj, extra_auto_classes)

    def push_to_hub(
        self,
        repo_path: str,
        access_token: Optional[str] = None,
        commit_message: str = "(no commit message)",
        with_code: Optional[bool] = None,
        private: bool = False,
        branch: str = "main",
        tag: str = None,
    ) -> Commit:
        """
        Pushes the retriever and the config to the given repo_path.

        Args:
            repo_path: The path on Modaic hub to save the DSPy programand config to.
            access_token: Your Modaic access token.
            commit_message: The commit message to use when pushing to the hub.
            with_code: Whether to save the code along with the retriever.json and config.json.
                - Defaults to True if the Retriever was loaded via AutoRetriever, otherwise defaults to False
        """
        # Default to with_code=True if self._source is provided, otherwise default to false
        if with_code is None:
            with_code = self._source is not None

        return sync_and_push(
            self,
            repo_path,
            access_token=access_token,
            commit_message=commit_message,
            private=private,
            branch=branch,
            tag=tag,
            with_code=with_code,
        )


class Indexer(Retriever):
    config: PrecompiledConfig

    @abstractmethod
    def index(self, contents: Any, **kwargs):
        pass


def is_local_path(s: str | Path) -> bool:
    # absolute or relative filesystem path
    if isinstance(s, Path):
        return True
    s = str(s)

    if os.path.isabs(s) or s.startswith((".", "/", "\\")):
        return True
    parts = s.split("/")
    # hub IDs: "repo" or "user/repo"
    if len(parts) == 1:
        raise ValueError(
            f"Invalid repo: '{s}'. Please prefix local paths with './', '/', or '../' . And use 'user/repo' format for hub paths."
        )
    elif len(parts) == 2 and all(parts):
        return False
    return True


SECRET_MASK = "********"
COMMON_SECRETS = ["api_key", "hf_token"]


def _clean_secrets(path: Path, extra_secrets: Optional[list[str]] = None):
    """
    Removes all secret keys from `lm` dict in program.json file
    """
    secret_keys = COMMON_SECRETS + (extra_secrets or [])

    with open(path, "r") as f:
        d = json.load(f)

    for predictor in d.values():
        lm = predictor.get("lm", None)
        if lm is None:
            continue
        for k in lm.keys():
            if k in secret_keys:
                lm[k] = SECRET_MASK

    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def _get_state_with_secrets(path: Path, secrets: dict[str, str | dict[str, str] | None]):
    """`
    Fills secret keys in `lm` dict in program.json file

    Args:
        path: The path to the program.json file.
        secrets: A dictionary containing the secrets to fill in the `lm` dict.
            - Dict[k,v] where k is the name of a secret (e.g. "api_key") and v is the value of the secret
            - If v is a string, every lm will use v for k
            - if v is a dict, each key of v should be the name of a named predictor
            (e.g. "my_program.predict", "my_program.summarizer") mapping to the secret value for that predictor
    Returns:
        A dictionary containing the state of the program.json file with the secrets filled in.
    """
    with open(path, "r") as f:
        named_predictors = json.load(f)

    def _get_secret(predictor_name: str, secret_name: str) -> Optional[str]:
        if secret_val := secrets.get(secret_name):
            if isinstance(secret_val, str):
                return secret_val
            elif isinstance(secret_val, dict):
                return secret_val.get(predictor_name)
        return None

    for predictor_name, predictor in named_predictors.items():
        lm = predictor.get("lm", {})
        for kw, arg in lm.items():
            if kw in COMMON_SECRETS and arg != "" and arg != SECRET_MASK:
                warnings.warn(
                    f"{str(path)} exposes the secret key {kw}. Please remove it or ensure this file is not made public.",
                    stacklevel=2,
                )
            secret = _get_secret(predictor_name, kw)
            if secret is not None and arg != "" and arg != SECRET_MASK:
                raise ValueError(
                    f"Failed to fill insert secret value for {predictor_name}['lm']['{kw}']. It is already set to {arg}"
                )
            elif secret is None and kw in COMMON_SECRETS:
                raise MissingSecretError(f"Please specify a value for {kw} in the secrets dictionary", kw)
            elif secret is not None:
                lm[kw] = secret
    return named_predictors


# Deprecated alias for backward compatibility
PrecompiledAgent = PrecompiledProgram


def __getattr__(name: str):
    """Handle deprecated imports with warnings."""
    if name == "PrecompiledAgent":
        warnings.warn(
            "PrecompiledAgent is deprecated and will be removed in a future version. "
            "Please use PrecompiledProgram instead for better parity with DSPy.",
            DeprecationWarning,
            stacklevel=2,
        )
        return PrecompiledProgram
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
