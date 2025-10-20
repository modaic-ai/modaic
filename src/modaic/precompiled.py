import inspect
import json
import os
import pathlib
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Type,
    TypeVar,
    Union,
)

import dspy
from git import config
from pydantic import BaseModel

from modaic.module_utils import create_agent_repo
from modaic.observability import Trackable, track_modaic_obj

from .exceptions import MissingSecretError
from .hub import load_repo, push_folder_to_hub
from .module_utils import _module_path

if TYPE_CHECKING:
    from modaic.context.base import Context

C = TypeVar("C", bound="PrecompiledConfig")
A = TypeVar("A", bound="PrecompiledAgent")
R = TypeVar("R", bound="Retriever")


class PrecompiledConfig(BaseModel):
    def save_precompiled(
        self,
        path: str | Path,
        _extra_auto_classes: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Saves the config to a config.json file in the given local folder.
        Also saves the auto_classes.json with AutoConfig and any other auto classes passed to _extra_auto_classes

        Args:
            path: The local folder to save the config to.
            _extra_auto_classes: An argument used internally to add extra auto classes to agent repo
        """
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # NOTE: since we don't allow PrecompiledConfig.push_to_hub(), when _extra_auto_classes is None we will assume that we don't need to save the auto_classes.json
        if _extra_auto_classes is None:
            return

        auto_classes = {"AutoConfig": self}
        if _extra_auto_classes is not None:
            auto_classes.update(_extra_auto_classes)

        auto_classes_paths = {k: _module_path(cls) for k, cls in auto_classes.items()}

        with open(path / "auto_classes.json", "w") as f:
            json.dump(auto_classes_paths, f, indent=2)

    @classmethod
    def from_precompiled(cls: Type[C], path: str | Path, **kwargs) -> C:
        """
        Loads the config from a config.json file in the given path. The path can be a local directory or a repo on Modaic Hub.

        Args:
            path: The path to load the config from. Can be a local directory or a repo on Modaic Hub.
            **kwargs: Additional keyword arguments used to override the default config.

        Returns:
            An instance of the PrecompiledConfig class.
        """
        local = is_local_path(path)
        local_dir = load_repo(path, local)
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
class PrecompiledAgent(dspy.Module):
    """
    Bases: `dspy.Module`

    PrecompiledAgent supports observability tracking through DSPy callbacks.
    """

    config: PrecompiledConfig
    retriever: "Retriever"

    def __init__(
        self,
        config: PrecompiledConfig,
        *,
        retriever: Optional["Retriever"] = None,
        repo: Optional[str] = None,
        project: Optional[str] = None,
        trace: bool = False,
    ):
        # create DSPy callback for observability if tracing is enabled
        callbacks = []
        # FIXME This logic is not correct.
        if trace and (repo or project):
            try:
                from opik.integrations.dspy.callback import OpikCallback

                # create project name from repo and project
                if repo and project:
                    project_name = f"{repo}-{project}"
                elif repo and not project:
                    project_name = repo
                else:
                    raise ValueError("Must provide either repo to enable observability tracking")

                opik_callback = OpikCallback(project_name=project_name, log_graph=True)
                callbacks.append(opik_callback)
            except ImportError:
                # opikcallback not available, continue without tracking
                pass

        # initialize DSPy Module with callbacks
        super().__init__()
        # FIXME this adds the same callback for every agent. Should only be the current agent.
        if callbacks:
            # set callbacks using DSPy's configuration
            import dspy

            current_settings = dspy.settings
            existing_callbacks = getattr(current_settings, "callbacks", [])
            dspy.settings.configure(callbacks=existing_callbacks + callbacks)

        self.config = config
        self.retriever = retriever

        # update retriever repo and project if provided
        if self.retriever and hasattr(self.retriever, "set_repo_project"):
            self.retriever.set_repo_project(repo=repo, project=project, trace=trace)

        # TODO: throw a warning if the config of the retriever has different values than the config of the agent

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Make sure subclasses have an annotated config attribute
        if not (config_class := cls.__annotations__.get("config")) or config_class is PrecompiledConfig:
            raise ValueError(
                f"""config class could not be found in {cls.__name__}. \n
                Hint: Please add an annotation for config to your subclass.
                Example:
                class {cls.__name__}(PrecompiledAgent):
                    config: YourConfigClass
                    def __init__(self, config: YourConfigClass, **kwargs):
                        super().__init__(config, **kwargs)
                        ...
                """
            )

    def forward(self, **kwargs) -> str:
        """
        Forward pass for the agent.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Forward pass result.
        """
        raise NotImplementedError(
            "Forward pass for PrecompiledAgent is not implemented. You must implement a forward method in your subclass."
        )

    def save_precompiled(self, path: str, _with_auto_classes: bool = False) -> None:
        """
        Saves the agent.json and the config.json to the given local folder.

        Args:
            path: The local folder to save the agent and config to. Must be a local path.
            _with_auto_classes: Internally used argument used to configure whether to save the auto classes mapping.
        """
        path = pathlib.Path(path)
        extra_auto_classes = None
        if _with_auto_classes:
            extra_auto_classes = {"AutoAgent": self}
            if self.retriever is not None:
                extra_auto_classes["AutoRetriever"] = self.retriever
        self.config.save_precompiled(path, extra_auto_classes)
        self.save(path / "agent.json")
        _clean_secrets(path / "agent.json")

    @classmethod
    def from_precompiled(
        cls: Type[A],
        path: str | Path,
        config_options: Optional[dict] = None,
        api_key: Optional[str | dict[str, str]] = None,
        hf_token: Optional[str | dict[str, str]] = None,
        **kwargs,
    ) -> A:
        """
        Loads the agent and the config from the given path.

        Args:
            path: The path to load the agent and config from. Can be a local path or a path on Modaic Hub.
            config_options: A dictionary containg key-value pairs used to override the default config.
            api_key: Your API key.
            hf_token: Your Hugging Face token.
            **kwargs: Additional keyword arguments forwarded to the PrecompiledAgent's constructor.

        Returns:
            An instance of the PrecompiledAgent class.
        """

        if cls is PrecompiledAgent:
            raise ValueError("from_precompiled() can only be used on a subclass of PrecompiledAgent.")

        ConfigClass: Type[PrecompiledConfig] = cls.__annotations__["config"]  # noqa: N806
        local = is_local_path(path)
        local_dir = load_repo(path, local)
        config_options = config_options or {}
        config = ConfigClass.from_precompiled(local_dir, **config_options)
        agent = cls(config, **kwargs)
        agent_state_path = local_dir / "agent.json"
        if agent_state_path.exists():
            secrets = {"api_key": api_key, "hf_token": hf_token}
            state = _get_state_with_secrets(agent_state_path, secrets)
            agent.load_state(state)
        return agent

    def push_to_hub(
        self,
        repo_path: str,
        access_token: Optional[str] = None,
        commit_message: str = "(no commit message)",
        with_code: bool = False,
    ) -> None:
        """
        Pushes the agent and the config to the given repo_path.

        Args:
            repo_path: The path on Modaic hub to save the agent and config to.
            access_token: Your Modaic access token.
            commit_message: The commit message to use when pushing to the hub.
            with_code: Whether to save the code along with the agent.json and config.json.
        """
        _push_to_hub(
            self,
            repo_path=repo_path,
            access_token=access_token,
            commit_message=commit_message,
            with_code=with_code,
        )


class Retriever(ABC, Trackable):
    config: PrecompiledConfig

    def __init__(self, config: PrecompiledConfig, **kwargs):
        ABC.__init__(self)
        Trackable.__init__(self, **kwargs)
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Make sure subclasses have an annotated config attribute
        # Unimplemented abstract classes get a pass (like Indexer for example)
        if inspect.isabstract(cls):
            return
        if not (config_class := cls.__annotations__.get("config")) or config_class is PrecompiledConfig:
            raise ValueError(
                f"""config class could not be found in {cls.__name__}. \n
                Hint: Please add an annotation for config to your subclass.
                Example:
                class {cls.__name__}({cls.__bases__[0].__name__}):
                    config: YourConfigClass
                    def __init__(self, config: YourConfigClass, **kwargs):
                        super().__init__(config, **kwargs)
                        ...
                """
            )

    @track_modaic_obj
    @abstractmethod
    def retrieve(self, query: str, **kwargs):
        pass

    @classmethod
    def from_precompiled(cls: Type[R], path: str | Path, config_options: Optional[dict] = None, **kwargs) -> R:
        """
        Loads the retriever and the config from the given path.
        """
        if cls is PrecompiledAgent:
            raise ValueError("from_precompiled() can only be used on a subclass of PrecompiledAgent.")

        ConfigClass: Type[PrecompiledConfig] = cls.__annotations__["config"]  # noqa: N806
        local = is_local_path(path)
        local_dir = load_repo(path, local)
        config_options = config_options or {}
        config = ConfigClass.from_precompiled(local_dir, **config_options)

        retriever = cls(config, **kwargs)
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
        self.config.save_precompiled(path_obj, extra_auto_classes)

    def push_to_hub(
        self,
        repo_path: str,
        access_token: Optional[str] = None,
        commit_message: str = "(no commit message)",
        with_code: bool = False,
    ) -> None:
        """
        Pushes the retriever and the config to the given repo_path.

        Args:
            repo_path: The path on Modaic hub to save the agent and config to.
            access_token: Your Modaic access token.
            commit_message: The commit message to use when pushing to the hub.
            with_code: Whether to save the code along with the retriever.json and config.json.
        """
        _push_to_hub(self, repo_path, access_token, commit_message, with_code)


class Indexer(Retriever):
    config: PrecompiledConfig

    @abstractmethod
    def index(self, contents: Any, **kwargs):
        pass


# CAVEAT: PrecompiledConfig does not support push_to_hub() intentionally,
# this is to avoid confusion when pushing a config to the hub thinking it
# will update the config.json when in reality it will overwrite the entire
# directory to an empty one with just the config.json
def _push_to_hub(
    self: Union[PrecompiledAgent, "Retriever"],
    repo_path: str,
    access_token: Optional[str] = None,
    commit_message: str = "(no commit message)",
    with_code: bool = True,
) -> None:
    """
    Pushes the agent or retriever and the config to the given repo_path.
    """
    repo_dir = create_agent_repo(repo_path, with_code=with_code)
    self.save_precompiled(repo_dir, _with_auto_classes=with_code)
    push_folder_to_hub(
        repo_dir,
        repo_path=repo_path,
        access_token=access_token,
        commit_message=commit_message,
    )


def is_local_path(s: str | Path) -> bool:
    # absolute or relative filesystem path
    if isinstance(s, Path):
        return True
    s = str(s)

    print("SSSS", s)
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
    Removes all secret keys from `lm` dict in agent.json file
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
    Fills secret keys in `lm` dict in agent.json file

    Args:
        path: The path to the agent.json file.
        secrets: A dictionary containing the secrets to fill in the `lm` dict.
            - Dict[k,v] where k is the name of a secret (e.g. "api_key") and v is the value of the secret
            - If v is a string, every lm will use v for k
            - if v is a dict, each key of v should be the name of a named predictor
            (e.g. "my_module.predict", "my_module.summarizer") mapping to the secret value for that predictor
    Returns:
        A dictionary containing the state of the agent.json file with the secrets filled in.
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
                    f"{str(path)} exposes the secret key {kw}. Please remove it or ensure this file is not made public."
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
