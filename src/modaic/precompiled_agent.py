import json
from typing import Type, Dict, ClassVar, Optional, List
import pathlib
import dspy
from modaic.module_utils import create_modaic_temp_dir
from dataclasses import dataclass
from typing import get_args, get_origin
from .context.base import Context
from git import Repo
from pathlib import Path


@dataclass
class PrecompiledConfig:
    agent_type: ClassVar[str]
    indexer_type: ClassVar[str]

    def save_precompiled(
        self, path: str, _extra_auto_classes: Optional[Dict[str, object]] = None
    ) -> None:
        """
        Saves the config to a config.json file in the given path.

        Args:
            path: The path to save the config to.
        """
        # print(self.__dict__)
        # print(self.agent_type)
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        auto_classes = {"AutoConfig": self}
        if _extra_auto_classes is not None:
            auto_classes.update(_extra_auto_classes)

        auto_classes_paths = {k: _module_path(cls) for k, cls in auto_classes.items()}

        with open(path / "auto_classes.json", "w") as f:
            json.dump(auto_classes_paths, f, indent=2)

    @classmethod
    def from_precompiled(cls, path: str) -> "PrecompiledConfig":
        """
        Loads the config from a config.json file in the given path.

        Args:
            path: The path to load the config from.

        Returns:
            An instance of the PrecompiledConfig class.
        """
        path = pathlib.Path(path) / "config.json"
        with open(path, "r") as f:
            config_dict = json.load(f)
            return cls(**config_dict)

    @classmethod
    def from_dict(cls, dict: Dict) -> "PrecompiledConfig":
        """
        Loads the config from a dictionary.

        Args:
            dict: A dictionary containing the config.

        Returns:
            An instance of the PrecompiledConfig class.
        """
        instance = cls(**dict)
        return instance

    @classmethod
    def from_json(cls, path: str) -> "PrecompiledConfig":
        """
        Loads the config from a json file.

        Args:
            path: The path to load the config from.

        Returns:
            An instance of the PrecompiledConfig class.
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict:
        """
        Converts the config to a dictionary.
        """
        result = {}
        for field_name in self.__annotations__:
            result[field_name] = getattr(self, field_name)

        result["agent_type"] = self.agent_type
        return result

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for field_name, field in cls.__dataclass_fields__.items():
            if (
                field_name != "agent_type"
                and field_name != "indexer_type"
                and get_origin(field.type) is ClassVar
            ):
                raise TypeError(f"{cls.__name__} {field_name} must not be a ClassVar")

    def push_to_hub(
        self, repo_id: str, _extra_auto_classes: Optional[Dict[str, object]] = None
    ) -> None:
        """
        Pushes the config to the given repo_id.
        """
        repo_name = repo_id.split("/")[-1]
        temp_dir = create_modaic_temp_dir(repo_name)
        self.save_precompiled(temp_dir, _extra_auto_classes)
        _push_folder_to_hub(temp_dir, repo_id)


class PrecompiledAgent(dspy.Module):
    """
    Bases: `dspy.Module`
    """

    config_class: Type[PrecompiledConfig]

    def __init__(
        self,
        config: PrecompiledConfig,
        indexer: Optional["PrecompiledIndexer"] = None,
        **kwargs,
    ):
        self.config = config
        self.indexer = indexer
        assert config.agent_type == self.__class__.__name__, (
            f"Config agent_type must match agent class name. Expected {self.__class__.__name__}, got {config.agent_type}"
        )
        assert isinstance(config, self.config_class), (
            f"Config must be an instance of {self.config_class.__name__}"
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

    def save_precompiled(self, path: str) -> None:
        """
        Saves the agent and the config to the given path.

        Args:
            path: The path to save the agent and config to. Must be a local path.
        """
        path = pathlib.Path(path)
        extra_auto_classes = {"AutoAgent": self}
        if self.indexer is not None:
            extra_auto_classes["AutoIndexer"] = self.indexer
        self.config.save_precompiled(path, extra_auto_classes)
        self.save(path / "agent.json")

    @classmethod
    def from_precompiled(cls, path: str, **kwargs) -> "PrecompiledAgent":
        """
        Loads the agent and the config from the given path.

        Args:
            path: The path to load the agent and config from. Can be a local path or a path on Modaic Hub.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of the PrecompiledAgent class.
        """
        assert cls.config_class is not None, (
            f"Config class must be set for {cls.__name__}. \nHint: PrecompiledAgent.from_precompiled(path) will not work. You must use a subclass of PrecompiledAgent."
        )
        path = pathlib.Path(path)
        config = cls.config_class.from_precompiled(path)
        agent = cls(config, **kwargs)
        return agent

    def push_to_hub(self, repo_id: str) -> None:
        """
        Pushes the agent and the config to the given repo_id.

        Args:
            repo_id: The path on Modaic hub to save the agent and config to.
        """
        repo_name = repo_id.split("/")[-1]
        temp_dir = create_modaic_temp_dir(repo_name)
        self.save_precompiled(temp_dir)
        _push_folder_to_hub(temp_dir, repo_id)

    def __init_subclass__(cls, **kwargs):
        # Here we check that the subclass correctly links to it's config class
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "config_class"):
            raise TypeError(
                f"{cls.__name__} must define a class attribute 'config_class'"
            )
        if not issubclass(cls.config_class, PrecompiledConfig):
            raise TypeError(
                f"{cls.__name__} config_class must be a subclass of PrecompiledConfig"
            )


class PrecompiledIndexer:
    config_class: Type[PrecompiledConfig]

    def __init__(self, config: PrecompiledConfig, *args, **kwargs):
        self.config = config
        assert config.indexer_type == self.__class__.__name__, (
            f"Config indexer_type must match indexer class name. Expected {self.__class__.__name__}, got {config.indexer_type}"
        )
        assert isinstance(config, self.config_class), (
            f"Config must be an instance of {self.config_class.__name__}"
        )

    # @abstractmethod
    def ingest(self, contexts: List[Context], *args, **kwargs):
        pass

    def push_to_hub(self, repo_id: str) -> None:
        """
        Pushes the indexer and the config to the given repo_id.

        Args:
            repo_id: The path on Modaic hub to save the indexer and config to.
        """
        self.config.push_to_hub(repo_id, {"AutoIndexer": self})


def _module_path(instance: object) -> str:
    """
    Returns the module path of the given instance.
    Example:
        >>> _module_path(PrecompiledAgent)
        "modaic.precompiled_agent.PrecompiledAgent"
    """
    return f"{type(instance).__module__}.{type(instance).__name__}"


def _push_folder_to_hub(folder: Path, repo_id: str) -> None:
    """
    Pushes the folder to the given repo_id.
    """
    repo = Repo.init(folder)
