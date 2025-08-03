import json
from typing import Type, Dict
import pathlib
import dspy


class PrecompiledConfig:
    agent_type: str

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            try:
                setattr(self, k, v)
            except AttributeError as e:
                print(
                    f"Warning: {k} is not a valid attribute for {self.__class__.__name__}"
                )
                raise e

    def save_precompiled(self, path: str) -> None:
        """
        Saves the config to a config.json file in the given path.

        Args:
            path: The path to save the config to.
        """
        print(self.__dict__)
        print(self.agent_type)
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w") as f:
            json.dump(self.__dict__, f, indent=2)

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
            return cls.from_dict(json.load(f))

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "agent_type"):
            raise TypeError(
                f"{cls.__name__} must define a class attribute 'agent_type'"
            )


class PrecompiledAgent(dspy.Module):
    """
    Bases: `dspy.Module`
    """

    config_class: Type[PrecompiledConfig]

    def __init__(self, config: PrecompiledConfig, **kwargs):
        self.config = config
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
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_precompiled(path)
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
        pass

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
