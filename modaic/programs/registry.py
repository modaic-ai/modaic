# registry.py
import warnings
from importlib import import_module
from typing import Callable, Dict, NamedTuple, Tuple, Type


class Key(NamedTuple):
    name: str
    type: str  # yes, attribute name 'type' is fine here


class Registry:
    def __init__(self):
        self._paths: Dict[Key, Tuple[str, str]] = {}  # Key -> (module_path, qualname)
        self._cache: Dict[Key, Type] = {}
        self._frozen: bool = False

    def register(self, key: Key, cls: Type) -> None:
        if self._frozen:
            raise RuntimeError("Registry is frozen; no further registrations allowed.")
        if key in self._paths:
            mod, qual = self._paths[key]
            raise KeyError(f"Collision for {key}: already registered to {mod}:{qual}")
        if not isinstance(cls, type):
            raise TypeError("register() expects a class as the second argument.")

        module_path = cls.__module__
        qualname = cls.__qualname__  # supports nested classes
        self._paths[key] = (module_path, qualname)
        self._cache.pop(key, None)  # just in case

    def freeze(self) -> None:
        self._frozen = True

    def get(self, key: Key) -> Type:
        # Fast path
        if key in self._cache:
            return self._cache[key]

        try:
            module_path, qualname = self._paths[key]
        except KeyError:
            raise KeyError(f"Unknown key {key}. Was it registered before freeze()?") from None

        mod = import_module(module_path)
        obj = mod
        for part in qualname.split("."):  # walk nested qualnames safely
            obj = getattr(obj, part)

        if not isinstance(obj, type):
            raise TypeError(f"Resolved {module_path}:{qualname} is not a class.")

        self._cache[key] = obj
        return obj


# Instantiate per “kind”
ProgramRegistry = Registry()


def builtin_program(name: str) -> Callable[[Type], Type]:
    """Decorator to register a builtin module."""

    def _wrap(cls: Type) -> Type:
        key = Key(name, "program")
        ProgramRegistry.register(key, cls)
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
        key = Key(name, "program")
        ProgramRegistry.register(key, cls)
        return cls

    return _wrap


def builtin_indexer(name: str) -> Callable[[Type], Type]:
    def _wrap(cls: Type) -> Type:
        key = Key(name, "indexer")
        ProgramRegistry.register(key, cls)
        return cls

    return _wrap


def builtin_config(name: str) -> Callable[[Type], Type]:
    def _wrap(cls: Type) -> Type:
        key = Key(name, "config")
        ProgramRegistry.register(key, cls)
        return cls

    return _wrap
