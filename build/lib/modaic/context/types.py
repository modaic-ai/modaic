from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Callable, List
from enum import auto
from abc import ABC, abstractmethod
import copy
import inspect


def serializable(method):
    method._is_serializable = True
    return method


class SourceType(Enum):
    FILE_OBJECT = auto()
    LOCAL_PATH = auto()
    URL = auto()
    SQL_DB = auto()


@dataclass
class Source:
    origin: Any
    type: SourceType
    metadata: dict = field(default_factory=dict)


class SerializedContext:
    def __init__(
        self, context_class: str, metadata: dict = {}, source: Source = None, **kwargs
    ):
        self.context_class: str = context_class
        self.metadata: dict = metadata
        self.source: Source = source
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        vals = "\n\t".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"{self.__class__.__name__}(\n\t{vals}\n)"

    @classmethod
    def from_dict(cls, d: dict):
        new_d = {}
        match d:
            case {"source": source_dict, "metadata": metadata, **rest_of_dict}:
                match source_dict:
                    case {
                        "type": source_type,
                        "origin": origin_dict,
                        "metadata": metadata_dict,
                    }:
                        type = SourceType[source_type]
                        new_source = Source(
                            type=type, origin=origin_dict, metadata=metadata_dict
                        )
                        new_d = {
                            **rest_of_dict,
                            "source": new_source,
                            "metadata": metadata,
                        }
                        return cls(**new_d)

        raise ValueError(f"Invalid SerializedContext dictionary format: {d}")

    def to_dict(self):
        match self.__dict__:
            case {"source": source, "metadata": metadata, **rest_of_dict}:
                match source:
                    case Source(
                        type=source_type, origin=origin, metadata=source_metadata
                    ):
                        source_dict = {
                            "type": source_type.name,
                            "origin": origin,
                            "metadata": source_metadata,
                        }
                        return {
                            "source": source_dict,
                            "metadata": metadata,
                            **rest_of_dict,
                        }

        raise ValueError(f"Invalid SerializedContext: {self}")

    def copy(self):
        return copy.deepcopy(self)


class Context(ABC):
    def __init__(self, source: Optional[Source] = None, metadata: dict = {}):
        self.source = source
        self.metadata = metadata

    @abstractmethod
    def embedme(self) -> str:
        pass

    @abstractmethod
    def readme(self):
        pass

    def serialize(self) -> SerializedContext:
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                d[k] = v

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and getattr(attr, "_is_serializable", False):
                try:
                    d[attr.__name__] = attr()
                except TypeError:
                    raise TypeError(
                        f"Cannot serializing function {self.__class__.__name__}::{attr_name}. Functions decorated with `@serializable` must take no arguments besides `self`."
                    )

        return SerializedContext(context_class=self.__class__.__name__, **d)

    @classmethod
    def deserialize(cls, serialized: SerializedContext, **kwargs):
        try:
            match serialized.to_dict():
                case {"source": source, **rest_of_dict}:
                    match source:
                        case {
                            "type": source_type,
                            "origin": origin,
                            "metadata": source_metadata,
                        }:
                            type = SourceType[source_type]
                            source = Source(
                                type=type, origin=origin, metadata=source_metadata
                            )
                            sig = inspect.signature(cls)
                            valid_params = set(sig.parameters.keys()) - {"self"}
                            filtered_kwargs = {
                                k: v
                                for k, v in rest_of_dict.items()
                                if k in valid_params
                            }
                            return cls(source=source, **filtered_kwargs)
        except:  # noqa
            raise ValueError(
                f"Invalid SerializedContext: {serialized}. Could not initialize class {cls.__name__} with params {serialized.to_dict()}"
            )

    def set_source(self, source: Source):
        self.source = source

    def set_metadata(self, metadata: dict):
        self.metadata = metadata

    def add_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return cls.deserialize(SerializedContext.from_dict(d), **kwargs)

    def resolve_source(self) -> str:
        """
        Get the top level origin of the context.

        Returns:
            The top level origin of the context, or None if no source is set.
        """
        if self.source is None:
            return None
        elif self.source.type == SourceType.CONTEXT_OBJECT:
            return self.source.origin.resolve_source()
        else:
            return self.source.origin

    def __str__(self):
        return (
            f"{self.__class__.__name__}(source={self.source}, metadata={self.metadata})"
        )

    def __repr__(self):
        return self.__str__()


class Atomic(Context):
    """
    Atomic Context is a single piece of information that can be embedded and used for retrieval.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# TODO add support for PIL.Image.Image and Video embed types we'll need to replace dspy.Embedder with a more general embedder
class Molecular(Context):
    """
    Molecular context objects can be chunked into smaller Context objects.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chunks: List[Context] = []

    def chunk(
        self,
        chunk_fn: str | Callable[[Context], List[Context]],
        set_source: bool = True,
        **kwargs,
    ) -> bool:
        """
        Chunk the context object into smaller Context objects.
        Args:
            chunk_fn: The function to use to chunk the context object. The function should take in a Context object and return a list of Context objects.
            **kwargs: dict - Additional keyword arguments to pass to the chunking function.
        Returns:
            bool - True if the chunking function was found, False otherwise.
        """
        self._chunks = chunk_fn(self, **kwargs)
        if set_source:
            for chunk in self._chunks:
                source = copy.deepcopy(self.source)
                chunk.set_source(source)
        return True

    def apply_to_chunks(self, apply_fn: Callable[[Context], None], **kwargs):
        """
        Applies apply_fn to each chunk in chunks.

        Args:
            apply_fn: The function to apply to each chunk. Function should take in a Context object and mutate it.
            **kwargs: Additional keyword arguments to pass to apply_fn.
        """
        for chunk in self._chunks:
            apply_fn(chunk, **kwargs)

    def get_chunks(self):
        return self._chunks
