from enum import Enum
from typing import Optional, Callable, List, TYPE_CHECKING, Union, ClassVar, Type
from enum import auto
from abc import ABC, abstractmethod
import copy
import inspect
import warnings
from pydantic import BaseModel, Field, create_model
import weakref
import pydantic
import uuid
import PIL

if TYPE_CHECKING:
    from modaic.databases.database import ContextDatabase
    from modaic.databases.sql_database import SQLDatabase


class SourceType(Enum):
    LOCAL_PATH = "local_path"
    URL = "url"
    SQL_DB = "sql_db"


class Source(BaseModel):
    origin: Optional[str] = None
    type: Optional[SourceType] = None
    metadata: dict = Field(default_factory=dict)

    def __init__(
        self,
        origin: Optional[str] = None,
        type: Optional[SourceType] = None,
        parent: Union["Context", "ContextDatabase", "SQLDatabase", None] = None,
        metadata: dict = None,
    ):
        if metadata is None:
            metadata = {}

        # Initialize the BaseModel with the serializable fields
        super().__init__(origin=origin, type=type, metadata=metadata)

        # Set the weakref separately (not validated/serialized)
        object.__setattr__(self, "_parent", weakref.ref(parent) if parent else None)

    @property
    def parent(self):
        return self._parent() if self._parent else None

    def model_dump(self, **kwargs):
        """Override model_dump method to exclude _parent field"""
        result = super().model_dump(**kwargs)
        result.pop("_parent", None)
        return result

    def model_dump_json(self, **kwargs):
        """Override model_dump_json method to exclude _parent field"""
        return super().model_dump_json(exclude={"_parent"}, **kwargs)

    class Config:
        # Allow arbitrary types (for SourceType enum)
        arbitrary_types_allowed = True


class SerializedContext(BaseModel):
    context_class: ClassVar[str]
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: Source
    metadata: dict


class Context(ABC):
    serialized_context_class: ClassVar[Type[SerializedContext]] = NotImplemented

    def __init__(self, source: Optional[Source] = None, metadata: dict = {}):
        self.source = source
        self.metadata = metadata

    @abstractmethod
    def embedme(self) -> str | PIL.Image.Image:
        """
        Abstract method defined by all subclasses of `Context` to define how embedding modeles should embed the context.
        Returns:
            The string or image that should be used to embed the context.
        """
        pass

    @abstractmethod
    def readme(self) -> str:
        """
        Abstract method defined by all subclasses of `Context` to define how LLMs should read the context.
        Returns:
            The string that should be read by LLMs.
        """
        pass

    def serialize(self) -> SerializedContext:
        d = {}
        model_fields = self.serialized_context_class.model_fields
        for k, v in self.__dict__.items():
            if k in model_fields:
                d[k] = v
        try:
            serialized = self.serialized_context_class(**d)
        except pydantic.ValidationError as e:
            raise ValueError(
                f"""Failed to serialize class: {self.__class__.__name__} with params: {self.__dict__}. 
                
                Did you forget to add an attibute from {self.serialized_context_class.__name__} to {self.__class__.__name__}? 
                
                Error: {e}
                """,
            )
        return serialized

    @classmethod
    def deserialize(cls, serialized: SerializedContext | dict, **kwargs):
        assert isinstance(serialized, (SerializedContext, dict)), (
            "serialized must be a SerializedContext object or a dict"
        )
        if isinstance(serialized, dict):
            serialized = cls.serialized_context_class.model_validate(serialized)
        try:
            return cls(**{**serialized.model_dump(), **kwargs})
        except Exception as e:  # noqa
            raise ValueError(
                f"""Invalid SerializedContext: {serialized}. Could not initialize class {cls.__name__} with params {serialized.model_dump()}
                Error: {e}
                """
            )

    def set_source(self, source: Source, copy: bool = False):
        self.source = copy.deepcopy(source) if copy else source

    def set_metadata(self, metadata: dict, copy: bool = False):
        self.metadata = copy.deepcopy(metadata) if copy else metadata

    def add_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return cls.deserialize(SerializedContext.from_dict(d), **kwargs)

    def __str__(self):
        field_vals = "\n\t".join(
            [
                f"{k}={v},"
                for k, v in self.__dict__.items()
                if k in self.serialized_context_class.model_fields
            ]
        )
        return f"""{self.__class__.__name__}(\n\t{field_vals}\n\t)"""

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
            for i, chunk in enumerate(self._chunks):
                metadata = copy.deepcopy(self.source.metadata) if self.source else {}
                Molecular.update_chunk_id(metadata, i)
                source = Source(
                    origin=self.source.origin if self.source else None,
                    type=self.source.type if self.source else None,
                    parent=self,
                    metadata=metadata,
                )
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

    @staticmethod
    def update_chunk_id(metadata: dict, chunk_id: int):
        if "chunk_id" in metadata and isinstance(metadata["chunk_id"], int):
            metadata["chunk_id"] = {"id": metadata["chunk_id"], "chunk_id": chunk_id}
        elif "chunk_id" in metadata and isinstance(metadata["chunk_id"], dict):
            Molecular.update_chunk_id(metadata["chunk_id"], chunk_id)
        else:
            metadata["chunk_id"] = chunk_id
