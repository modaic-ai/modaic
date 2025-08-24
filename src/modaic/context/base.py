from enum import Enum
from typing import (
    Optional,
    Callable,
    List,
    TYPE_CHECKING,
    Union,
    ClassVar,
    Type,
    Any,
    Literal,
)
from abc import ABC, abstractmethod
import copy as c
from pydantic import BaseModel, Field, ConfigDict
from pydantic._internal._model_construction import ModelMetaclass
import weakref
import pydantic
import uuid
import PIL
from .query_language import Prop

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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


class ContextSchemaMeta(ModelMetaclass):
    def __getattr__(cls, name: str) -> Any:
        # 1) Let Pydantic's own metaclass handle private attrs etc.
        try:
            return ModelMetaclass.__getattr__(cls, name)
        except AttributeError:
            pass  # not a private attr; continue

        # 2) Safely look up fields without triggering descriptors or our __getattr__ again
        d = type.__getattribute__(cls, "__dict__")
        fields = d.get("__pydantic_fields__")
        if fields and name in fields:
            return Prop(name)  # FieldInfo (or whatever Pydantic stores)

        # 3) Not a field either
        raise AttributeError(name)


class ContextSchema(BaseModel, metaclass=ContextSchemaMeta):
    """
    Base class used to define the schema of a context object when they are serialized.

    Attributes:
        context_class: The class of the context object that this serialized context is for.
        id: The id of the serialized context.
        source: The source of the context object.
        metadata: The metadata of the context object.

    Example:
        In this example, `CaptionedImageSchema` stores the caption and the caption embedding while `CaptionedImage` is the `Context` class that is used to store the context object.
        Note that the image is loaded dynamically in the `CaptionedImage` class and is not serialized to `CaptionedImageSchema`.
        ```python
        from modaic.context import ContextSchema
        from modaic.types import String, Vector, Float16Vector

        class CaptionedImageSchema(ContextSchema):
            caption: String[100]
            caption_embedding: Float16Vector[384]
            image_path: String[100]

        class CaptionedImage(Atomic):
            schema = CaptionedImageSchema

            def __init__(self, image_path: str, caption: str, caption_embedding: np.ndarray, **kwargs):
                super().__init__(**kwargs)
                self.caption = caption
                self.caption_embedding = caption_embedding
                self.image_path = image_path
                self.image = PIL.Image.open(image_path)

            def embedme(self) -> PIL.Image.Image:
                return self.image
        ```
    """

    context_class: ClassVar[str]
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: Source = None
    metadata: dict = Field(default_factory=dict)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Allow class-header keywords without raising TypeError.

        Params:
            **kwargs: Arbitrary keywords from subclass declarations (e.g., type="Label").
        """
        super().__init_subclass__()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        if "type" in kwargs:
            cls._label = kwargs["type"]
        elif cls.__name__.endswith("Schema"):
            cls._label = cls.__name__[:-6]
        else:
            cls._label = cls.__name__

    def __lshift__(self, other: "Relationship"):
        assert isinstance(other, Relationship), (
            f"Cannot use '<<' between ContextSchema and object of type {other.__class__.__name__}"
        )
        edge = Edge(other, start_node=None, end_node=self)
        edge.set_left_sign("<<")
        return edge

    def __rshift__(self, other: "Relationship"):
        assert isinstance(other, Relationship), (
            f"Cannot use '>>' between ContextSchema and object of type {other.__class__.__name__}"
        )
        edge = Edge(other, start_node=self, end_node=None)
        edge.set_left_sign(">>")
        return edge

    def __str__(self) -> str:
        """
        Returns a string representation of the ContextSchema instance, including all field values.

        Returns:
            str: String representation with all field values.
        """
        values = self.model_dump()
        return f"{self.__class__._label}({values})"

    def __repr__(self):
        return self.__str__()


class Context(ABC):
    schema: ClassVar[Type[ContextSchema]] = NotImplemented

    def __init__(
        self, source: Optional[Source] = None, metadata: Optional[dict] = None
    ):
        """
        Args:
            source: The source of the context.
            metadata: The metadata of the context. If None, an empty dict is created
        """
        if metadata is None:
            metadata = {}
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

    def readme(self) -> str | pydantic.BaseModel:
        """
        How LLMs should read the context. By default returns self.serialize()

        Returns:
            LLM readable format of the context.
        """
        return self.serialize()

    def serialize(self) -> ContextSchema:
        """
        Serializes the context object into its associated `ContextSchema` object. Defined at self.schema.

        Returns:
            The serialized context object.
        """
        d = {}
        model_fields = self.schema.model_fields
        for k, v in self.__dict__.items():
            if k in model_fields:
                d[k] = v
        try:
            serialized = self.schema(**d)
        except pydantic.ValidationError as e:
            raise ValueError(
                f"""Failed to serialize class: {self.__class__.__name__} with params: {self.__dict__}. 
                
                Did you forget to add an attibute from {self.schema.__name__} to {self.__class__.__name__}? 
                
                Error: {e}
                """,
            )
        return serialized

    @classmethod
    def deserialize(cls, serialized: ContextSchema | dict, **kwargs):
        """
        Deserializes a `ContextSchema` object into a `Context` object.

        Args:
            serialized: The serialized context object or a dict.
            **kwargs: Additional keyword arguments to pass to the Context object's constructor. (will overide any attributes set in the ContextSchema object)

        Returns:
            The deserialized context object.
        """
        assert isinstance(serialized, (ContextSchema, dict)), (
            "serialized must be a ContextSchema object or a dict"
        )
        if isinstance(serialized, dict):
            serialized = cls.schema.model_validate(serialized)
        try:
            payload = serialized.model_dump()
            # Preserve nested BaseModel instances where appropriate (e.g., Source)
            if "source" in payload:
                payload["source"] = serialized.source
            return cls(**{**payload, **kwargs})
        except Exception as e:  # noqa
            raise ValueError(
                f"""Invalid ContextSchema: {serialized}. Could not initialize class {cls.__name__} with params {serialized.model_dump()}
                Error: {e}
                """
            )

    def set_source(self, source: Source, copy: bool = False):
        """
        Sets the source of the context object.

        Args:
            source: Source - The source of the context object.
            copy: bool - Whether to copy the source object to make it safe to mutate.
        """
        self.source = c.deepcopy(source) if copy else source

    def set_metadata(self, metadata: dict, copy: bool = False):
        """
        Sets the metadata of the context object.

        Args:
            metadata: The metadata of the context object.
            copy: Whether to copy the metadata object to make it safe to mutate.
        """
        self.metadata = c.deepcopy(metadata) if copy else metadata

    def add_metadata(self, metadata: dict):
        """
        Adds metadata to the context object.
        Args:
            metadata: The metadata to add to the context object.
        """
        self.metadata.update(metadata)

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        """
        Deserializes a dict into a `Context` object.

        Args:
            d: The dict to deserialize.
            **kwargs: Additional keyword arguments to pass to the Context object's constructor. (will overide any attributes set in the dict)

        Returns:
            The deserialized context object.
        """
        return cls.deserialize(ContextSchema.from_dict(d), **kwargs)

    def __str__(self) -> str:
        """
        Returns a string representation of the context object, including the class name and its field values.

        Returns:
            str: The string representation of the object.
        """
        class_name = self.__class__.__name__
        field_vals = "\n\t".join(
            [
                f"{class_name}.{k}={v},"
                for k, v in self.__dict__.items()
                if k in self.schema.model_fields
            ]
        )
        return f"{class_name}(\n\t{field_vals}\n\t)"

    def __repr__(self):
        return self.__str__()


class Atomic(Context):
    """
    Base class for all Atomic Context objects. Atomic objects represent context at its finest granularity and are not chunkable.

    Example:
        In this example, `CaptionedImage` is an `Atomic` context object that stores the caption and the caption embedding.
        ```python
        from modaic.context import ContextSchema
        from modaic.types import String, Vector, Float16Vector

        class CaptionImageSchema(ContextSchema):
            caption: String[100]
            caption_embedding: Float16Vector[384]
            image_path: String[100]

        class CaptionedImage(Atomic):
            schema = CaptionImageSchema

            def __init__(self, image_path: str, caption: str, caption_embedding: np.ndarray, **kwargs):
                super().__init__(**kwargs)
                self.caption = caption
                self.caption_embedding = caption_embedding
                self.image_path = image_path
                self.image = PIL.Image.open(image_path)

            def embedme(self) -> PIL.Image.Image:
                return self.image
        ```
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# TODO add support for PIL.Image.Image and Video embed types we'll need to replace dspy.Embedder with a more general embedder
class Molecular(Context):
    """
    Base class for all `Molecular` Context objects. `Molecular` context objects represent context that can be chunked into smaller `Molecular` or `Atomic` context objects.

    Example:
        In this example, `MarkdownDoc` is a `Molecular` context object that stores a markdown document.
        ```python
        from modaic.context import Molecular
        from modaic.types import String, Vector, Float16Vector
        from langchain_text_splitters import MarkdownTextSplitter
        from modaic.context import Text

        class MarkdownDocSchema(ContextSchema):
            markdown: String

        class MarkdownDoc(Molecular):
            schema = MarkdownDocSchema

            def chunk(self):
                # Split the markdown into chunks of 1000 characters
                splitter = MarkdownTextSplitter()
                chunk_fn = lambda mdoc: [Text(text=t) for t in splitter.split_text(mdoc.markdown)]
                self.chunk_with(chunk_fn)

            def __init__(self, markdown: str, **kwargs):
                super().__init__(**kwargs)
                self.markdown = markdown
        ```

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunks: List[Context] = []

    def chunk_with(
        self,
        chunk_fn: str | Callable[[Context], List[Context]],
        set_source: bool = True,
        **kwargs,
    ):
        """
        Chunk the context object into smaller Context objects.

        Args:
            chunk_fn: The function to use to chunk the context object. The function should take in a specific type of Context object and return a list of Context objects.
            set_source: bool - Whether to automatically set the source of the chunks using the Context object. (sets chunk.source to self.source, sets chunk.source.parent to self, and updates the chunk.source.metadata with the chunk_id)
            **kwargs: dict - Additional keyword arguments to pass to the chunking function.
        """
        self.chunks = chunk_fn(self, **kwargs)
        if set_source:
            for i, chunk in enumerate(self.chunks):
                metadata = c.deepcopy(self.source.metadata) if self.source else {}
                Molecular.update_chunk_id(metadata, i)
                source = Source(
                    origin=self.source.origin if self.source else None,
                    type=self.source.type if self.source else None,
                    parent=self,
                    metadata=metadata,
                )
                chunk.set_source(source)

    def apply_to_chunks(self, apply_fn: Callable[[Context], None], **kwargs):
        """
        Applies apply_fn to each chunk in chunks.

        Args:
            apply_fn: The function to apply to each chunk. Function should take in a Context object and mutate it.
            **kwargs: Additional keyword arguments to pass to apply_fn.
        """
        for chunk in self.chunks:
            apply_fn(chunk, **kwargs)

    @staticmethod
    def update_chunk_id(metadata: dict, chunk_id: int):
        if "chunk_id" in metadata and isinstance(metadata["chunk_id"], int):
            metadata["chunk_id"] = {"id": metadata["chunk_id"], "chunk_id": chunk_id}
        elif "chunk_id" in metadata and isinstance(metadata["chunk_id"], dict):
            Molecular.update_chunk_id(metadata["chunk_id"], chunk_id)
        else:
            metadata["chunk_id"] = chunk_id


class Relationship(ContextSchema):
    """
    Base class for all Relationship objects.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        if "type" in kwargs:
            cls._label = kwargs["type"]
        else:
            cls._label = cls.__name__

    def __rshift__(self, other: ContextSchema):
        raise ValueError("""Cannot start OGM expression with Relationship object. Relationship object must always be in the middle. Ensure your expressions look like the following only:
                         ContextSchema << / >> Relationship << / >> ContextSchema 
                         """)

    def __lshift__(self, other: ContextSchema):
        raise ValueError("""Cannot start OGM expression with Relationship object. Relationship object must always be in the middle. Ensure your expressions look like the following only:
                         ContextSchema << / >> Relationship << / >> ContextSchema 
                         """)

    def __str__(self):
        """
        Returns a string representation of the Relationship object, including all fields and their values.

        Returns:
            str: String representation of the Relationship object with all fields and their values.
        """
        fields_repr = ", ".join(f"{k}={repr(v)}" for k, v in self.model_dump().items())
        return f"{self.__class__._label}({fields_repr})"

    def __repr__(self):
        return self.__str__()


class Edge:
    def __init__(
        self,
        relationship: Relationship,
        *,
        start_node: Optional[ContextSchema] = None,
        end_node: Optional[ContextSchema] = None,
    ):
        self.start_node = start_node
        self.relationship = relationship
        self.end_node = end_node
        self.left_sign = None
        self.right_sign = None

    def set_left_sign(self, sign: Literal["<<", ">>"]):
        self.left_sign = sign

    def set_right_sign(self, sign: Literal["<<", ">>"]):
        self.right_sign = sign

    def set_start_node(self, start_node: ContextSchema):
        self.start_node = start_node

    def set_end_node(self, end_node: ContextSchema):
        self.end_node = end_node

    def __lshift__(self, other: ContextSchema):
        assert isinstance(
            other, ContextSchema
        ), f"""Cannot use '<<' between Edge and object of type {other.__class__.__name__}. Ensure your expressions look like the following only:
            ContextSchema << / >> Relationship << / >> ContextSchema
            """
        # Throw error for >> << edge case
        if self.left_sign == ">>":
            left_node = self.start_node if self.start_node else self.end_node
            right_node = other
            raise ValueError(
                f"Unrecognized edge type ({left_node.__class__.__name__} >> {self.relationship.__class__.__name__} << {right_node.__class__.__name__})"
            )
        self.set_right_sign("<<")
        # Try to set it to the start_node. If it's already set, its an undirected edge.
        if self.start_node is None:
            self.set_start_node(other)
        elif self.end_node is None:
            self.set_end_node(other)
        else:
            raise ValueError(
                """Edge is already completed. Cannnot chain << and >> operators. 
                Ensure your expressions look like the following only:
                ContextSchema << / >> Relationship << / >> ContextSchema
                """
            )
        return self

    def __rshift__(self, other: ContextSchema):
        assert isinstance(
            other, ContextSchema
        ), f"""Cannot use '>>' between Edge and object of type {other.__class__.__name__}. Ensure your expressions look like the following only:
            ContextSchema << / >> Relationship << / >> ContextSchema
            """
        self.set_right_sign(">>")
        # Try to set it to the end_node. If it's already set, its an undirected edge.
        if self.end_node is None:
            self.set_end_node(other)
        elif self.start_node is None:
            self.set_start_node(other)
        else:
            raise ValueError(
                """Edge is already completed. Cannnot chain << and >> operators. 
                Ensure your expressions look like the following only:
                ContextSchema << / >> Relationship << / >> ContextSchema
                """
            )
        return self

    def __repr__(self) -> str:
        """
        Returns a string representation of the Edge, showing the label and id for each of start_node, relationship, and end_node,
        as well as the left and right signs (or '??' if not set).

        Returns:
            str: String representation of the Edge with labels, ids, and signs.
        """

        def label_and_id(obj):
            if obj is None:
                return "None"
            label = getattr(obj.__class__, "_label", obj.__class__.__name__)
            obj_id = getattr(obj, "id", None)
            return f"{label}({obj_id})"

        left_sign = self.left_sign if self.left_sign is not None else "??"
        right_sign = self.right_sign if self.right_sign is not None else "??"
        start_str = label_and_id(self.start_node)
        relationship_str = label_and_id(self.relationship)
        end_str = label_and_id(self.end_node)
        return (
            f"Edge({start_str} {left_sign} {relationship_str} {right_sign} {end_str})"
        )

    def __str__(self):
        return repr(self)

    @property
    def directed(self):
        if self.left_sign is None or self.right_sign is None:
            raise ValueError(
                "Attempted to access 'directed' property of Edge object that is not fully initialized. Please set left_sign and right_sign."
            )
        return self.left_sign == self.right_sign
