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
    Dict,
    get_origin,
    get_args,
    Annotated,
    Literal,
    Final,
    ClassVar,
)
from types import UnionType
from abc import ABC, abstractmethod
import copy as c
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
    AfterValidator,
    field_serializer,
)
from pydantic.v1 import Field as V1Field
from pydantic._internal._model_construction import ModelMetaclass
import weakref
import pydantic
import uuid
import PIL
from .query_language import Prop
import warnings
import types

if TYPE_CHECKING:
    import gqlalchemy
    from modaic.databases.database import ContextDatabase
    from modaic.databases.sql_database import SQLDatabase
    from modaic.databases.graph_database import GraphDatabase


GQLALCHEMY_EXCLUDED_FIELDS = [
    "id",
    "_gqlalchemy_id",
    "_type_registry",
    "_labels",
    "_gqlalchemy_class_registry",
    "_type",
]


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

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = Field(default_factory=dict)
    source: Optional[Source] = None

    _gqlalchemy_id: Optional[int] = PrivateAttr(default=None)

    # CAVEAT: All ContextSchema subclasses share the same instance of _type_registry. This is intentional.
    _type_registry: ClassVar[Dict[str, Type["ContextSchema"]]] = {}
    _labels: ClassVar[frozenset[str]] = frozenset()
    _gqlalchemy_class_registry: ClassVar[
        Dict[str, Type["gqlalchemy.models.GraphObject"]]
    ] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Allow class-header keywords without raising TypeError.

        Params:
            **kwargs: Arbitrary keywords from subclass declarations (e.g., type="Label").
        """
        super().__init_subclass__()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        if "type" in kwargs:
            cls._type = kwargs["type"]
        elif cls.__name__.endswith("Schema"):
            cls._type = cls.__name__[:-6]
        else:
            cls._type = cls.__name__

        assert cls._type != "Node" and cls._type != "Relationship", (
            f"Class {cls.__name__} cannot use name 'Node' or 'Relationship' as type. Please use a different name. You can use a custom type by using the 'type' keyword. Example: `class {cls.__name__}(ContextSchema, type=<custom_type_name>)`"
        )

        # TODO: revisit this. Should we allow multiple parents?
        # Get parent class labels
        parent_labels = [
            b._labels for b in cls.__bases__ if issubclass(b, ContextSchema)
        ]
        assert len(parent_labels) == 1, (
            f"ContextSchema class {cls.__name__} cannot have multiple ContextSchema classes as parents. Should it? Submit an issue to tell us about your use case. https://github.com/modaic-ai/modaic/issues"
        )
        cls._labels = frozenset({cls._type}) | parent_labels[0]
        assert cls._type not in cls._type_registry, (
            f"Cannot have multiple ContextSchema/Relation classes with type = '{cls._type}'"
        )
        cls._type_registry[cls._type] = cls

    def __str__(self) -> str:
        """
        Returns a string representation of the ContextSchema instance, including all field values.

        Returns:
            str: String representation with all field values.
        """
        values = self.model_dump()
        return f"{self.__class__._type}({values})"

    def __repr__(self):
        return self.__str__()

    def to_gqlalchemy(self, db: "GraphDatabase") -> "gqlalchemy.Node":
        """
        Convert the ContextSchema object to a GQLAlchemy object.
        """
        try:
            import gqlalchemy
            from modaic.databases.graph_database import GraphDatabase
        except ImportError:
            raise ImportError(
                "GQLAlchemy is not installed. Please install the graph extension for modaic with `uv add modaic[graph]`"
            )
        assert isinstance(db, GraphDatabase), (
            "Expected db to be a modaic.databases.GraphDatabase instance. "
            f"Got {type(db)} instead."
        )
        cls = self.__class__

        # Dynamically create a GQLAlchemy Node class for the ContextSchema if it doesn't exist
        if cls._type not in cls._gqlalchemy_class_registry:
            field_annotations = get_annotations(
                cls,
                exclude=GQLALCHEMY_EXCLUDED_FIELDS,
            )
            field_defaults = get_defaults(cls, exclude=GQLALCHEMY_EXCLUDED_FIELDS)
            gqlalchemy_class = type(
                f"{cls.__name__}Node",
                (gqlalchemy.Node,),
                {
                    "__annotations__": {**field_annotations, "modaic_id": str},
                    "modaic_id": V1Field(unique=True, db=db._client),
                    **field_defaults,
                },
                label=cls._type,
            )
            cls._gqlalchemy_class_registry[cls._type] = gqlalchemy_class
        # Return a new GQLAlchemy Node object
        gqlalchemy_class = cls._gqlalchemy_class_registry[cls._type]
        if self._gqlalchemy_id is None:
            return gqlalchemy_class(
                _labels=set(self._labels),
                modaic_id=self.id,
                **self.model_dump(exclude={"id"}),
            )
        else:
            return gqlalchemy_class(
                _labels=set(self._labels),
                modaic_id=self.id,
                _id=self._gqlalchemy_id,
                **self.model_dump(exclude={"id"}),
            )

    @classmethod
    def from_gqlalchemy(cls, gqlalchemy_node: "gqlalchemy.Node") -> "ContextSchema":
        """
        Convert a GQLAlchemy Node into a `ContextSchema` instance. If cls is the ContextSchema class itself, it will return the best subclass of ContextSchema that matches the labels of the GQLAlchemy Node.
        Args:
            gqlalchemy_node: The GQLAlchemy Node to convert.

        Returns:
            The converted ContextSchema or ContextSchema subclass instance.
        """
        if cls is not ContextSchema:
            assert cls._type in gqlalchemy_node._labels, (
                f"Cannot convert GQLAlchemy Node {gqlalchemy_node} to {cls.__name__} because it does not have the label '{cls._type}'"
            )

            try:
                kwargs = {**gqlalchemy_node._properties}
                modaic_id = kwargs.pop("modaic_id")
                kwargs["id"] = modaic_id
                context_obj = cls(**kwargs)
                context_obj._gqlalchemy_id = gqlalchemy_node._id
                return context_obj
            except ValidationError as e:
                raise ValueError(
                    f"Failed to convert GQLAlchemy Node {gqlalchemy_node} to {cls.__name__} because it does not have the required fields.\nError: {e}"
                )

        # If cls is ContextSchema, we need to find the best subclass of ContextSchema that matches the labels of the GQLAlchemy Node.
        best_subclass = None
        for label in gqlalchemy_node._labels:
            if current_subclass := ContextSchema._type_registry.get(label):
                # check if the current subclass has more parents than the best subclass
                if best_subclass is None or len(current_subclass.__mro__) > len(
                    best_subclass.__mro__
                ):
                    best_subclass = current_subclass

        if best_subclass is None:
            raise ValueError(
                f"Cannot convert GQLAlchemy Node {gqlalchemy_node} to a ContextSchema, no matching ContextSchema class found with type from '{gqlalchemy_node._labels}'"
            )
        return best_subclass.from_gqlalchemy(gqlalchemy_node)

    def save(self, db: "GraphDatabase"):
        """
        Save the ContextSchema object to the graph database.
        """
        try:
            from modaic.databases.graph_database import GraphDatabase
        except ImportError:
            raise ImportError(
                "GQLAlchemy is not installed. Please install the graph extension for modaic with `uv add modaic[graph]`"
            )

        assert isinstance(db, GraphDatabase), (
            "Expected db to be a modaic.databases.GraphDatabase instance. "
            f"Got {type(db)} instead."
        )

        result = db.save_node(self)

        for k in self.model_dump(exclude={"id"}):
            setattr(self, k, getattr(result, k))
        self._gqlalchemy_id = result._id

    def load(self, database: "GraphDatabase"):
        """
        Loads a node from Memgraph.
        If the node._id is not None it fetches the node from Memgraph with that
        internal id.
        If the node has unique fields it fetches the node from Memgraph with
        those unique fields set.
        Otherwise it tries to find any node in Memgraph that has all properties
        set to exactly the same values.
        If no node is found or no properties are set it raises a GQLAlchemyError.
        """
        raise NotImplementedError("Not implemented")


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


class RelationMeta(ContextSchemaMeta):
    def __new__(cls, name, bases, dct):
        # Make Relation class allow extra fields but subclasses default to ignore (pydantic default)
        # BUG: Doesn't allow users to define their own "extra" behavior
        if name == "Relation":
            dct["model_config"] = ConfigDict(extra="allow")
        elif "model_config" not in dct:
            dct["model_config"] = ConfigDict(extra="ignore")
        elif dct["model_config"].get("extra", None) is None:
            dct["model_config"]["extra"] = "ignore"

        return super().__new__(cls, name, bases, dct)


class Relation(ContextSchema, metaclass=RelationMeta):
    """
    Base class for all Relation objects.
    """

    start_node: Optional[ContextSchema | int] = None
    end_node: Optional[ContextSchema | int] = None

    @property
    def start_node_gql_id(self) -> int:
        return self.node_to_gql_id(self.start_node)

    @property
    def end_node_gql_id(self) -> int:
        return self.node_to_gql_id(self.end_node)

    @field_serializer("start_node", "end_node")
    def node_to_gql_id(self, node: ContextSchema | int) -> int:
        if isinstance(node, ContextSchema):
            return node._gqlalchemy_id
        else:
            return node

    def get_start_node(self, db: "GraphDatabase") -> ContextSchema:
        """
        Get the start node object of the relation as a ContextSchema object.
        Args:
            db: The GraphDatabase instance to use to fetch the start node.

        Returns:
            The start node object as a ContextSchema object.
        """
        if isinstance(self.start_node, ContextSchema):
            return self.start_node
        else:
            return ContextSchema.from_gqlalchemy(
                next(
                    db.execute_and_fetch(
                        f"MATCH (n) WHERE id(n) = {self.start_node} RETURN n"
                    )
                )
            )

    def get_end_node(self, db: "GraphDatabase") -> ContextSchema:
        """
        Get the end node object of the relation as a ContextSchema object.
        Args:
            db: The GraphDatabase instance to use to fetch the end node.

        Returns:
            The end node object as a ContextSchema object.
        """
        if isinstance(self.end_node, ContextSchema):
            return self.end_node
        else:
            return ContextSchema.from_gqlalchemy(
                next(
                    db.execute_and_fetch(
                        f"MATCH (n) WHERE id(n) = {self.start_node} RETURN n"
                    )
                )
            )

    @field_validator("start_node", "end_node")
    def check_node(cls, v):
        assert isinstance(v, (ContextSchema, int)), (
            f"start_node/end_node must be a ContextSchema or int, got {type(v)}: {v}"
        )
        assert not isinstance(v, Relation), (
            f"start_node/end_node cannot be a Relation object: {v}"
        )
        return v

    @model_validator(mode="after")
    def post_init(self):
        # Sets type for inline declaration of Relation objects
        if type(self) is Relation:
            assert "_type" in self.model_dump(), (
                "Inline declaration of Relation objects must specify the '_type' field"
            )
            self._type = self.model_dump()["_type"]
        return self

    # other >> self
    def __rrshift__(self, other: ContextSchema | int):
        # left_node >> self >> right_node
        self.start_node = other
        return self

    # self >> other
    def __rshift__(self, other: ContextSchema | int):
        # left_node >> self >> right_node
        self.end_node = other
        return self

    # other << self
    def __rlshift__(self, other: ContextSchema | int):
        # left_node << self << right_node
        self.end_node = other
        return self

    # self << other
    def __lshift__(self, other: ContextSchema | int):
        # left_node << self << right_node
        self.start_node = other
        return self

    def __str__(self):
        """
        Returns a string representation of the Relation object, including all fields and their values.

        Returns:
            str: String representation of the Relation object with all fields and their values.
        """
        fields_repr = ", ".join(f"{k}={repr(v)}" for k, v in self.model_dump().items())
        return f"{self.__class__._type}({fields_repr})"

    def __repr__(self):
        return self.__str__()

    def to_gqlalchemy(self, db: "GraphDatabase") -> "gqlalchemy.Relationship":
        """
        Convert the ContextSchema object to a GQLAlchemy object.

        Args:
            db: The GraphDatabase instance to use to save the start_node and end_node if they are not already saved.

        Returns:
            The GQLAlchemy Relationship object.

        Raises:
            AssertionError: If db is not a modaic.databases.GraphDatabase instance.
            ImportError: If GQLAlchemy is not installed.

        !!! warning
            Saves the start_node and end_node to the database if they are not already saved.
        """
        try:
            import gqlalchemy
            from modaic.databases.graph_database import GraphDatabase
        except ImportError:
            raise ImportError(
                "GQLAlchemy is not installed. Please install the graph extension for modaic with `uv add modaic[graph]`"
            )

        assert isinstance(db, GraphDatabase), (
            "Expected db to be a modaic.databases.GraphDatabase instance. "
            f"Got {type(db)} instead."
        )

        cls = self.__class__

        # Dynamically create a GQLAlchemy Node class for the ContextSchema if it doesn't exist
        if self._type not in cls._gqlalchemy_class_registry:
            ad_hoc_annotations = get_ad_hoc_annotations(self) if cls is Relation else {}
            field_annotations = get_annotations(
                cls,
                exclude=GQLALCHEMY_EXCLUDED_FIELDS + ["start_node", "end_node"],
            )
            field_defaults = get_defaults(
                cls, exclude=GQLALCHEMY_EXCLUDED_FIELDS + ["start_node", "end_node"]
            )
            gqlalchemy_class = type(
                f"{cls.__name__}Rel",
                (gqlalchemy.Relationship,),
                {
                    "__annotations__": {
                        **ad_hoc_annotations,
                        **field_annotations,
                        "modaic_id": str,
                    },
                    "modaic_id": V1Field(unique=True, db=db._client),
                    **field_defaults,
                },
                type=self._type,
            )
            cls._gqlalchemy_class_registry[self._type] = gqlalchemy_class

        gqlalchemy_class = cls._gqlalchemy_class_registry[self._type]

        if self.start_node is not None and self.start_node_gql_id is None:
            self.start_node.save(db)
        if self.end_node is not None and self.end_node_gql_id is None:
            self.end_node.save(db)

        if self._gqlalchemy_id is None:
            return gqlalchemy.Relationship.parse_obj(
                {
                    "_type": self._type,
                    "modaic_id": self.id,
                    "_start_node_id": self.start_node_gql_id,
                    "_end_node_id": self.end_node_gql_id,
                    **self.model_dump(
                        exclude={"id", "start_node", "end_node", "_type"}
                    ),
                }
            )
        else:
            return gqlalchemy.Relationship.parse_obj(
                {
                    "_type": self._type,
                    "modaic_id": self.id,
                    "_id": self._gqlalchemy_id,
                    "_start_node_id": self.start_node_gql_id,
                    "_end_node_id": self.end_node_gql_id,
                    **self.model_dump(
                        exclude={"id", "start_node", "end_node", "_type"}
                    ),
                }
            )

    @classmethod
    def from_gqlalchemy(cls, gqlalchemy_rel: "gqlalchemy.Relationship") -> "Relation":
        """
        Convert a GQLAlchemy `Relationship` into a `Relation` instance. If `cls` is the `Relation` class itself, it will try to return an instance of a subclass of `Relation` that matches the type of the GQLAlchemy Relationship. If none are found it will fallback to an instance of `Relation` since the `Relation` class allows definiing inline.
        If `cls` is instead a subclass of `Relation`, it will return an instance of that subclass and fail if the properties do not align.
        Args:
            gqlalchemy_obj: The GQLAlchemy Relationship to convert.

        Raises:
            ValueError: If the GQLAlchemy Relationship does not have the required fields.
            AssertionError: If the GQLAlchemy Relationship does not have the required type.

        Returns:
            The converted Relation or Relation subclass instance.
        """
        if cls is not Relation:
            assert cls._type == gqlalchemy_rel._type, (
                f"Cannot convert GQLAlchemy Relationship {gqlalchemy_rel} to {cls.__name__} because it does not have {cls.__name__}'s type: '{cls._type}'"
            )
            try:
                kwargs = {**gqlalchemy_rel._properties}
                kwargs["id"] = kwargs.pop("modaic_id")
                kwargs["start_node"] = gqlalchemy_rel._start_node_id
                kwargs["end_node"] = gqlalchemy_rel._end_node_id
                new_relation = cls(**kwargs)
                new_relation._gqlalchemy_id = gqlalchemy_rel._id
                return new_relation
            except ValidationError as e:
                raise ValueError(
                    f"Failed to convert GQLAlchemy Relationship {gqlalchemy_rel} to {cls.__name__} because it does not have the required fields.\nError: {e}"
                )

        # If cls is Relation, we need to find the subclass of Relation that matches the type of the GQLAlchemy Relationship.
        # CAVEAT: Relation is a subclass of ContextSchema, so we can just use the same ContextSchema._type_registry.
        if subclass := ContextSchema._type_registry.get(gqlalchemy_rel._type):
            assert issubclass(subclass, Relation), (
                f"Found Relation subclass with matching type, but cannot convert GQLAlchemy Relationship {gqlalchemy_rel} to {subclass.__name__} because it is not a subclass of Relation"
            )
            return subclass.from_gqlalchemy(gqlalchemy_rel)
        # If no subclass is found, we can just create a new Relation object with the properties of the GQLAlchemy Relationship.
        else:
            kwargs = {**gqlalchemy_rel._properties}
            kwargs["id"] = kwargs.pop("modaic_id")
            kwargs["start_node"] = gqlalchemy_rel._start_node_id
            kwargs["end_node"] = gqlalchemy_rel._end_node_id
            kwargs["_type"] = gqlalchemy_rel._type
            new_relation = cls(**kwargs)
            new_relation._gqlalchemy_id = gqlalchemy_rel._id
            return new_relation

    def save(self, db: "GraphDatabase"):
        """
        Save the Relation object to the GraphDatabase.
        """

        try:
            from modaic.databases.graph_database import GraphDatabase
        except ImportError:
            raise ImportError(
                "GQLAlchemy is not installed. Please install the graph extension for modaic with `uv add modaic[graph]`"
            )

        assert isinstance(db, GraphDatabase), (
            "Expected db to be a modaic.databases.GraphDatabase instance. "
            f"Got {type(db)} instead."
        )
        result = db.save_relationship(self)
        for k in self.model_dump(exclude={"id", "start_node", "end_node"}):
            setattr(self, k, getattr(result, k))
        self._gqlalchemy_id = result._id

    def load(self, db: "GraphDatabase"):
        """
        Loads a relationship from GraphDatabase.
        If the relationship._id is not None it fetches the relationship from GraphDatabase with that
        internal id.
        If the relationship has unique fields it fetches the relationship from GraphDatabase with
        those unique fields set.
        Otherwise it tries to find any relationship in GraphDatabase that has all properties
        set to exactly the same values.
        If no relationship is found or no properties are set it raises a GQLAlchemyError.
        """
        raise NotImplementedError("Not implemented")


def cast_type_if_base_model(field_type):
    """
    If field_type is a typing construct, reconstruct it from origin/args.
    If it's a Pydantic BaseModel subclass, map it to `dict`.
    Otherwise return the type itself.
    """
    origin = get_origin(field_type)

    # Non-typing constructs
    if origin is None:
        # Only call issubclass on real classes
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return dict
        return field_type

    args = get_args(field_type)

    # Annotated[T, m1, m2, ...]
    if origin is Annotated:
        base, *meta = args
        # Annotated allows multiple args; pass a tuple to __class_getitem__
        return Annotated.__class_getitem__((cast_type_if_base_model(base), *meta))

    # Unions: typing.Union[...] or PEP 604 (A | B)
    if origin in (Union, UnionType):
        return Union[tuple(cast_type_if_base_model(a) for a in args)]

    # Literal / Final / ClassVar accept tuple args via typing protocol
    if origin in (Literal, Final, ClassVar):
        return origin.__getitem__([cast_type_if_base_model(a) for a in args])

    # Builtin generics (PEP 585): list[T], dict[K, V], set[T], tuple[...]
    if origin in (list, set, frozenset):
        (T,) = args
        return origin[cast_type_if_base_model(T)]
    if origin is dict:
        K, V = args
        return dict[cast_type_if_base_model(K), cast_type_if_base_model(V)]
    if origin is tuple:
        # tuple[int, ...] vs tuple[int, str]
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple[cast_type_if_base_model(args[0]), ...]
        return tuple[
            tuple([cast_type_if_base_model(a) for a in args])
        ]  # tuple[(A, B, C)]

    # ABC generics (e.g., Mapping, Sequence, Iterable, etc.) usually accept tuple args
    try:
        return origin.__class_getitem__([cast_type_if_base_model(a) for a in args])
    except Exception:
        # Last resort: try simple unpack for 1–2 arity generics
        if len(args) == 1:
            return origin[cast_type_if_base_model(args[0])]
        elif len(args) == 2:
            return origin[
                cast_type_if_base_model(args[0]), cast_type_if_base_model(args[1])
            ]
        raise


def get_annotations(cls: Type, exclude: Optional[List[str]] = None):
    if exclude is None:
        exclude = []
    if not issubclass(cls, ContextSchema):
        return {}
    elif cls is ContextSchema:
        res = {
            k: cast_type_if_base_model(v)
            for k, v in cls.__annotations__.items()
            if k not in exclude
        }
        return res
    else:
        annotations = {}
        for base in cls.__bases__:
            annotations.update(get_annotations(base, exclude))
        annotations.update(
            {
                k: cast_type_if_base_model(v)
                for k, v in cls.__annotations__.items()
                if k not in exclude
            }
        )
        return annotations


def cast_if_base_model(field_default):
    if isinstance(field_default, BaseModel):
        return field_default.model_dump()
    return field_default


def get_defaults(cls: Type[ContextSchema], exclude: Optional[List[str]] = None):
    if exclude is None:
        exclude = []
    defaults: dict[str, Any] = {}
    for name, v2_field in cls.model_fields.items():
        if name in exclude or v2_field.is_required():
            continue
        kwargs = {}
        if extra_kwargs := getattr(v2_field, "json_schema_extra", None):
            kwargs.update(extra_kwargs)

        factory = v2_field.default_factory
        if factory is not None:
            kwargs["default_factory"] = lambda f=factory: cast_if_base_model(f())
        else:
            kwargs["default"] = cast_if_base_model(v2_field.default)

        v1_field = V1Field(**kwargs)
        defaults[name] = v1_field

    return defaults


def get_ad_hoc_annotations(rel: Relation):
    """
    Gets "adhoc" annotations for a Relation object. Specifically, for when Relations are created inline.
    (i.e. when you do `Relation(x="test", _type="TEST_REL")`).
    This is for those fields that were decleared inline.
    Args:
        rel: The Relation object to get the adhoc annotations for.

    Returns:
        A dictionary of the adhoc annotations.
    """
    annotations = {}
    for name, val in rel.model_dump(
        exclude=GQLALCHEMY_EXCLUDED_FIELDS + ["start_node", "end_node"]
    ).items():
        if val is None:
            annotations[name] = Any
        elif isinstance(val, BaseModel):
            annotations[name] = dict
        else:
            annotations[name] = type(val)
    return annotations
