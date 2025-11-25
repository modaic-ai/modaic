import copy
import typing as t
import uuid
from functools import lru_cache, wraps
from types import UnionType
from typing import Any, Literal

from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import ModelPrivateAttr
from pydantic.main import IncEx
from pydantic.v1 import Field as V1Field
from pydantic_core import SchemaSerializer

from ..query_language import Prop
from ..storage.file_store import FileStore
from ..types import Field, Schema

if t.TYPE_CHECKING:
    import gqlalchemy

    from modaic.databases.graph_database import GraphDatabase
    from modaic.storage.file_store import FileStore


GQLALCHEMY_EXCLUDED_FIELDS = [
    "id",
    "_gqlalchemy_id",
    "_type_registry",
    "_labels",
    "_gqlalchemy_class_registry",
    "_type",
]


class ModaicHydrationError(Exception):
    """Error raised when a function tries to use a Context param that is not hydrated."""

    pass


class ModelHydratedAttr(ModelPrivateAttr):
    def __init__(self):
        super().__init__(default=None, default_factory=None)


def HydratedAttr():  # noqa: N802, ANN201
    """
    Created a hydrated field. Hydrated fields are fields that are None by default and are hydrated by Context.hydrate()
    """
    return ModelHydratedAttr()


def requires_hydration(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
    """
    Decorator that ensures all hydrated attributes are set before calling the function.

    Args:
        func: The method being wrapped.

    Returns:
        The wrapped method that raises if any hydrated attribute is None.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]

        for attr in self.__class__.__hydrated_attributes__:
            if getattr(self, attr) is None:
                raise ModaicHydrationError(
                    f"Attribute {attr} is not hydrated. Please call `self.hydrate()` to hydrate the attribute."
                )
        return func(*args, **kwargs)

    return wrapper


def _get_unhidden_serializer(cls: type[BaseModel]) -> SchemaSerializer:
    """
    Creates a new serializer from cls.__pydantic_core_schema__ with the hidden fields included.
    This is nescesarry to recursively dump hidden Context objects with hidden fields inside of other context objects.
    """
    core = copy.deepcopy(cls.__pydantic_core_schema__)

    def walk(node: dict | list):
        if isinstance(node, dict):
            if (
                node.get("type") == "model"
                and node.get("serialization", {}).get("function", None) is Context.hidden_serializer
            ):
                del node["serialization"]
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(core)
    return SchemaSerializer(core)


class ContextMeta(ModelMetaclass):
    def __getattr__(cls, name: str) -> t.Any:  # noqa: N805
        """
        Enablees the creation of Prop classes via ContextClass.property_name. Does this in a safe way that doesn't conflict with pydantic's own metaclass.
        """
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


class Context(BaseModel, metaclass=ContextMeta):
    """
    Base class for all Context objects.

    Attributes:
        id: The id of the serialized context.
        source: The source of the context object.
        metadata: The metadata of the context object.

    Example:
        In this example, `CaptionedImage` stores the caption and the caption embedding the image path and the image itself. Since we can't serialize the image, we use the `HydratedAttr` decorator to mark the `_image` field as requiring hydration.
        ```python
        from modaic.context import Context
        from modaic.types import String, Vector, Float16Vector

        class CaptionedImage(Context):
            caption: String[100]
            caption_embedding: Float16Vector[384]
            _image: PIL.Image.Image = HydratedAttr()

            def hydrate(self, file_store: FileStore):
                image_path = file_store.get_files(self.id)["image"]
                self._image = PIL.Image.open(image_path)

        ```
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), hidden=True)
    parent: t.Optional[str] = Field(default=None, hidden=True)
    metadata: dict = Field(default_factory=dict, hidden=True)

    _gqlalchemy_id: t.Optional[int] = PrivateAttr(default=None)
    _chunks: t.List["Context"] = PrivateAttr(default_factory=list)

    # CAVEAT: All Context subclasses share the same instance of _type_registry. This is intentional.
    _type_registry: t.ClassVar[t.Dict[str, t.Type["Context"]]] = {}
    _labels: t.ClassVar[frozenset[str]] = frozenset()
    _gqlalchemy_class_registry: t.ClassVar[t.Dict[str, t.Type["gqlalchemy.models.GraphObject"]]] = {}

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        """Allow class-header keywords without raising TypeError.

        Args:
            **kwargs: Arbitrary keywords from subclass declarations (e.g., type="Label").
        """
        super().__init_subclass__()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        if "type" in kwargs:
            cls._type = kwargs["type"]
        else:
            cls._type = cls.__name__

        assert cls._type != "Node" and cls._type != "Relationship", (
            f"Class {cls.__name__} cannot use name 'Node' or 'Relationship' as type. Please use a different name. You can use a custom type by using the 'type' keyword. Example: `class {cls.__name__}(Context, type=<custom_type_name>)`"
        )

        # TODO: revisit this. Should we allow multiple parents?
        # Get parent class labels
        parent_labels = [b._labels for b in cls.__bases__ if issubclass(b, Context)]
        assert len(parent_labels) == 1, (
            f"Context class {cls.__name__} cannot have multiple Context classes as parents. Should it? Submit an issue to tell us about your use case. https://github.com/modaic-ai/modaic/issues"
        )
        cls._labels = frozenset({cls._type}) | parent_labels[0]
        assert cls._type not in cls._type_registry, (
            f"Cannot have multiple Context/Relation classes with type = '{cls._type}'"
        )
        cls._type_registry[cls._type] = cls

        cls.__hydrated_attributes__ = set()
        for name in (private_attrs := cls.__private_attributes__):
            if isinstance(private_attrs[name], ModelHydratedAttr):
                cls.__hydrated_attributes__.add(name)

        cls.__modaic_serializer__ = _get_unhidden_serializer(cls)

    def __str__(self) -> str:
        """
        Returns a string representation of the Context instance, including all field values.

        Returns:
            str: String representation with all field values.
        """
        values = self.model_dump(mode="json", include_hidden=True)
        return f"{self.__class__._type}({values})"

    def __repr__(self):
        return self.__str__()

    def to_gqlalchemy(self, db: "GraphDatabase") -> "gqlalchemy.Node":
        """
        Convert the Context object to a GQLAlchemy object.
        !!! warning
            This method is not thread safe. We are actively working on a solution to make it thread safe.
        """
        try:
            import gqlalchemy

            from modaic.databases.graph_database import GraphDatabase
        except ImportError:
            raise ImportError(
                "GQLAlchemy is not installed. Please install the graph extension for modaic with `uv add modaic[graph]`"
            ) from None
        assert isinstance(db, GraphDatabase), (
            f"Expected db to be a modaic.databases.GraphDatabase instance. Got {type(db)} instead."
        )
        cls = self.__class__

        # Dynamically create a GQLAlchemy Node class for the Context if it doesn't exist
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
                **self.model_dump(exclude={"id"}, include_hidden=True),
            )
        else:
            return gqlalchemy_class(
                _labels=set(self._labels),
                modaic_id=self.id,
                _id=self._gqlalchemy_id,
                **self.model_dump(exclude={"id"}, include_hidden=True),
            )

    @classmethod
    def from_gqlalchemy(cls, gqlalchemy_node: "gqlalchemy.Node") -> "Context":
        """
        Convert a GQLAlchemy Node into a `Context` instance. If cls is the Context class itself, it will return the best subclass of Context that matches the labels of the GQLAlchemy Node.
        Args:
            gqlalchemy_node: The GQLAlchemy Node to convert.

        Returns:
            The converted Context or Context subclass instance.

        """
        if cls is not Context:
            if cls._type not in gqlalchemy_node._labels:
                raise ValueError(
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
                ) from e

        # If cls is Context, we need to find the best subclass of Context that matches the labels of the GQLAlchemy Node.
        best_subclass = Context._best_subclass(frozenset(gqlalchemy_node._labels))
        return best_subclass.from_gqlalchemy(gqlalchemy_node)

    def save(self, db: "GraphDatabase"):
        """
        Save the Context object to the graph database.

        !!! warning
            This method is not thread safe. We are actively working on a solution to make it thread safe.
        """
        try:
            from modaic.databases.graph_database import GraphDatabase
        except ImportError:
            raise ImportError(
                "GQLAlchemy is not installed. Please install the graph extension for modaic with `uv add modaic[graph]`"
            ) from None

        assert isinstance(db, GraphDatabase), (
            f"Expected db to be a modaic.databases.GraphDatabase instance. Got {type(db)} instead."
        )

        result = db.save_node(self)

        for k in self.model_dump(exclude={"id"}, include_hidden=True):
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

    @staticmethod
    @lru_cache
    def _best_subclass(labels: t.FrozenSet[str]) -> t.Type["Context"]:
        best_subclass = None
        for label in labels:
            if current_subclass := Context._type_registry.get(label):
                # check if the current subclass has more parents than the best subclass
                if best_subclass is None or len(current_subclass.__mro__) > len(best_subclass.__mro__):
                    best_subclass = current_subclass

        if best_subclass is None:
            raise ValueError(f"Cannot find a matching Context class for labels: {labels}")
        return best_subclass

    # TODO: Make iterable-friendly
    def chunk_with(
        self,
        chunk_fn: t.Callable[["Context"], t.Iterable["Context"]],
        kwargs: t.Optional[t.Dict] = None,
    ) -> None:
        """
        Chunks the context object into a list of context objects.
        """
        if kwargs is None:
            kwargs = {}
        self._chunks = list(chunk_fn(self, **kwargs))
        for chunk in self._chunks:
            chunk.parent = self.id

    def apply_to_chunks(self, apply_fn: t.Callable[["Context"], None], **kwargs):
        """
        Applies apply_fn to each chunk in chunks.

        Args:
            apply_fn: The function to apply to each chunk. Function should take in a Context object and mutate it.
            **kwargs: Additional keyword arguments to pass to apply_fn.
        """
        for chunk in self.chunks:
            apply_fn(chunk, **kwargs)

    @property
    def chunks(self) -> t.List["Context"]:
        """
        Returns the chunks of the context object.
        """
        return self._chunks

    @property
    def is_hydrated(self) -> bool:
        """
        Returns True if the context object is hydrated.
        """
        if not hasattr(self, "__hydrated_attributes__"):
            return True
        return all(getattr(self, attr) is not None for attr in self.__hydrated_attributes__)

    @classmethod
    def schema(cls) -> Schema:
        if hasattr(cls, "_schema"):
            return cls._schema
        cls._schema = Schema.from_json_schema(cls.model_json_schema())
        return cls._schema

    @model_serializer(mode="wrap")
    def hidden_serializer(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> dict[str, Any]:
        dump = handler(self)
        if info.context is None or not info.context.get("include_hidden"):
            for name, field in self.__class__.model_fields.items():
                if (extra := getattr(field, "json_schema_extra", None)) and extra.get("hidden"):
                    dump.pop(name, None)
        return dump

    def model_dump(
        self,
        *,
        mode: str | Literal["json", "python"] = "python",
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        fallback: t.Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
        include_hidden: bool = False,
    ) -> Any:
        """
        Override of pydantics BaseModel.model_dump to allow for showing hidden fields

        Args:
            include_hidden: Whether to show hidden fields.

        Returns:
            The dictionary representation of the model.
        """
        if include_hidden:
            return self.__modaic_serializer__.to_python(
                self,
                mode=mode,
                include=include,
                exclude=exclude,
                context=context,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                round_trip=round_trip,
                warnings=warnings,
                fallback=fallback,
                serialize_as_any=False,
            )

        else:
            return super().model_dump(
                mode=mode,
                include=include,
                exclude=exclude,
                context=context,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                round_trip=round_trip,
                warnings=warnings,
                fallback=fallback,
                serialize_as_any=serialize_as_any,
            )

    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        fallback: t.Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
        include_hidden: bool = False,
    ) -> bytes | str:
        """
        Override of pydantic's BaseModel.model_dump_json to allow for showing hidden fields
        """
        if include_hidden:
            return self.__modaic_serializer__.to_json(
                self,
                indent=indent,
                include=include,
                exclude=exclude,
                context=context,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                round_trip=round_trip,
                warnings=warnings,
                fallback=fallback,
                serialize_as_any=True,
            )
        else:
            return super().model_dump_json(
                indent=indent,
                include=include,
                exclude=exclude,
                context=context,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                round_trip=round_trip,
                warnings=warnings,
                fallback=fallback,
                serialize_as_any=serialize_as_any,
            )


class RelationMeta(ContextMeta):
    def __new__(cls, name, bases, dct):  # noqa: ANN002, ANN001, ANN003
        # Make Relation class allow extra fields but subclasses default to ignore (pydantic default)
        # BUG: Doesn't allow users to define their own "extra" behavior
        if name == "Relation":
            dct["model_config"] = ConfigDict(extra="allow")
        elif "model_config" not in dct:
            dct["model_config"] = ConfigDict(extra="ignore")
        elif dct["model_config"].get("extra", None) is None:
            dct["model_config"]["extra"] = "ignore"

        return super().__new__(cls, name, bases, dct)


class Relation(Context, metaclass=RelationMeta):
    """
    Base class for all Relation objects.
    """

    _start_node: t.Optional[Context] = PrivateAttr(default=None)
    _end_node: t.Optional[Context] = PrivateAttr(default=None)

    start_node: t.Optional[int] = None
    end_node: t.Optional[int] = None

    @t.overload
    def __init__(self, start_node: Context | int, end_node: Context | int, **data: Any) -> "Relation": ...

    @model_validator(mode="wrap")
    @classmethod
    def truncate(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> "Relation":
        """
        Truncates the start_node and end_node to their gqlalchemy ids.
        """
        ids = {}
        objs = {}
        for name in ["start_node", "end_node"]:
            node = data[name]
            if isinstance(node, Context):
                ids[name] = node._gqlalchemy_id
                objs[name] = node
            else:
                ids[name] = node
                objs[name] = None
        data["start_node"] = ids["start_node"]
        data["end_node"] = ids["end_node"]
        self = handler(data)
        self._start_node = objs["start_node"]
        self._end_node = objs["end_node"]
        return self

    def get_start_node_obj(self, db: "GraphDatabase") -> Context:
        """
        Get the start node object of the relation as a Context object.
        Args:
            db: The GraphDatabase instance to use to fetch the start node.

        Returns:
            The start node object as a Context object.
        """
        if self._start_node:
            return self._start_node
        else:
            return Context.from_gqlalchemy(
                next(db.execute_and_fetch(f"MATCH (n) WHERE id(n) = {self.start_node} RETURN n"))
            )

    def get_end_node_obj(self, db: "GraphDatabase") -> Context:
        """
        Get the end node object of the relation as a Context object.
        Args:
            db: The GraphDatabase instance to use to fetch the end node.

        Returns:
            The end node object as a Context object.
        """
        if self._end_node:
            return self.end_node
        else:
            return Context.from_gqlalchemy(
                next(db.execute_and_fetch(f"MATCH (n) WHERE id(n) = {self.end_node} RETURN n"))
            )

    @field_validator("start_node", "end_node")
    @classmethod
    def check_node(cls, v: Any) -> Context | int:
        assert isinstance(v, (Context, int)), f"start_node/end_node must be a Context or int, got {type(v)}: {v}"
        assert not isinstance(v, Relation), f"start_node/end_node cannot be a Relation object: {v}"
        return v

    @model_validator(mode="after")
    def post_init(self) -> "Relation":
        # Sets type for inline declaration of Relation objects
        if type(self) is Relation:
            assert "_type" in self.model_dump(), "Inline declaration of Relation objects must specify the '_type' field"
            self._type = self.model_dump()["_type"]
        return self

    # other >> self
    def __rrshift__(self, other: Context | int):
        # left_node >> self >> right_node
        self.start_node = other
        return self

    # self >> other
    def __rshift__(self, other: Context | int):
        # left_node >> self >> right_node
        self.end_node = other
        return self

    # other << self
    def __rlshift__(self, other: Context | int):
        # left_node << self << right_node
        self.end_node = other
        return self

    # self << other
    def __lshift__(self, other: Context | int):
        # left_node << self << right_node
        self.start_node = other
        return self

    def __str__(self):
        """
        Returns a string representation of the Relation object, including all fields and their values.

        Returns:
            str: String representation of the Relation object with all fields and their values.
        """
        fields_repr = ", ".join(f"{k}={repr(v)}" for k, v in self.model_dump(include_hidden=True).items())
        return f"{self.__class__._type}({fields_repr})"

    def __repr__(self):
        return self.__str__()

    def to_gqlalchemy(self, db: "GraphDatabase") -> "gqlalchemy.Relationship":
        """
        Convert the Context object to a GQLAlchemy object.

        <Warning>Saves the start_node and end_node to the database if they are not already saved.</Warning>

        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        Args:
            db: The GraphDatabase instance to use to save the start_node and end_node if they are not already saved.

        Returns:
            The GQLAlchemy Relationship object.

        Raises:
            AssertionError: If db is not a modaic.databases.GraphDatabase instance.
            ImportError: If GQLAlchemy is not installed.

        """
        try:
            import gqlalchemy

            from modaic.databases.graph_database import GraphDatabase
        except ImportError:
            raise ImportError(
                "GQLAlchemy is not installed. Please install the graph extension for modaic with `uv add modaic[graph]`"
            ) from None

        assert isinstance(db, GraphDatabase), (
            f"Expected db to be a modaic.databases.GraphDatabase instance. Got {type(db)} instead."
        )

        cls = self.__class__

        # Dynamically create a GQLAlchemy Node class for the Context if it doesn't exist
        if self._type not in cls._gqlalchemy_class_registry:
            ad_hoc_annotations = get_ad_hoc_annotations(self) if cls is Relation else {}
            field_annotations = get_annotations(
                cls,
                exclude=GQLALCHEMY_EXCLUDED_FIELDS + ["start_node", "end_node"],
            )
            field_defaults = get_defaults(
                cls,
                exclude=GQLALCHEMY_EXCLUDED_FIELDS + ["start_node", "end_node"],
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
                        exclude={"id", "start_node", "end_node", "_type"},
                        include_hidden=True,
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
                        exclude={"id", "start_node", "end_node", "_type"},
                        include_hidden=True,
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
                ) from e

        # If cls is Relation, we need to find the subclass of Relation that matches the type of the GQLAlchemy Relationship.
        # CAVEAT: Relation is a subclass of Context, so we can just use the same Context._type_registry.
        if subclass := Context._type_registry.get(gqlalchemy_rel._type):
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

        !!! warning
            This method is not thread safe. We are actively working on a solution to make it thread safe.
        """

        try:
            from modaic.databases.graph_database import GraphDatabase
        except ImportError:
            raise ImportError(
                "GQLAlchemy is not installed. Please install the graph extension for modaic with `uv add modaic[graph]`"
            ) from None

        assert isinstance(db, GraphDatabase), (
            f"Expected db to be a modaic.databases.GraphDatabase instance. Got {type(db)} instead."
        )
        result = db.save_relationship(self)
        for k in self.model_dump(exclude={"id", "start_node", "end_node"}, include_hidden=True):
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


def _cast_type_if_base_model(field_type: t.Type) -> t.Type:
    """
    If field_type is a typing construct, reconstruct it from origin/args.
    If it's a Pydantic BaseModel subclass, map it to `dict`.
    Otherwise return the type itself.
    """
    origin = t.get_origin(field_type)

    # Non-typing constructs
    if origin is None:
        # Only call issubclass on real classes
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return dict
        return field_type

    args = t.get_args(field_type)

    # Annotated[T, m1, m2, ...] # noqa: ERA001
    if origin is t.Annotated:
        base, *meta = args
        # Annotated allows multiple args; pass a tuple to __class_getitem__
        return t.Annotated.__class_getitem__((_cast_type_if_base_model(base), *meta))

    # Unions: typing.Union[...] or PEP 604 (A | B)
    if origin in (t.Union, UnionType):
        return t.Union[tuple(_cast_type_if_base_model(a) for a in args)]

    # Literal / Final / ClassVar accept tuple args via typing protocol
    if origin in (t.Literal, t.Final, t.ClassVar):
        return origin.__getitem__([_cast_type_if_base_model(a) for a in args])

    # Builtin generics (PEP 585): list[T], dict[K, V], set[T], tuple[...]
    if origin in (list, set, frozenset):
        (inner_type,) = args
        return origin[_cast_type_if_base_model(inner_type)]
    if origin is dict:
        key_type, value_type = args
        return dict[_cast_type_if_base_model(key_type), _cast_type_if_base_model(value_type)]
    if origin is tuple:
        # tuple[int, ...] vs tuple[int, str]
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple[_cast_type_if_base_model(args[0]), ...]
        return tuple[tuple([_cast_type_if_base_model(a) for a in args])]  # tuple[(A, B, C)]

    # ABC generics (e.g., Mapping, Sequence, Iterable, etc.) usually accept tuple args
    try:
        return origin.__class_getitem__([_cast_type_if_base_model(a) for a in args])
    except Exception:
        # Last resort: try simple unpack for 1–2 arity generics
        if len(args) == 1:
            return origin[_cast_type_if_base_model(args[0])]
        elif len(args) == 2:
            return origin[
                _cast_type_if_base_model(args[0]),
                _cast_type_if_base_model(args[1]),
            ]
        raise


def get_annotations(cls: t.Type, exclude: t.Optional[t.List[str]] = None) -> t.Dict[str, t.Type]:
    if exclude is None:
        exclude = []
    if not issubclass(cls, Context):
        return {}
    elif cls is Context:
        res = {k: _cast_type_if_base_model(v) for k, v in cls.__annotations__.items() if k not in exclude}
        return res
    else:
        annotations = {}
        for base in cls.__bases__:
            annotations.update(get_annotations(base, exclude))
        annotations.update({k: _cast_type_if_base_model(v) for k, v in cls.__annotations__.items() if k not in exclude})
        return annotations


def _cast_if_base_model(field_default: t.Any) -> t.Any:
    if isinstance(field_default, BaseModel):
        return field_default.model_dump()
    return field_default


def get_defaults(cls: t.Type[Context], exclude: t.Optional[t.List[str]] = None) -> t.Dict[str, t.Any]:
    if exclude is None:
        exclude = []
    defaults: t.Dict[str, t.Any] = {}
    for name, v2_field in cls.model_fields.items():
        if name in exclude or v2_field.is_required():
            continue
        kwargs = {}
        if extra_kwargs := getattr(v2_field, "json_schema_extra", None):
            kwargs.update(extra_kwargs)

        factory = v2_field.default_factory
        if factory is not None:
            kwargs["default_factory"] = lambda f=factory: _cast_if_base_model(f())
        else:
            kwargs["default"] = _cast_if_base_model(v2_field.default)

        v1_field = V1Field(**kwargs)
        defaults[name] = v1_field

    return defaults


def get_ad_hoc_annotations(rel: Relation) -> t.Dict[str, t.Type]:
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
        exclude=GQLALCHEMY_EXCLUDED_FIELDS + ["start_node", "end_node"],
        include_hidden=True,
    ).items():
        if val is None:
            annotations[name] = t.Any
        elif isinstance(val, BaseModel):
            annotations[name] = dict
        else:
            annotations[name] = type(val)
    return annotations


@t.runtime_checkable
class Hydratable(t.Protocol):
    def hydrate(self, file_store: FileStore) -> None:
        pass

    @classmethod
    def from_file(cls, file: str, file_store: FileStore, type: str, params: dict = None) -> "Hydratable":
        """
        Load a Hydratable instance from a file.

        Args:
            file: The file to load.
            file_store: The file store to use.
            type: The type of file to expect.
            params: Extra parameters to pass to the constructor.
        """
        pass


if t.TYPE_CHECKING:
    # @runtime_checkable
    class HydratableContext(Hydratable, Context):
        pass


def is_hydratable(obj: t.Any) -> bool:
    return isinstance(obj, Hydratable) and isinstance(obj, Context)


@t.runtime_checkable
class Embeddable(t.Protocol):
    """
    A protocol for objects that can be embedded. These objects define the embedme function which can either return a string or an image.
    The embedme function can either take no args, or take an index name as an argument, which will be used to select the index to embed for.
    """

    @t.overload
    def embedme(self) -> str | Image.Image: ...

    @t.overload
    def embedme(self, index: t.Optional[str] = None) -> str | Image.Image: ...


@t.runtime_checkable
class MultiEmbeddable(t.Protocol):
    """
    A protocol for objects that can be embedded and have multiple embeddings. These objects define the embedme function which can either return a string or an image.
    The embedme function can either take no args, or take an index name as an argument, which will be used to select the index to embed for.
    """

    @t.overload
    def embedme(self, index: t.Optional[str] = None) -> str | Image.Image: ...


def is_embeddable(obj: t.Any) -> bool:
    return isinstance(obj, Embeddable) and isinstance(obj, Context)


def is_multi_embeddable(obj: t.Any) -> bool:
    return isinstance(obj, MultiEmbeddable) and isinstance(obj, Context)


if t.TYPE_CHECKING:
    # @runtime_checkable
    class EmbeddableContext(t.Protocol, Context):
        pass


def _update_exclude(exclude: IncEx, hidden: t.Set[str]):
    if isinstance(exclude, set):
        return exclude.update(hidden)
    else:  # NOTE: if not a set, it's a dict
        return exclude.update({k: True for k in hidden})


def _dump_hidden_recursive(obj: t.Any):
    if isinstance(obj, BaseModel):
        return obj.model_dump(include_hidden=True)
    elif isinstance(obj, Context):
        return obj.model_dump(include_hidden=True)
    elif isinstance(obj, list):
        return [_dump_hidden_recursive(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dump_hidden_recursive(v) for k, v in obj.items()}
    else:
        return obj
