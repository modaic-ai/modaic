from typing import (
    List,
    Type,
    Dict,
    get_origin,
    get_args,
    Optional,
    Union,
    Literal,
    Protocol,
)
from types import NoneType
from typing_extensions import Annotated
from pydantic import Field, BaseModel
from pydantic.fields import FieldInfo
from collections.abc import Mapping
from annotated_types import MaxLen
from types import UnionType
from dataclasses import dataclass, asdict
import copy

# CAVEAT:In this module we use a roundabout way of defining types. With annotated and then returning the original custom type inpydantic_model_to_schema.
# We do this instead of just making a new pydantic type because it is more stable and easily works with pydantic's system.
# pydantic's website says making new pydantic types is not recommended. https://docs.pydantic.dev/latest/concepts/types/#summary

int8 = Annotated[
    int, Field(ge=-128, le=127, json_schema_extra={"original_class": "int8"})
]
int16 = Annotated[
    int, Field(ge=-32768, le=32767, json_schema_extra={"original_class": "int16"})
]
int32 = Annotated[
    int, Field(ge=-(2**31), le=2**31 - 1, json_schema_extra={"original_class": "int32"})
]
int64 = Annotated[
    int, Field(ge=-(2**63), le=2**63 - 1, json_schema_extra={"original_class": "int64"})
]
float32 = Annotated[
    float,
    Field(ge=-3.40e38, le=3.40e38, json_schema_extra={"original_class": "float32"}),
]
float64 = Annotated[
    float,
    Field(ge=-1.87e308, le=1.87e308, json_schema_extra={"original_class": "float64"}),
]
double = float64


# class VectorMeta(type):
#     def __new__(cls, name, bases, attrs):
#         if "dtype" not in attrs:
#             raise TypeError(f"{cls.__name__} requires a dtype")
#         return super().__new__(cls, name, bases, attrs)

#     def __getitem__(cls, dim):
#         if not isinstance(dim, int):
#             raise TypeError(
#                 f"{cls.__name__} requires exactly 1 parameters: {cls.__name__}[dim]"
#             )

#         if not isinstance(dim, int) or dim <= 0:
#             raise TypeError("Vector size must be a positive integer")

#         return Annotated[
#             List[cls.dtype],
#             Field(min_length=dim, max_length=dim, original_class=cls.__name__, dim=dim),
#         ]


# class Vector(List, metaclass=VectorMeta):
#     """
#     float vector field type for `Context` of the given dimension. Must be created with Vector[dim]

#     Args:
#         dim (int): Required. The dimension of the vector.

#     Example:
#         The `Context` class for a `CaptionedImage` Context type that stores both a primary embedding using the image and a secondary embedding using the caption.
#         ```python
#         from modaic.types import Vector
#         from modaic.context import Context

#         class CaptionImage(Context):
#             caption: String[100]
#             caption_embedding: Vector[384]
#         ```
#     """

#     dtype: Type[Any] = float


# class Float16Vector(Vector):
#     """
#     float16 vector field type for `Context` of the given dimension. Must be created with Float16Vector[dim]

#     Args:
#         dim (int): Required. The dimension of the vector.

#     Example:
#         ```python
#         from modaic.types import Float16Vector
#         from modaic.context import Context

#         # Case where we want to store a secondary embedding for the caption of an image.
#         class CaptionImage(Context):
#             caption: String[100]
#             caption_embedding: Float16Vector[384]
#         ```
#     """

#     dtype = Annotated[float, Field(ge=-65504, le=65504)]  # Float16 range


# class Float32Vector(Vector):
#     """
#     float32 vector field type for `Context` of the given dimension. Must be created with Float32Vector[dim]

#     Args:
#         dim (int): Required. The dimension of the vector.

#     Example:
#         The `Context` class for a `CaptionedImage` Context type that stores both a primary embedding using the image and a secondary embedding using the caption.
#         ```python
#         from modaic.types import Float32Vector
#         from modaic.context import Context

#         class CaptionImage(Context):
#             caption: String[100]
#             caption_embedding: Float32Vector[384]
#         ```
#     """

#     dtype = float32


# class Float64Vector(Vector):
#     """
#     float64 vector field type for `Context` of the given dimension. Must be created with Float64Vector[dim]

#     Args:
#         dim (int): Required. The dimension of the vector.

#     Example:
#         The `Context` class for a `CaptionedImage` Context type that stores both a primary embedding using the image and a secondary embedding using the caption.
#         ```python
#         from modaic.types import Float64Vector
#         from modaic.context import Context

#         class CaptionImage(Context):
#             caption: String[100]
#             caption_embedding: Float64Vector[384]
#         ```
#     """

#     dtype = float64


# class BFloat16Vector(Vector):
#     """
#     bfloat16 vector field type for `Context` of the given dimension. Must be created with BFloat16Vector[dim]

#     Args:
#         dim (int): Required. The dimension of the vector.

#     Example:
#         The `Context` class for a `CaptionedImage` Context type that stores both a primary embedding using the image and a secondary embedding using the caption.
#         ```python
#         from modaic.types import BFloat16Vector
#         from modaic.context import Context

#         class CaptionImage(Context):
#             caption: String[100]
#             caption_embedding: BFloat16Vector[384]
#         ```
#     """

#     dtype = Annotated[float, Field(ge=-3.4e38, le=3.40e38)]  # BFloat16 range


# class BinaryVector(Vector):
#     """
#     binary vector field type for `Context` of the given dimension. Must be created with BinaryVector[dim]

#     Args:
#         dim (int): Required. The dimension of the vector.

#     Example:
#         The `Context` class for a `SenateBill` Context type that uses a binary vector to store the vote distribution.
#         ```python
#         from modaic.types import BinaryVector
#         from modaic.context import Context

#         class SenateBill(Context):
#             bill_id: int
#             bill_title: String[10]
#             bill_description: String
#             vote_distribution: BinaryVector[100]
#         ```
#     """

#     dtype = bool


# class SparseVectorMeta(type):
#     def __new__(cls, name, bases, attrs):
#         if "dtype" not in attrs:
#             raise TypeError(f"{cls.__name__} requires a dtype")
#         return super().__new__(cls, name, bases, attrs)

#     def __getitem__(cls, dim):
#         if not isinstance(dim, int):
#             raise TypeError(
#                 f"{cls.__name__} requires exactly 1 parameters: {cls.__name__}[dim]"
#             )

#         if not isinstance(dim, int) or dim <= 0:
#             raise TypeError("Vector size must be a positive integer")

#         return Annotated[
#             List[cls.dtype],
#             Field(min_length=dim, max_length=dim, original_class=cls.__name__, dim=dim),
#         ]


# class SparseVector(List, metaclass=SparseVectorMeta):
#     """
#     Sparse vector field type for `Context` of the given dimension. Must be created with SparseVector[dim]
#     """

#     dtype: Type[Any] = float


class ArrayMeta(type):
    def __getitem__(cls, params):
        if isinstance(params, tuple) and len(params) == 2:
            dtype = params[0]
            max_size = params[1]
        elif isinstance(params, type) or get_origin(params) is Annotated:
            dtype = params
            max_size = None
        else:
            raise TypeError(
                f"{cls.__name__} requires either 2 parameters: {cls.__name__}[dtype, max_size] or 1 parameter: {cls.__name__}[dtype]"
            )

        assert isinstance(dtype, type) or get_origin(dtype) is Annotated, (
            f"dtype must be a type, got {dtype}"
        )
        assert max_size is None or (isinstance(max_size, int) and max_size > 0), (
            f"max_size must be an int or None, got {max_size}"
        )

        return Annotated[
            List[dtype],
            Field(
                min_length=0,
                max_length=max_size,
                json_schema_extra={"original_class": cls.__name__},
            ),
        ]


class Array(List, metaclass=ArrayMeta):
    """
    Array field type for `Context`. Must be created with Array[dtype, max_size]

    Args:
        dtype (Type): The type of the elements in the array.
        max_size (int): The maximum size of the array.

    Example:
        A `Email` context class that stores an email's content and recipients.
        ```python
        from modaic.types import Array
        from modaic.context import Context

        class Email(Context):
            content: str
            recipients: Array[str, 100]
        ```
    """

    pass


class StringMeta(type):
    def __getitem__(cls, params):
        if not isinstance(params, int):
            raise TypeError(
                f"{cls.__name__} requires exactly 1 parameters: {cls.__name__}[max_size]"
            )

        max_size = params
        if not isinstance(max_size, int) or max_size <= 1:
            raise TypeError(f"Max size must be a >= 1, got {max_size}")

        return Annotated[
            str,
            Field(
                max_length=max_size, json_schema_extra={"original_class": cls.__name__}
            ),
        ]


class String(str, metaclass=StringMeta):
    """String type that can be parameterized with max_length constraint.

    Args:
        max_size (int): The maximum length of the string.

    Example:
        ```python
        from modaic.types import String
        from modaic.context import Context

        class Email(Context):
            subject: String[100]
            content: str
            recipients: Array[str, 100]
        ```
    """

    pass


def fetch_type(metadata: list, type_class: Type) -> Optional[Type]:
    return next((x for x in metadata if isinstance(x, type_class)), None)


def get_original_class(field_info: FieldInfo, default: Optional[Type] = None) -> Type:
    if json_schema_extra := getattr(field_info, "json_schema_extra", None):
        return json_schema_extra.get("original_class", default)
    return default


Modaic_Type = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "bool",
    # "Vector",
    # "Float16Vector",
    # "Float32Vector",
    # "Float64Vector",
    # "BFloat16Vector",
    # "BinaryVector",
    "String",
    "Array",
]

allowed_types: Mapping[str, Modaic_Type] = {
    "Array": "Array",
    # "Vector": "Vector",
    "String": "String",
    "str": "String",
    "Mapping": "Mapping",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "float32": "float32",
    "float64": "float64",
    "double": "float64",
    "bool": "bool",
    "float": "float64",
    "int": "int64",
    "List": "Array",
    "list": "Array",
    # "Float16Vector": "Float16Vector",
    # "Float32Vector": "Float32Vector",
    # "Float64Vector": "Float64Vector",
    # "BFloat16Vector": "BFloat16Vector",
    # "BinaryVector": "BinaryVector",
}

listables: Mapping[str, Modaic_Type] = {
    "str": "String",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "int": "int64",
    "float32": "float32",
    "float64": "float64",
    "float": "float64",
    "double": "float64",
    "bool": "bool",
    "String": "String",
}


@dataclass
class InnerField:
    type: Modaic_Type
    size: Optional[int] = None


@dataclass
class SchemaField:
    type: Modaic_Type
    optional: bool = False
    size: Optional[int] = None
    inner_type: Optional[InnerField] = None


def unpack_type(field_type: Type) -> SchemaField:
    """
    Unpacks a type into a compatible modaic schema field.
    Modaic schema fields can be any of the following for type:
    - Array
    - Vector, Float16Vector, Float32Vector, Float64Vector, BFloat16Vector, BinaryVector
    - String
    - int8, int16, int32, int64, float32, float64, double(float64), bool, float(float64), int(int64)

    The function will return a SchemaField dataclass with the following fields:

    SchemaField - a dataclass with the following fields:
        optional (bool): Whether the field is optional.
        type (Type): The type of the field.
        size (int | None): The size of the field.
        inner_type (InnerField | None): The inner type of the field.

    InnerField - a dataclass with the following fields:
        type (Type): The type of the inner field.
        size (int | None): The size of the inner field.

    Args:
        field_type (Type): The type to unpack.

    Returns:
        SchemaField - a dataclass containing information to serialize the type.
    """
    # 1. Check if its Optional/Union
    if get_origin(field_type) is Union or get_origin(field_type) is UnionType:
        args = get_args(field_type)
        if len(args) == 2 and type(None) in args:
            not_none_type = args[0] if args[0] is not type(None) else args[1]
            return SchemaField(
                **{**asdict(unpack_type(not_none_type)), "optional": True}
            )
        else:
            raise ValueError(
                "Union's are not supported as modaic schemas. Except for Union[`Type`, None]"
            )
    # 2. Check if its an Annotated type
    elif get_origin(field_type) is Annotated:
        args = get_args(field_type)
        field_type = args[0]
        field_info = args[1]

        size = None
        if metadata := getattr(field_info, "metadata", None):
            max_len_obj = fetch_type(metadata, MaxLen)
            max_len = max_len_obj.max_length if max_len_obj else None
            if (
                max_len is not None
            ):  # Vector and Array types will have this Vector will have max_len==min_len
                size = max_len
        origin = (
            get_origin(field_type) if get_origin(field_type) is not None else field_type
        )
        simplified_type = origin.__name__
        if json_schema_extra := getattr(field_info, "json_schema_extra", None):
            simplified_type = json_schema_extra.get("original_class", simplified_type)
        if simplified_type in allowed_types:
            type_ = allowed_types[simplified_type]
        elif get_origin(field_type) is Union or get_origin(field_type) is UnionType:
            return SchemaField(**{**asdict(unpack_type(field_type))})
        elif issubclass(field_type, BaseModel) or issubclass(field_type, Mapping):
            type_ = "Mapping"
        else:
            raise ValueError(f"Type {simplified_type} is not allowed in Modaic models.")

        schema_field = SchemaField(
            **{**asdict(unpack_type(field_type)), "size": size, "type": type_}
        )

        return schema_field

    elif (
        field_type is list
        or get_origin(field_type) is List
        or get_origin(field_type) is list
    ):
        args = get_args(field_type)
        if len(args) == 1:
            inner_type = unpack_type(args[0])
            if inner_type.type in listables:
                # CAVEAT Only sets default value for list type. Notice this can be overwritten by annotated case if its the type annotated.
                # This helps handle cases like x: list[str], x: Array[int] (when using list data but no size is set)
                inner_type = InnerField(
                    type=inner_type.type, size=inner_type.size
                )  # CAVEAT: just grab out type and size so we don't get an error
                return SchemaField(inner_type=inner_type, type="Array", size=None)
            else:
                raise ValueError(f"type: {inner_type.type} is not listable")
        else:
            if len(args) > 1:
                raise ValueError(
                    f"List type {field_type} Can only store a single type."
                )
            else:
                raise ValueError(
                    f"Failed to convrert list type {field_type} with ambiguous dtype."
                )
    # Base cases
    elif isinstance(field_type, type) and (
        issubclass(field_type, BaseModel) or issubclass(field_type, Mapping)
    ):
        return SchemaField(type="Mapping", size=None)
    elif field_type.__name__ in allowed_types:
        type_ = allowed_types[field_type.__name__]
        return SchemaField(type=type_, size=None)
    else:
        raise ValueError(f"Type {field_type.__name__} is not allowed in Modaic models.")


def pydantic_to_modaic_schema(
    pydantic_model: Type[BaseModel],
) -> Dict[str, SchemaField]:
    """
    Unpacks a type into a dictionary of compatible modaic schema fields.
    Modaic schema fields can be any of the following for type:
    - Array
    - Vector, Float16Vector, Float32Vector, Float64Vector, BFloat16Vector, BinaryVector
    - String
    - int8, int16, int32, int64, float32, float64, double(float64), bool, float(float64), int(int64)

    The function will return a dictionary mapping field names to SchemaField dataclasses.

    SchemaField - a dataclass with the following fields:
        optional (bool): Whether the field is optional.
        type (Type): The type of the field.
        size (int | None): The size of the field.
        inner_type (InnerField | None): The inner type of the field.

    InnerField is a dataclass with the following fields:
        type (Type): The type of the inner field.
        size (int | None): The size of the inner field.

    Args:
        pydantic_model: The pydantic model to unpack.

    Returns:
        schema: A dictionary mapping field names to SchemaField dataclasses.
    """
    s: Dict[str, SchemaField] = {}
    for field_name, field_info in pydantic_model.model_fields.items():
        type_ = field_info.annotation
        new_field = copy.deepcopy(field_info)
        new_field.annotation = NoneType
        field_type = Annotated[type_, new_field]
        unpacked = unpack_type(field_type)
        s[field_name] = unpacked
    return s


# annotation=Union[
#     Annotated[
#         List[int], FieldInfo(
#             annotation=NoneType,
#             required=True,
#             json_schema_extra={'original_class': 'Array'},
#             metadata=[MinLen(min_length=0), MaxLen(max_length=10)])],
#     NoneType]
# required=True
