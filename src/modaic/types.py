from typing import (
    List,
    Type,
    Any,
    TypedDict,
    Dict,
    get_origin,
    get_args,
    Optional,
    Union,
)
from typing_extensions import Annotated
from pydantic import Field, conint, confloat, constr, BaseModel
from pydantic.types import StringConstraints
from collections.abc import Mapping, Sequence
from annotated_types import Gt, Le, MinLen, MaxLen


int8 = Annotated[int, Field(ge=-128, le=127)]
int16 = Annotated[int, Field(ge=-32768, le=32767)]
int32 = Annotated[int, Field(ge=-(2**31), le=2**31 - 1)]
int64 = Annotated[int, Field(ge=-(2**63), le=2**63 - 1)]
float32 = Annotated[float, Field(ge=-3.40e38, le=3.40e38)]
float64 = Annotated[float, Field(ge=-1.87e308, le=1.87e308)]
double = float64


class VectorMeta(type):
    def __new__(cls, name, bases, attrs):
        if "dtype" not in attrs:
            raise TypeError(f"{cls.__name__} requires a dtype")
        return super().__new__(cls, name, bases, attrs)

    def __getitem__(cls, dim):
        print(f"params: {dim}")
        if not isinstance(dim, int):
            raise TypeError(
                f"{cls.__name__} requires exactly 1 parameters: {cls.__name__}[dim]"
            )

        if not isinstance(dim, int) or dim <= 0:
            raise TypeError("Vector size must be a positive integer")

        return Annotated[
            List[cls.dtype],
            Field(min_length=dim, max_length=dim, original_class=cls, dim=dim),
        ]


class Vector(List, metaclass=VectorMeta):
    """
    float vector field type for `SerializedContext` of the given dimension. Must be created with Vector[dim]

    Args:
        dim (int): Required. The dimension of the vector.

    Example:
        The `SerializedContext` class for a `CaptionedImage` Context type that stores both a primary embedding using the image and a secondary embedding using the caption.
        ```python
        from modaic.types import Vector
        from modaic.context import SerializedContext

        class SerializedCaptionedImage(SerializedContext):
            caption: String[100]
            caption_embedding: Vector[384]
        ```
    """

    dtype: Type[Any] = float


class Float16Vector(Vector):
    """
    float16 vector field type for `SerializedContext` of the given dimension. Must be created with Float16Vector[dim]

    Args:
        dim (int): Required. The dimension of the vector.

    Example:
        ```python
        from modaic.types import Float16Vector
        from modaic.context import SerializedContext

        # Case where we want to store a secondary embedding for the caption of an image.
        class SerializedCaptionedImage(SerializedContext):
            caption: String[100]
            caption_embedding: Float16Vector[384]
        ```
    """

    dtype = Annotated[float, Field(ge=-65504, le=65504)]


class Float32Vector(Vector):
    """
    float32 vector field type for `SerializedContext` of the given dimension. Must be created with Float32Vector[dim]

    Args:
        dim (int): Required. The dimension of the vector.

    Example:
        The `SerializedContext` class for a `CaptionedImage` Context type that stores both a primary embedding using the image and a secondary embedding using the caption.
        ```python
        from modaic.types import Float32Vector
        from modaic.context import SerializedContext

        class SerializedCaptionedImage(SerializedContext):
            caption: String[100]
            caption_embedding: Float32Vector[384]
        ```
    """

    dtype = float32


class Float64Vector(Vector):
    """
    float64 vector field type for `SerializedContext` of the given dimension. Must be created with Float64Vector[dim]

    Args:
        dim (int): Required. The dimension of the vector.

    Example:
        The `SerializedContext` class for a `CaptionedImage` Context type that stores both a primary embedding using the image and a secondary embedding using the caption.
        ```python
        from modaic.types import Float64Vector
        from modaic.context import SerializedContext

        class SerializedCaptionedImage(SerializedContext):
            caption: String[100]
            caption_embedding: Float64Vector[384]
        ```
    """

    dtype = float64


class BFloat16Vector(Vector):
    """
    bfloat16 vector field type for `SerializedContext` of the given dimension. Must be created with BFloat16Vector[dim]

    Args:
        dim (int): Required. The dimension of the vector.

    Example:
        The `SerializedContext` class for a `CaptionedImage` Context type that stores both a primary embedding using the image and a secondary embedding using the caption.
        ```python
        from modaic.types import BFloat16Vector
        from modaic.context import SerializedContext

        class SerializedCaptionedImage(SerializedContext):
            caption: String[100]
            caption_embedding: BFloat16Vector[384]
        ```
    """

    dtype = Annotated[float, Field(ge=-3.4e38, le=3.40e38)]


class BinaryVector(Vector):
    """
    binary vector field type for `SerializedContext` of the given dimension. Must be created with BinaryVector[dim]

    Args:
        dim (int): Required. The dimension of the vector.

    Example:
        The `SerializedContext` class for a `SenateBill` Context type that uses a binary vector to store the vote distribution.
        ```python
        from modaic.types import BinaryVector
        from modaic.context import SerializedContext

        class SerializedSenateBill(SerializedContext):
            bill_id: int
            bill_title: String[10]
            bill_description: String
            vote_distribution: BinaryVector[100]
        ```
    """

    dtype = bool


class ArrayMeta(type):
    def __getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError(
                f"{cls.__name__} requires exactly 2 parameters: {cls.__name__}[dtype, max_size]"
            )

        dtype = params[0]
        max_size = params[1]
        assert isinstance(dtype, Type), f"dtype must be a type, got {dtype}"
        if not isinstance(max_size, int) or max_size <= 1:
            raise TypeError(f"Max size must be a >= 1, got {max_size}")

        return Annotated[
            List[dtype],
            Field(min_length=0, max_length=max_size, original_class=cls),
        ]


class Array(List, metaclass=ArrayMeta):
    """
    Array field type for `SerializedContext`. Must be created with Array[dtype, max_size]

    Args:
        dtype (Type): The type of the elements in the array.
        max_size (int): The maximum size of the array.

    Example:
        A `SerializedEmail` for `Email` context class that stores an email's content and recipients.
        ```python
        from modaic.types import Array
        from modaic.context import SerializedContext

        class SerializedEmail(SerializedContext):
            content: str
            recipients: Array[str, 100]
        ```
    """

    pass


class StringMeta(type):
    def __getitem__(cls, params):
        print(f"params: {params}")
        if not isinstance(params, int):
            raise TypeError(
                f"{cls.__name__} requires exactly 1 parameters: {cls.__name__}[max_size]"
            )

        max_size = params
        if not isinstance(max_size, int) or max_size <= 1:
            raise TypeError(f"Max size must be a >= 1, got {max_size}")

        return Annotated[str, Field(max_length=max_size, original_class=cls)]


class String(str, metaclass=StringMeta):
    """String type that can be parameterized with max_length constraint.

    Args:
        max_size (int): The maximum length of the string.

    Example:
        ```python
        from modaic.types import String
        from modaic.context import SerializedContext

        class SerializedEmail(SerializedContext):
            subject: String[100]
            content: str
            recipients: Array[str, 100]
        ```
    """

    pass


class SchemaField(TypedDict):
    optional: bool
    type: Type
    max_size: Optional[int] = None
    dim: Optional[int] = None


allowed_types = {
    Array: Array,
    Vector: Vector,
    String: String,
    str: String,
    Mapping: Mapping,
    int8: int8,
    int16: int16,
    int32: int32,
    int64: int64,
    float32: float32,
    float64: float64,
    double: double,
    bool: bool,
    float: float64,
    int: int64,
}


def fetch_type(metadata: list, type_class: Type) -> Optional[Type]:
    return next((x for x in metadata if isinstance(x, type_class)), None)


def pydantic_model_to_schema(pydantic_model: Type[BaseModel]) -> Dict[str, SchemaField]:
    schema: Dict[str, SchemaField] = {}
    for field_name, field_info in pydantic_model.model_fields.items():
        schema_field: SchemaField = {}
        # print(field_info)
        field_type = field_info.annotation
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                schema_field["optional"] = True
            elif len(args) > 2:
                raise ValueError(
                    "Union's are not supported. For modaic schemas. Except for single unions with None (Optional type)"
                )
            else:
                schema_field["optional"] = False
        else:
            schema_field["optional"] = False

        if metadata := getattr(field_info, "metadata", None) is not None:
            max_len_obj = fetch_type(metadata, MaxLen)
            min_len_obj = fetch_type(metadata, MinLen)
            max_len = max_len_obj.max_length if max_len_obj else None
            min_len = min_len_obj.min_length if min_len_obj else None
            if max_len is not None and min_len is not None and min_len == max_len:
                schema_field["dim"] = max_len
            elif max_len is not None:
                schema_field["max_size"] = max_len

        if extra_metadata := getattr(field_info, "extra", None) is not None:
            pass

    return schema
