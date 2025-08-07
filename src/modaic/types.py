from typing import List, Type, Any, TypedDict, Dict, get_origin, get_args, Optional
from typing_extensions import Annotated
from pydantic import Field, conint, confloat, constr, BaseModel
from pydantic.types import StringConstraints
from collections.abc import Mapping, Sequence

int8 = conint(ge=-128, le=127)
int16 = conint(ge=-32768, le=32767)
int32 = conint(ge=-(2**31), le=2**31 - 1)
int64 = conint(ge=-(2**63), le=2**63 - 1)
float32 = confloat(ge=-3.40e38, le=3.40e38)
float64 = confloat(ge=-1.87e308, le=1.87e308)
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
            Field(min_length=dim, max_length=dim, modaic_type=cls, dim=dim),
        ]


class Vector(List, metaclass=VectorMeta):
    dtype: Type[Any] = float


class Float16Vector(Vector):
    dtype = confloat(ge=-65504, le=65504)


class Float32Vector(Vector):
    dtype = float32


class Float64Vector(Vector):
    dtype = float64


class BFloat16Vector(Vector):
    dtype = confloat(ge=-3.39e38, le=3.39e38)


class BinaryVector(Vector):
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
            Field(
                min_length=0, max_length=max_size, modaic_type=cls, max_size=max_size
            ),
        ]


class Array(List, metaclass=ArrayMeta):
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

        return Annotated[str, Field(max_length=max_size)]


class String(str, metaclass=StringMeta):
    """String type that can be parameterized with max_length constraint.

    Can be used as:
    - String[50] for type annotations with max length validation
    - String("hello") for creating string instances
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


def pydantic_model_to_schema(pydantic_model: Type[BaseModel]) -> Dict[str, SchemaField]:
    schema: Dict[str, SchemaField] = {}
    for field_name, field_info in pydantic_model.model_fields.items():
        schema_field: SchemaField = {}
        # print(field_info)
        field_type = field_info.annotation
        origin = get_origin(field_type)
        if origin is not None:
            args = get_args(field_type)
            schema_field["optional"] = type(None) in args
        else:
            schema_field["optional"] = False
        
        if isinstance(field)

        if (
            field_info.json_schema_extra
            and "modaic_type" in field_info.json_schema_extra
        ):
            extra = field_info.json_schema_extra
            schema_field["type"] = extra["modaic_type"]
            type_ = schema_field["type"]
            if type_ is Array:
                schema_field["max_size"] = extra["max_size"]
            elif type_ is Vector:
                schema_field["dim"] = extra["dim"]
            elif type_ is String:  # Handle string with max_size
                schema_field["max_length"] = extra["max_size"]
        elif isinstance(field_type, Mapping):
            schema_field["type"] = Mapping
        elif isinstance(
            field_type, String
        ):  # edge case for String with out [max_size] specified
            pass
        elif isinstance(field_type, Sequence):
            raise ValueError(
                f"""Error converting pydantic model: {pydantic_model} to schema. Unsupported field type: {field_type}. 
                Sequence type is not supported for Modaic schemas. Use modaic.typing.Array, modaic.typing.Vector, or modaic.typing.String instead.
                """
            )
        else:
            schema_field["type"] = field_info.annotation

        if schema_field["type"] not in allowed_types:
            raise ValueError(
                f"""Error converting pydantic model: {pydantic_model} to schema. Unsupported field type: {field_type}.
                """
            )

        schema[field_name] = schema_field
    return schema
