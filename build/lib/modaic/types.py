from typing import Any, List, Literal, Optional, Tuple, Type, get_origin

from pydantic import (
    BaseModel,
    RootModel,
    WithJsonSchema,
    field_validator,
    model_validator,
)
from pydantic import (
    Field as PydanticField,
)
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from typing_extensions import Annotated

from .exceptions import SchemaError

int8 = Annotated[
    int,
    PydanticField(ge=-128, le=127),
    WithJsonSchema({"type": "integer", "format": "int8"}),
]
int16 = Annotated[
    int,
    PydanticField(ge=-32768, le=32767),
    WithJsonSchema({"type": "integer", "format": "int16"}),
]
int32 = Annotated[
    int,
    PydanticField(ge=-(2**31), le=2**31 - 1),
    WithJsonSchema({"type": "integer", "format": "int32"}),
]
int64 = Annotated[
    int,
    PydanticField(ge=-(2**63), le=2**63 - 1),
    WithJsonSchema({"type": "integer", "format": "int64"}),
]
double = Annotated[
    float,
    PydanticField(ge=-1.87e308, le=1.87e308),
    WithJsonSchema({"type": "number", "format": "double"}),
]


class ArrayMeta(type):
    def __getitem__(cls, params: type | tuple[type, int]):
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

        assert isinstance(dtype, type) or get_origin(dtype) is Annotated, f"dtype must be a type, got {dtype}"
        assert max_size is None or (isinstance(max_size, int) and max_size > 0), (
            f"max_size must be an int or None, got {max_size}"
        )

        return Annotated[
            List[dtype],
            PydanticField(min_length=0, max_length=max_size),
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
    def __getitem__(cls, params: int):
        if not isinstance(params, int):
            raise TypeError(f"{cls.__name__} requires exactly 1 parameters: {cls.__name__}[max_size]")

        max_size = params
        if not isinstance(max_size, int) or max_size <= 1:
            raise TypeError(f"Max size must be a >= 1, got {max_size}")

        return Annotated[
            str,
            PydanticField(max_length=max_size),
            WithJsonSchema({"type": "string", "maxLength": max_size}),
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


int_format = Literal["int8", "int16", "int32", "int64"]
float_format = Literal["float", "double"]


class SchemaField(BaseModel):
    optional: bool
    type: Literal["array", "integer", "number", "object", "string", "boolean"]
    format: int_format | float_format | None
    size: Optional[int]
    inner_type: Optional["InnerField"]
    is_id: bool = False
    is_unique: bool = False

    @model_validator(mode="after")
    def validate_field(self) -> "SchemaField":
        if self.type == "array" and self.inner_type is None:
            raise SchemaError("Array type must have an inner type")
        if self.is_id and not self.is_unique:
            raise SchemaError("id field must be unique")
        if self.is_id and self.optional:
            raise SchemaError("id field cannot be optional")
        if self.is_id and self.type != "string":
            raise SchemaError("id field must be a string")
        # NOTE: handle case where the float type was used and therefore not annotated with a format
        if self.type == "number":
            self.format = self.format or "float"
        return self

    @staticmethod
    def from_json_schema_property(
        field_schema: dict, is_id: bool = False, is_unique: Optional[bool] = None
    ) -> "SchemaField":
        inspected_type, is_optional = _inspect_type(field_schema)
        if "maxItems" in inspected_type and "maxLength" in inspected_type:
            raise SchemaError("maxItems and maxLength cannot both be present in a schema field")
        if "items" in inspected_type:
            inner_type = InnerField.from_json_schema_property(inspected_type["items"])
        else:
            inner_type = None
        if is_unique is None:
            is_unique = is_id

        return SchemaField(
            optional=is_optional,
            type=inspected_type["type"],
            format=inspected_type.get("format", None),
            size=inspected_type.get("maxItems", None) or inspected_type.get("maxLength", None),
            inner_type=inner_type,
            is_id=is_id,
            is_unique=is_unique,
        )


class InnerField(BaseModel):
    type: Literal["integer", "number", "string", "boolean"]
    format: int_format | float_format | None = None
    size: Optional[int] = None

    @model_validator(mode="after")
    def validate_field(self) -> "InnerField":
        if self.type == "number":
            self.format = self.format or "float"
        return self

    @staticmethod
    def from_json_schema_property(inner_schema: dict) -> "InnerField":
        inspected_type, is_optional = _inspect_type(inner_schema)
        # NOTE: handle case where the float type was used and therefore not annotated with a format
        if is_optional:
            raise SchemaError("Array/List elements cannot be None/null")
        if inspected_type["type"] == "object" or inspected_type["type"] == "array":
            raise SchemaError("Arrays and Dicts are not supported for Array/List elements")
        return InnerField(
            type=inspected_type["type"],
            format=inspected_type.get("format", None),
            size=inspected_type.get("size", None),
        )


class Schema(RootModel[dict[str, SchemaField]]):
    @field_validator("root")
    @classmethod
    def validate_is_id(cls, v: dict[str, SchemaField]) -> dict[str, SchemaField]:
        offenders = [k for k, sf in v.items() if sf.is_id and k != "id"]
        if offenders:
            raise SchemaError(
                "is_id can only be True for the key 'id'; offending keys: " + ", ".join(repr(k) for k in offenders)
            )
        return v

    @staticmethod
    def from_json_schema(schema: dict) -> "Schema":
        """
        Converts an OpenAPI JSON schema to a Modaic schema that can be used with modaic databases.
        Warnings:
            Not designed to handle all edge cases of OpenAPI schemas. Only designed to work with jsons output by pydantics BaseModel model_json_schema()
        """
        validated_fields = {}
        for field_name, field_schema in schema["properties"].items():
            schema_field = SchemaField.from_json_schema_property(field_schema, is_id=field_name == "id")
            validated_fields[field_name] = schema_field
        return Schema(validated_fields)

    def as_dict(self) -> dict[str, SchemaField]:
        return self.root


def _inspect_type(field_schema: dict) -> Tuple[dict, bool]:
    """
    This function inspects the json schema ensuring it is a valid modaic schema. Returns from this function are guaranteed to:
    1. Be a dictionary containing the key "type"
    2. Not have unions other than a single union with null
    3. All {"$ref": "..."} is replaced with {"type": "object"}
    Returns:
        Tuple[dict, bool]: the dict containing the type, and a boolean indicating if the field is optional
    """
    if anyOf := field_schema.get("anyOf", None):  # noqa: N806
        if len(anyOf) > 2 or not any(_is_null(item) for item in anyOf):
            raise SchemaError("Unions are not supported for Modaic Schemas")
        elif any(not _is_null(type_ := item) for item in anyOf):
            return _handle_if_ref(type_), True
        else:
            raise SchemaError("Invalid field schema")
    elif "type" in field_schema:
        return _handle_if_ref(field_schema), False
    elif "$ref" in field_schema:
        return {"type": "object"}, False
    else:
        raise SchemaError("Invalid field schema")


def _handle_if_ref(field_schema: dict) -> dict:
    """
    Handles the case where the field is a reference to another schema. Returns an {"type": "object"}
    """
    if "$ref" in field_schema:
        return {"type": "object"}
    else:
        return field_schema


def _is_null(field_schema: dict) -> bool:
    return field_schema.get("type", "") == "null"


def Field(default: Any = PydanticUndefined, *, hidden: bool = False, **kwargs) -> FieldInfo:  # noqa: N802, ANN001
    if hidden:
        return PydanticField(default=default, **kwargs, json_schema_extra={"hidden": True})
    else:
        return PydanticField(default=default, **kwargs)
