import sys
import traceback
import typing as t
from typing import TYPE_CHECKING, Annotated, Optional, Tuple, Type

import dspy
from dspy import InputField, OutputField, make_signature
from pydantic import BeforeValidator, Field, PlainSerializer, create_model
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue

if TYPE_CHECKING:
    from pydantic.json_schema import CoreSchemaOrField

INCLUDED_FIELD_KWARGS = {
    "desc",
    "alias",
    "alias_priority",
    "validation_alias",
    "serialization_alias",
    "title",
    "description",
    "exclude",
    "discriminator",
    "deprecated",
    "frozen",
    "validate_default",
    "repr",
    "init",
    "init_var",
    "kw_only",
    "pattern",
    "strict",
    "coerce_numbers_to_str",
    "gt",
    "ge",
    "lt",
    "le",
    "multiple_of",
    "allow_inf_nan",
    "max_digits",
    "decimal_places",
    "min_length",
    "max_length",
    "union_mode",
    "fail_fast",
}


def _handle_any_of(obj: dict, defs: Optional[dict] = None) -> t.Type:
    """
    Deserializes anyOf into a union type
    """
    return t.Union[tuple(json_to_type(item, defs) for item in obj["anyOf"])]


def _handle_object(obj: dict, defs: Optional[dict] = None) -> dict:
    """
    Deserializes basic objects types into dict type
    """
    additional_properties = obj.get("additionalProperties")
    if additional_properties == True:  # noqa: E712 we need to expliclity check for True, not just truthy
        return dict
    value_type = json_to_type(additional_properties, defs)
    return dict[str, value_type]


def _handle_array(obj: dict, defs: Optional[dict] = None) -> list:
    """
    Deserializes arrays into lists, sets, or tuple type.
    """
    if (items := obj.get("items")) is not None:
        set_or_list = set if obj.get("uniqueItems") else list

        if items == {}:
            return set_or_list
        else:
            return set_or_list[json_to_type(items, defs)]

    elif "maxItems" in obj and "minItems" in obj and (prefix_items := obj.get("prefixItems")):
        item_types = tuple(json_to_type(item, defs) for item in prefix_items)
        return Tuple[item_types]
    else:
        raise ValueError(f"Invalid array: {obj}")


def _handle_custom_type(ref: str, defs: Optional[dict] = None) -> t.Type:
    """
    Deserializes custom types defined in $def into dspy special types and BaseModels
    """
    # CAVEAT if user defines custom types that overlap with these names they will be overwritten by the dspy types
    dspy_types = {
        "dspy.Image": dspy.Image,
        "dspy.Audio": dspy.Audio,
        "dspy.History": dspy.History,
        "dspy.Tool": dspy.Tool,
        "dspy.ToolCalls": dspy.ToolCalls,
        "dspy.Code": dspy.Code,
    }
    name = ref.split("/")[-1]
    obj = defs[name]
    if dspy_type := dspy_types.get(obj["type"]):
        return dspy_type
    if obj["type"] == "object":
        fields = {}
        for name, field in obj["properties"].items():
            field_kwargs = {k: v for k, v in field.items() if k in INCLUDED_FIELD_KWARGS}
            if default := field.get("default"):
                fields[name] = (json_to_type(field, defs), Field(default=default, **field_kwargs))
            else:
                fields[name] = (json_to_type(field, defs), Field(..., **field_kwargs))
        return create_model(name, **fields)

    else:
        raise ValueError(f"Invalid type: {obj}")


def json_to_type(json_type: dict, defs: Optional[dict] = None) -> t.Type:
    """
    Desserializes a json schema into a python type
    """
    primitive_types = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "null": None,
    }
    if j_type := json_type.get("type"):
        if j_type in primitive_types:
            return primitive_types[j_type]
        elif j_type == "array":
            return _handle_array(json_type, defs)
        elif j_type == "object":
            return _handle_object(json_type, defs)
        else:
            raise ValueError(f"Invalid type: {j_type}")
    elif ref := json_type.get("$ref"):
        return _handle_custom_type(ref, defs)
    elif json_type.get("anyOf"):
        return _handle_any_of(json_type, defs)
    else:
        raise ValueError(f"Invalid json schema: {json_type}")


def _deserialize_dspy_signatures(obj: dict | Type[dspy.Signature]) -> Type[dspy.Signature]:
    """
    Deserizlizes a dictionary into a DSPy signature. Not all signatures can be deserialized.
    - All fields (and fields of fields) cannot have default factories
    - Frozensets will be serialized to sets
    - tuples without arguments will be serialized to lists
    """
    try:
        if issubclass(obj, dspy.Signature):
            return obj
        fields = {}
        defs = obj.get("$defs", {})
        properties: dict[str, dict] = obj.get("properties", {})
        for name, field in properties.items():
            field_kwargs = {k: v for k, v in field.items() if k in INCLUDED_FIELD_KWARGS}
            InputOrOutputField = InputField if field.get("__dspy_field_type") == "input" else OutputField  # noqa: N806
            if default := field.get("default"):
                fields[name] = (json_to_type(field, defs), InputOrOutputField(default=default, **field_kwargs))
            else:
                fields[name] = (json_to_type(field, defs), InputOrOutputField(**field_kwargs))
        signature = make_signature(
            signature=fields, instructions=obj.get("description"), signature_name=obj.get("title")
        )
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
    return signature


class DSPyTypeSchemaGenerator(GenerateJsonSchema):
    def generate_inner(self, schema: "CoreSchemaOrField") -> JsonSchemaValue:
        cls = schema.get("cls")
        super_generate_inner = super().generate_inner

        def handle_dspy_type(name: str) -> dict:
            schema["metadata"]["pydantic_js_functions"] = [lambda cls, core_schema: {"type": f"dspy.{name}"}]
            return super_generate_inner(schema)

        for dspy_type in [dspy.Image, dspy.Audio, dspy.History, dspy.Tool, dspy.ToolCalls, dspy.Code]:
            if cls is dspy_type:
                return handle_dspy_type(dspy_type.__name__)
        return super_generate_inner(schema)


SerializableSignature = Annotated[
    Type[dspy.Signature],
    BeforeValidator(_deserialize_dspy_signatures),
    PlainSerializer(lambda s: s.model_json_schema(schema_generator=DSPyTypeSchemaGenerator)),
]
