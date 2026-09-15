import copy
import inspect
import typing as t
from typing import TYPE_CHECKING, Annotated, Optional, Tuple, Type

import dspy
from dspy import InputField, OutputField, make_signature
from pydantic import BeforeValidator, Field, PlainSerializer, create_model
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue

from modaic.types import Enum, Scale

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

DSPY_CUSTOM_TYPES = {
    "dspy.Image": dspy.Image,
    "dspy.Audio": dspy.Audio,
    "dspy.History": dspy.History,
    "dspy.Tool": dspy.Tool,
    "dspy.ToolCalls": dspy.ToolCalls,
    "dspy.Code": dspy.Code,
    "dspy.Reasoning": dspy.Reasoning,
}
_DSPY_MEDIA_TYPES = (dspy.Image, dspy.Audio)
_DSPY_BASE_TYPE_SCHEMA_KEY = "x-modaic-dspy-base-type"
_DSPY_MEDIA_TYPES_BY_SCHEMA_NAME = {f"dspy.{dspy_type.__name__}": dspy_type for dspy_type in _DSPY_MEDIA_TYPES}
_DEFS_REF_PREFIX = "#/$defs/"


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


def _handle_custom_type(
    ref: str,
    defs: Optional[dict] = None,
) -> t.Type:
    """
    Deserializes custom types defined in $def into dspy special types and BaseModels
    """
    # CAVEAT if user defines custom types that overlap with these names they will be overwritten by the dspy types
    definition_name = ref.split("/")[-1]
    obj = defs[definition_name]
    dspy_base_type = obj.get(_DSPY_BASE_TYPE_SCHEMA_KEY)
    if dspy_type := DSPY_CUSTOM_TYPES.get(obj["type"]):
        return dspy_type
    if obj["type"] == "object":
        fields = {}
        for field_name, field in obj["properties"].items():
            field_kwargs = {k: v for k, v in field.items() if k in INCLUDED_FIELD_KWARGS}
            if "default" in field:
                fields[field_name] = (
                    json_to_type(field, defs),
                    Field(default=field["default"], **field_kwargs),
                )
            else:
                fields[field_name] = (json_to_type(field, defs), Field(..., **field_kwargs))
        if base_type := _DSPY_MEDIA_TYPES_BY_SCHEMA_NAME.get(dspy_base_type):
            return create_model(definition_name, __base__=base_type, __doc__=obj.get("description"), **fields)
        return create_model(definition_name, __doc__=obj.get("description"), **fields)

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
    # modaic.Scale / modaic.Enum tag their serialized schema so the round-trip can
    # rebuild the original annotation instead of collapsing to a plain Literal.
    # Checked before const/enum so the marker wins over the underlying literal.
    #
    # Scale[...] / Enum[...] are themselves Annotated[Literal[...], <validator>]: the Literal
    # makes them real typing aliases that dspy.make_signature accepts, and pydantic keeps the
    # validator in field.metadata, so loose coercion + the marker survive every signature
    # rebuild — the round-trip is idempotent.
    if (modaic_type := json_type.get("x-modaic-type")) is not None:
        args = json_type["x-modaic-args"]
        if modaic_type == "Scale":
            return Scale[args[0], args[1]]
        if modaic_type == "Enum":
            return Enum[tuple(args)]
    if "const" in json_type:
        return t.Literal[json_type["const"]]
    elif enum := json_type.get("enum"):
        return t.Literal.__getitem__(tuple(enum))
    elif j_type := json_type.get("type"):
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


def _deserialize_dspy_signatures(
    obj: dict | Type[dspy.Signature],
) -> Type[dspy.Signature]:
    """
    Deserizlizes a dictionary into a DSPy signature. Not all signatures can be deserialized.
    - All fields (and fields of fields) cannot have default factories
    - Frozensets will be serialized to sets
    - tuples without arguments will be serialized to lists
    """
    if inspect.isclass(obj) and issubclass(obj, dspy.Signature):
        return obj
    fields = {}
    defs = obj.get("$defs", {})
    properties: dict[str, dict] = obj.get("properties", {})
    for name, field in properties.items():
        field_kwargs = {k: v for k, v in field.items() if k in INCLUDED_FIELD_KWARGS}
        InputOrOutputField = InputField if field.get("__dspy_field_type") == "input" else OutputField  # noqa: N806
        if "default" in field:
            fields[name] = (
                json_to_type(field, defs),
                InputOrOutputField(default=field["default"], **field_kwargs),
            )
        else:
            fields[name] = (
                json_to_type(field, defs),
                InputOrOutputField(**field_kwargs),
            )
    signature = make_signature(
        signature=fields,
        instructions=obj.get("description"),
        signature_name=obj.get("title"),
    )
    return signature


class DSPyTypeSchemaGenerator(GenerateJsonSchema):
    def generate_inner(self, schema: "CoreSchemaOrField") -> JsonSchemaValue:
        cls = schema.get("cls")
        super_generate_inner = super().generate_inner

        def handle_dspy_type(name: str) -> dict:
            # Pydantic may hand the generator a reference to the model class's
            # shared core schema. Never mutate it while adding our JSON-only
            # marker or later calls to model_json_schema() can inherit it.
            tagged_schema = dict(schema)
            metadata = dict(schema.get("metadata") or {})
            metadata["pydantic_js_functions"] = [lambda cls, core_schema: {"type": f"dspy.{name}"}]
            tagged_schema["metadata"] = metadata
            return super_generate_inner(tagged_schema)

        for dspy_type in DSPY_CUSTOM_TYPES.values():
            if cls is dspy_type:
                return handle_dspy_type(dspy_type.__name__)

        for dspy_type in _DSPY_MEDIA_TYPES:
            if isinstance(cls, type) and issubclass(cls, dspy_type):
                structural_schema = super_generate_inner(schema)
                marker = f"dspy.{dspy_type.__name__}"
                ref = structural_schema.get("$ref")
                if not isinstance(ref, str) or not ref.startswith(_DEFS_REF_PREFIX):
                    raise RuntimeError(
                        f"Expected Pydantic to emit a $defs reference for DSPy media subclass {cls.__name__}"
                    )

                # Pydantic can discard siblings of a reused $ref. Tag its
                # canonical definition so every reference retains the type.
                definition_name = ref.removeprefix(_DEFS_REF_PREFIX)
                definition = self.definitions.get(definition_name)
                if definition is None:
                    raise RuntimeError(f"Missing JSON Schema definition for DSPy media subclass {cls.__name__}")
                self.definitions[definition_name] = {
                    **definition,
                    _DSPY_BASE_TYPE_SCHEMA_KEY: marker,
                }
                return structural_schema
        return super_generate_inner(schema)


def _deserialize_dspy_lm(lm: dict | dspy.LM) -> dspy.LM:
    if type(lm) is dspy.LM:
        return lm
    if isinstance(lm, dict):
        return dspy.LM(**lm)


def serialize_signature(s: dspy.Signature) -> dict:
    signature = copy.deepcopy(s)
    return signature.model_json_schema(schema_generator=DSPyTypeSchemaGenerator)


SerializableSignature = Annotated[
    Type[dspy.Signature],
    BeforeValidator(_deserialize_dspy_signatures),
    PlainSerializer(serialize_signature),
]
