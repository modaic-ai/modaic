from pydantic.fields import FieldInfo, Field
from typing import Any
import json
from pydantic import TypeAdapter
from pydantic import BaseModel, field_serializer
from modaic.context import Context


def format_field_value(field_info: FieldInfo, value: Any, assume_text=True) -> str | dict:
    """
    Formats the value of the specified field according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself.

    Args:
      field_info: Information about the field, including its DSPy field type and annotation.
      value: The value of the field.
    Returns:
      The formatted value of the field, represented as a string.
    """
    string_value = None
    if isinstance(value, list) and field_info.annotation is str:
        # If the field has no special type requirements, format it as a nice numbered list for the LM.
        string_value = _format_input_list_field_value(value)
    else:
        # print("serializing for json")
        jsonable_value = serialize_for_json(value)
        if isinstance(jsonable_value, dict) or isinstance(jsonable_value, list):
            string_value = json.dumps(jsonable_value, ensure_ascii=False)
        else:
            # If the value is not a Python representation of a JSON object or Array
            # (e.g. the value is a JSON string), just use the string representation of the value
            # to avoid double-quoting the JSON string (which would hurt accuracy for certain
            # tasks, e.g. tasks that rely on computing string length)
            string_value = str(jsonable_value)

    if assume_text:
        return string_value
    else:
        return {"type": "text", "text": string_value}


def _format_input_list_field_value(value: list[Any]) -> str:
    """
    Formats the value of an input field of type list[Any].

    Args:
      value: The value of the list-type input field.
    Returns:
      A string representation of the input field's list value.
    """
    if len(value) == 0:
        return "N/A"
    if len(value) == 1:
        return _format_blob(value[0])

    return "\n".join([f"[{idx + 1}] {_format_blob(txt)}" for idx, txt in enumerate(value)])


def _format_blob(blob: str) -> str:
    """
    Formats the specified text blobs so that an LM can parse it correctly within a list
    of multiple text blobs.

    Args:
        blob: The text blob to format.
    Returns:
        The formatted text blob.
    """
    if "\n" not in blob and "«" not in blob and "»" not in blob:
        return f"«{blob}»"

    modified_blob = blob.replace("\n", "\n    ")
    return f"«««\n    {modified_blob}\n»»»"


def serialize_for_json(value: Any) -> Any:
    """
    Formats the specified value so that it can be serialized as a JSON string.

    Args:
        value: The value to format as a JSON string.
    Returns:
        The formatted value, which is serializable as a JSON string.
    """
    # Attempt to format the value as a JSON-compatible object using pydantic, falling back to
    # a string representation of the value if that fails (e.g. if the value contains an object
    # that pydantic doesn't recognize or can't serialize)
    # print("type(value)", type(value))
    try:
        return TypeAdapter(type(value)).dump_python(value, mode="json")
    except Exception:
        return str(value)


class CustomContext(Context):
    a: str
    b: int


class DSPyInput(BaseModel):
    a: str
    b: int
    ctx: CustomContext


ctx = CustomContext(a="test", b=1)
dspy_input = DSPyInput(a="test", b=1, ctx=ctx)

formatted_ctx = format_field_value(DSPyInput.model_fields["ctx"], dspy_input.ctx)
# print(formatted_ctx)
# print(type(formatted_ctx))

t = TypeAdapter(CustomContext)
# # print(isinstance(t, BaseModel))
# # print(type(t))
dumped = t.dump_python(ctx, mode="json")
# print("dspy dump", dumped)

ctx.model_rebuild()
dumped = t.dump_python(ctx, mode="json")
# print("dspy dump (rebuilt)", dumped)
# print("dspy dump", dumped)
# print("pydantic dump", ctx.model_dump(mode="json"))

dumped = ctx.model_dump(mode="json")
dumped2 = ctx.model_dump(mode="json", show_hidden=True)
print("dumped", dumped)
print("dumped2", dumped2)
# class TestModel(BaseModel):
#     a: str = Field(exclude=True)
#     b: int
#     l: list[int] = Field(default_factory=lambda: [1, 2, 3])

#     @field_serializer("a", check_fields=False)
#     def _ser_a(self, v, info):
#         # Hide unless caller explicitly asks for hidden fields
#         print("serializing a")
#         return self.a

#     @field_serializer("b")
#     def serialize_b(self, b: int) -> str:
#         print("serializing b")
#         return self.b


# test_model = TestModel(a="test", b=1)
# print(test_model.model_validate())
# print(test_model.model_dump(include={"a": True, "b": True, "l": True}))
