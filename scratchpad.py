from typing import (
    Optional,
    get_origin,
    get_args,
    Union,
    List,
    Sequence,
    Type,
)
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from modaic.types import (
    Vector,
    Float16Vector,
    Array,
    String,
    int32,
    pydantic_model_to_schema,
)
from modaic.databases.integrations.milvus import _convert_scontext_to_milvus_schema
from annotated_types import Gt, Le, MinLen, MaxLen
from types import UnionType

# print("Union is instance", isinstance(Union, Union))


class Model(BaseModel):
    x: Array[int, 10]
    y: Optional[Array[int, 10]]
    z: list
    w: str
    v: int32
    s: String[10]
    a: Vector[10]
    test: Union[int, str, None]


# _, required_array = list(Model.model_fields.items())[0]
# print("required_array", required_array)
# print(isinstance(required_array, Sequence))
# print("metadata type", type(required_array))


# def fetch_type(metadata: list, type_class: Type) -> Optional[Type]:
#     return next((x for x in metadata if isinstance(x, type_class)), None)


# for field_name, field_info in Model.model_fields.items():
#     print("field_name", field_name)
#     print("field_type", field_info.annotation)
#     print("field_info", field_info)
#     print()
#     print(type(field_info.annotation))
#     match field_info:
#         case FieldInfo(metadata=metadata):
#             max_len_obj = fetch_type(metadata, MaxLen)
#             min_len_obj = fetch_type(metadata, MinLen)
#             max_len = max_len_obj.max_length if max_len_obj else None
#             min_len = min_len_obj.min_length if min_len_obj else None
#             if max_len is not None and min_len is not None and min_len == max_len:
#                 print("dim type")
#         case _:
#             print("not annotations")


name, optional_array = list(Model.model_fields.items())[7]
# print("optional_array", optional_array)
# print(
#     "optional_array_type",
# )
# print("annotation", optional_array.annotation)
print("field_name", name)
field_type = optional_array.annotation
origin = get_origin(field_type)
if origin is Union or origin is UnionType:
    args = get_args(field_type)
    if len(args) == 2 and type(None) in args:
        x = True
    else:
        raise ValueError(
            "Union's are not supported for modaic schemas. Except for single unions with None (Optional type)"
        )
else:
    x = False
print("origin", origin)
print("x", x)


# _, list_field = list(Model.model_fields.items())[2]
# print("list_field", list_field)
# print(isinstance(list_field, Sequence))
