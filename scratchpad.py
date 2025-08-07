from typing import Optional, get_origin, get_args, Union, List, Sequence, Type
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


class Model(BaseModel):
    x: Array[int, 10]
    y: Optional[Array[int, 10]]
    z: list
    w: str
    v: int32
    s: String[10]
    a: Vector[int, 10]


_, required_array = list(Model.model_fields.items())[0]
print("required_array", required_array)
print(isinstance(required_array, Sequence))
print("metadata type", type(required_array))


def fetch_type(metadata: list, type_class: Type) -> Optional[Type]:
    return next((x for x in metadata if isinstance(x, type_class)), None)


for field_name, field_info in Model.model_fields.items():
    print("field_name", field_name)
    print("field_type", field_info.annotation)
    print("field_info", field_info)
    print()
    print(type(field_info.annotation))
    match field_info:
        case FieldInfo(metadata=metadata):
            max_len_obj = fetch_type(metadata, MaxLen)
            min_len_obj = fetch_type(metadata, MinLen)
            max_len = max 
        case _:
            


# _, optional_array = list(Model.model_fields.items())[1]
# print("optional_array", optional_array)
# print(
#     "optional_array_type",
# )


# _, list_field = list(Model.model_fields.items())[2]
# print("list_field", list_field)
# print(isinstance(list_field, Sequence))
