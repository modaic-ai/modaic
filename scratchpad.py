from pydantic import BaseModel
from modaic.types import Array, String, Vector, int16, pydantic_to_modaic_schema
from modaic.context import Source, SourceType, SerializedContext
from typing import Annotated, Optional, Union
from pydantic import Field
from collections.abc import Mapping
from typing import ClassVar


# print(Bad)


class SerializedModel(SerializedContext):
    context_class: ClassVar[str] = "Model"
    arr: Array[int, 10]
    x: Union[int, None]


new_model = SerializedModel(
    arr=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    x=1,
    metadata={"a": 1, "b": {"c": 2, "d": [3, 4, 5]}},
    source=Source(
        origin="https://www.google.com",
        type=SourceType.URL,
        metadata={"GET": 1, "POST": {"url": 4, "headers": [5, 6, 7]}},
    ),
)
# print(new_model)
json = new_model.model_dump(mode="json")
print(type(json))
print(json)

# schema = pydantic_to_modaic_schema(Model)
# print("########################\n\n")
# for field, field_info in schema.items():
#     print(field, field_info)


# unpack_type(Array[String[10], 10])
# unpack_type(Vector[10])
# print("type of Array[String[10], 10]:", type(Array[String[10], 10]))
# print("Array[String[10], 10]:", Array[String[10], 10])
# for field_name, field_info in Model.model_fields.items():
#     print(field_name, field_info)
#     print("is BaseModel:", field_info.annotation is BaseModel)
#     print("is Source:", field_info.annotation is Source)
#     print("is Mapping:", field_info.annotation is Mapping)
#     print("is dict:", field_info.annotation is dict)
#     print("is subclass of BaseModel:", issubclass(field_info.annotation, BaseModel))
#     print("is subclass of Mapping:", issubclass(field_info.annotation, Mapping))
#     print("is subclass of dict:", issubclass(field_info.annotation, dict))
