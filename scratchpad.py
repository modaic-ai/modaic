from pydantic import BaseModel
from modaic.types import Array, String, Vector, int16, pydantic_model_to_schema


class Model(BaseModel):
    arr: Array[int, 10]
    x: Array[String[10], 10]
    # y: Optional[Array[int, 10]]
    z: list[int]
    w: str
    v: int16
    s: String[10]
    a: Vector[10]
    # test: Union[int, None]
    # workpls: int | None


schema = pydantic_model_to_schema(Model)
print("########################\n\n")
for field, field_info in schema.items():
    print(field, field_info)

# unpack_type(Array[String[10], 10])
# unpack_type(Vector[10])
# print("type of Array[String[10], 10]:", type(Array[String[10], 10]))
# print("Array[String[10], 10]:", Array[String[10], 10])
