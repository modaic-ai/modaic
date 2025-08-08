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
    int16,
    pydantic_model_to_schema,
    unpack_type,
)

# from modaic.databases.integrations.milvus import _convert_scontext_to_milvus_schema
from annotated_types import Gt, Le, MinLen, MaxLen


class Model(BaseModel):
    x: Array[String[10], 10]
    y: Optional[Array[int, 10]]
    z: list[int]
    w: str
    v: int16
    s: String[10]
    a: Vector[10]
    test: Union[int, None]
    workpls: int | None


schema = pydantic_model_to_schema(Model)

for field, field_info in schema.items():
    print(field, field_info)

# unpack_type(Array[String[10], 10])
# unpack_type(Vector[10])
