from typing import Optional, get_origin, get_args, Union
from pydantic import BaseModel
from modaic.types import (
    Vector,
    Float16Vector,
    Array,
    String,
    pydantic_model_to_schema,
)
from modaic.databases.integrations.milvus import _convert_scontext_to_milvus_schema


x = Array[int, 10]
print(x)
