from modaic.types import Array, int8, Optional, String
from pydantic import BaseModel, Field
from modaic.context import Context
from typing import Dict, Any, List


class CustomContext(Context):
    custom_field: dict


class Test(BaseModel):
    # test: Array[int, 10] = []
    # name: String[100] = "test"
    # test2: Optional[int8] = None
    # test3: Optional[CustomContext] = None
    # id: dict
    # f: float
    # x: int8
    b: bool


print(Test.model_json_schema())
