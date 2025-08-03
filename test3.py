from modaic.context import Table, Context, SerializedContext, Atomic
import pandas as pd
from pydantic import BaseModel


class Test(Atomic):
    def __init__(self, a: int, b: str, c: str, d: list):
        self.a = a
        self.b = b
        self._c = c


s = SerializedContext(context_class=Test, a=0, b=1, c=2).schema()
print(s)
print(issubclass(s, BaseModel))
print(s.model_json_schema())
