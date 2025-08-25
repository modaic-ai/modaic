# from pydantic import BaseModel, ConfigDict
# from typing import Any

# # from pydantic._internal._model_construction import ModelMetaclass
# from modaic.context import ContextSchema, Relation
from gqlalchemy.models import Relationship


r = Relationship(_start_node=1, _end_node=2, _type="TEST")


class ContainedBy(Relationship, type="CONTAINED_BY"):
    x: str
    y: int


r2 = ContainedBy(x="test", y=1)

print(r2._properties)
print(r2._type)
print(r2._start_node_id)
print(r2._end_node_id)
