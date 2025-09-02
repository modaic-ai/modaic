from modaic.context import Text, Context
from gqlalchemy import Node, Memgraph
from pydantic import BaseModel, Field
from pydantic.v1 import Field as V1Field
from typing import (
    Any,
    get_origin,
    get_args,
    Annotated,
    Union,
    Literal,
    Final,
    ClassVar,
    Type,
    List,
)
from types import UnionType
from modaic.context.base import get_annotations, get_defaults

# config = MemgraphConfig()

# db = GraphDatabase(config)
db = Memgraph()


class SpecialText(Text):
    count: int = 1


class EvenMoreSpecialText(SpecialText):
    x: List[int] = Field(default_factory=lambda: [1, 2, 3])


text = Text(text="test")
text_node = text.to_gqlalchemy(db)
text_node.save(db)
print("TEXT", text_node)

special_text = SpecialText(text="test", count=2)
special_text_node = special_text.to_gqlalchemy(db)
special_text_node.save(db)
print("SPECIAL TEXT", special_text_node)

even_more_special_text = EvenMoreSpecialText(text="test")
even_more_special_text_node = even_more_special_text.to_gqlalchemy(db)
even_more_special_text_node.save(db)
print("EVEN MORE SPECIAL TEXT", even_more_special_text_node)


text2 = Text(text="test2")
text2_node = text2.to_gqlalchemy(db)
text2_node.save(db)
print("TEXT2", text2_node)
