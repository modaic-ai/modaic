# from pydantic import BaseModel
from modaic.types import Array, String, int8, pydantic_to_modaic_schema

from modaic.context import Source, SourceType, ContextSchema
from typing import Annotated, Optional, Union
from pydantic import Field, BaseModel
from collections.abc import Mapping
from typing import ClassVar


class CaptionedImageSchema(ContextSchema):
    caption: str
    caption_embedding: list[int]
    image_path: str


# query = (
#     (CaptionedImageSchema.caption == "hello") & CaptionedImageSchema.caption_embedding
#     == [1, 2, 3] & CaptionedImageSchema.image_path
#     == "path/to/image.jpg"
# )

# print(query)

# query = (
#     CaptionedImageSchema.caption
#     == "hello" & CaptionedImageSchema.caption_embedding
#     == [1, 2, 3]
# )
# query = (CaptionedImageSchema.caption == "hello") & (
#     CaptionedImageSchema.caption_embedding == [1, 2, 3]
# )

x = "hello" | CaptionedImageSchema.caption_embedding
print("x:", x)
y = CaptionedImageSchema.caption == x
print("y:", y)
z = y == [1, 2, 3]
print("z:", z)


# print(query)


# query2 = (
#     CaptionedImageSchema.caption
#     == "hello" | CaptionedImageSchema.caption_embedding
#     == [1, 2, 3]
# )

# print(query2)
