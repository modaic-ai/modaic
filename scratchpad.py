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
    waddup: list[int]


# query = (CaptionedImageSchema.caption == "hello") & (
#     CaptionedImageSchema.caption_embedding == [1, 2, 3]
# ) & (CaptionedImageSchema.image_path == CaptionedImageSchema.caption) | (
#     CaptionedImageSchema.waddup == [1, 2, 3]
# ) & (CaptionedImageSchema.metadata["table"] == "hello")

query = (CaptionedImageSchema.metadata["table"] == "hello") & (
    CaptionedImageSchema.caption_embedding == [1, 2, 3]
)

print(query)

# x = "hello" in CaptionedImageSchema.caption

# query = (
#     CaptionedImageSchema.caption
#     == "hello" & CaptionedImageSchema.caption_embedding
#     == [1, 2, 3]
# )
# # query = (CaptionedImageSchema.caption == "hello") & (
# #     CaptionedImageSchema.caption_embedding == [1, 2, 3]
# # )

# x = "hello" | CaptionedImageSchema.caption_embedding
# print("x:", x)
# y = CaptionedImageSchema.caption == x
# print("y:", y)
# z = y == [1, 2, 3]
# print("z:", z)


# print(query)


# query2 = (
#     CaptionedImageSchema.caption
#     == "hello" | CaptionedImageSchema.caption_embedding
#     == [1, 2, 3]
# )

# print(query2)
