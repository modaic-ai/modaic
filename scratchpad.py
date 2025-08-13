# from pydantic import BaseModel
from modaic.types import Array, String, int8, pydantic_to_modaic_schema

from modaic.context import Source, SourceType, ContextSchema
from typing import Annotated, Optional, Union
from pydantic import Field
from collections.abc import Mapping
from typing import ClassVar


class CaptionedImageSchema(ContextSchema):
    caption: str
    caption_embedding: list[int]
    image_path: str


print(CaptionedImageSchema.caption)
