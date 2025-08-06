from typing import Optional, Union
from pydantic import BaseModel, Field, model_validator
import weakref
from modaic.context import (
    Source,
    SourceType,
    Context,
    Text,
    SerializedContext,
    SerializedText,
)
from modaic.databases import ContextDatabase, SQLDatabase
import json
from typing import ClassVar


tx = Text(text="test", source=Source(origin="db.csv", type=SourceType.LOCAL_PATH))
stx = tx.serialize()
print(stx.model_dump_json())

print(stx.model_dump())
print("Text", Text.deserialize(stx))
print("Text", Text.deserialize(stx.model_dump()))

print("--------------------------------")
print("text", Text.deserialize(stx.model_dump()).text)
# print("Text", Text.deserialize(stx.model_dump_json()))
