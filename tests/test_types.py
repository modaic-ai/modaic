from typing import Optional

import pytest

from modaic.context.base import Context
from modaic.types import Array, String, pydantic_to_modaic_schema


class Email(Context):
    subject: String[100]
    content: str
    recipients: Array[str, 10]
    tags: list[str]
    priority: int
    score: float
    pinned: bool
    optional_summary: Optional[String[50]]
    optional_recipients: Optional[Array[int, 5]]


def test_pydantic_to_modaic_schema_types() -> None:
    """Validate that fields are unpacked into expected modaic schema entries."""
    modaic_schema = pydantic_to_modaic_schema(Email)

    assert isinstance(modaic_schema, dict)

    # Iterate like downstream code would and validate expected shapes
    for field_name, field_info in modaic_schema.items():
        if field_name == "subject":
            assert field_info.type == "String"
            assert field_info.size == 100
            assert field_info.optional is False
        elif field_name == "content":
            assert field_info.type == "String"
            assert field_info.size is None
            assert field_info.optional is False
        elif field_name == "recipients":
            assert field_info.type == "Array"
            assert field_info.size == 10
            assert field_info.inner_type is not None
            assert isinstance(field_info.inner_type, dict)
            assert field_info.inner_type.get("type") == "String"
            assert field_info.inner_type.get("size") is None
            assert field_info.optional is False
        elif field_name == "tags":
            assert field_info.type == "Array"
            assert field_info.size is None
            assert field_info.inner_type is not None
            assert isinstance(field_info.inner_type, dict)
            assert field_info.inner_type.get("type") == "String"
            assert field_info.inner_type.get("size") is None
            assert field_info.optional is False
        elif field_name == "priority":
            assert field_info.type == "int64"
            assert field_info.size is None
            assert field_info.optional is False
        elif field_name == "score":
            assert field_info.type == "float64"
            assert field_info.size is None
            assert field_info.optional is False
        elif field_name == "pinned":
            assert field_info.type == "bool"
            assert field_info.size is None
            assert field_info.optional is False
        elif field_name == "optional_summary":
            assert field_info.type == "String"
            assert field_info.size == 50
            assert field_info.optional is True
        elif field_name == "optional_recipients":
            assert field_info.type == "Array"
            assert field_info.size == 5
            assert field_info.inner_type is not None
            assert isinstance(field_info.inner_type, dict)
            assert field_info.inner_type.get("type") == "int64"
            assert field_info.inner_type.get("size") is None
            assert field_info.optional is True
        elif field_name == "id":
            assert field_info.type == "String"
            assert field_info.size is None
        elif field_name == "source":
            assert field_info.type == "Mapping"
            assert field_info.size is None
        elif field_name == "metadata":
            assert field_info.type == "Mapping"
            assert field_info.size is None
        else:
            pytest.fail(f"Unexpected field in schema: {field_name}")
