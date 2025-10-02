from typing import Optional, Union

import pytest
from pydantic import ValidationError

from modaic.context.base import Context
from modaic.exceptions import SchemaError
from modaic.types import Array, InnerField, Schema, SchemaField, String, double, int8, int16, int32, int64


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


class Employee(Context):
    name: String[50]
    age: int16
    salary: double
    active: bool
    skills: Array[str, 5]
    performance_score: float
    notes: Optional[String[200]] = None


class Company(Context):
    name: String[100]
    employee_ids: Array[str, 100]
    revenue: double
    priority: int32
    small_flag: int8


class ContextWithUnion(Context):
    union_field: Union[int, str]


class ContextWithComplexDict(Context):
    dict_field: dict[str, Union[int, str]]


class ContextWithComplexList(Context):
    list_field: list[Union[int, str]]


class ContextWithComplexTuple(Context):
    tuple_field: tuple[Union[int, str], ...]


class ContextWithListOfDicts(Context):
    list_of_dicts_field: list[dict[str, Union[int, str]]]


def test_context_schemas():
    """
    Test schema generation and field properties for Email, Employee, and Company Contexts.

    Params:
        None
    """
    email_schema = Email.schema().as_dict()
    assert email_schema["subject"].type == "string" and email_schema["subject"].size == 100
    assert email_schema["recipients"].type == "array" and email_schema["recipients"].inner_type.type == "string"
    assert email_schema["optional_summary"].optional is True
    assert email_schema["optional_recipients"].optional is True

    employee_schema = Employee.schema().as_dict()
    assert employee_schema["id"].is_id is True and employee_schema["id"].type == "string"
    assert employee_schema["name"].type == "string" and employee_schema["name"].size == 50
    assert employee_schema["age"].type == "integer"
    assert employee_schema["salary"].type == "number" and employee_schema["salary"].format == "double"
    assert employee_schema["active"].type == "boolean"
    assert employee_schema["skills"].type == "array" and employee_schema["skills"].inner_type.type == "string"

    company_schema = Company.schema().as_dict()
    assert company_schema["id"].is_unique is True
    assert company_schema["name"].type == "string" and company_schema["name"].size == 100
    assert company_schema["employee_ids"].type == "array"
    assert company_schema["revenue"].format == "double"
    assert company_schema["priority"].type == "integer"
    assert company_schema["small_flag"].type == "integer"


def test_schema_from_json_schema_roundtrip():
    # Build from a model json schema and then to Schema
    schema = Email.model_json_schema()
    modaic_schema = Schema.from_json_schema(schema)
    as_dict = modaic_schema.as_dict()
    assert "subject" in as_dict and as_dict["subject"].type == "string"


def test_type_validations():
    """
    Test validation for Array, String, int8, int16, int32, int64, float, and double types.

    This test covers:
    - Array length constraints
    - String max length constraints
    - Integer type bounds (int8, int16, int32, int64)
    - Float and double value validation

    Params:
        None
    """

    # Array validation
    class Contacts(Context):
        emails: Array[str, 2]

    Contacts(emails=["a@example.com", "b@example.com"])
    with pytest.raises(ValidationError):
        Contacts(emails=["a@example.com", "b@example.com", "c@example.com"])

    # String validation
    class Note(Context):
        title: String[5]

    Note(title="hello")
    with pytest.raises(ValidationError):
        Note(title="toolong")

    # int8 validation
    class Small(Context):
        value: int8

    Small(value=127)
    Small(value=-128)
    with pytest.raises(ValidationError):
        Small(value=128)
    with pytest.raises(ValidationError):
        Small(value=-129)

    # int16 validation
    class Mid(Context):
        value: int16

    Mid(value=32767)
    Mid(value=-32768)
    with pytest.raises(ValidationError):
        Mid(value=32768)
    with pytest.raises(ValidationError):
        Mid(value=-32769)

    # int32 validation
    class I32(Context):
        value: int32

    I32(value=2**31 - 1)
    I32(value=-(2**31))
    with pytest.raises(ValidationError):
        I32(value=2**31)
    with pytest.raises(ValidationError):
        I32(value=-(2**31) - 1)

    # int64 validation
    class I64(Context):
        value: int64

    I64(value=2**63 - 1)
    I64(value=-(2**63))
    with pytest.raises(ValidationError):
        I64(value=2**63)
    with pytest.raises(ValidationError):
        I64(value=-(2**63) - 1)

    # float validation
    class Score(Context):
        value: float

    Score(value=3.14)
    with pytest.raises(ValidationError):
        Score(value="not-a-float")

    # double validation
    class Precise(Context):
        value: double

    Precise(value=1.23)


def test_schema_field_and_constraints():
    """
    Test manual schema creation, field constraints, and format defaults.

    Params:
        None
    """
    # Invalid: is_id can only be True for key 'id'
    with pytest.raises(SchemaError):
        Schema(
            {
                "name": SchemaField(
                    type="string",
                    format=None,
                    size=10,
                    optional=False,
                    inner_type=None,
                    is_id=True,
                )
            }
        )

    # Valid: proper id field and a string field
    s = Schema(
        {
            "id": SchemaField(
                type="string",
                format=None,
                size=None,
                optional=False,
                inner_type=None,
                is_id=True,
                is_unique=True,
            ),
            "name": SchemaField(type="string", format=None, size=10, optional=False, inner_type=None),
        }
    )
    d = s.as_dict()
    assert d["id"].is_id is True and d["name"].type == "string"

    # Invalid array without inner type
    with pytest.raises(SchemaError):
        Schema({"arr": SchemaField(type="array", format=None, size=3, optional=False, inner_type=None)})

    # Ensure number format defaults to float when not annotated
    assert SchemaField(type="number", format=None, optional=False, size=None, inner_type=None).format == "float"
    assert InnerField(type="number", format=None, size=None).format == "float"

    # Array inner type required
    with pytest.raises(SchemaError):
        SchemaField(type="array", format=None, size=3, optional=False, inner_type=None)

    # String max length propagates to size
    sch = Email.schema().as_dict()
    assert sch["subject"].size == 100


def test_unions_and_complex_types():
    # Top-level union not allowed
    with pytest.raises(SchemaError):
        Schema.from_json_schema(ContextWithUnion.model_json_schema())

    # Dict with union values should not error
    Schema.from_json_schema(ContextWithComplexDict.model_json_schema())

    # List with union elements should error
    with pytest.raises(SchemaError):
        Schema.from_json_schema(ContextWithComplexList.model_json_schema())

    # Tuple with union elements should error
    with pytest.raises(SchemaError):
        Schema.from_json_schema(ContextWithComplexTuple.model_json_schema())

    # List of dicts should error
    with pytest.raises(SchemaError):
        Schema.from_json_schema(ContextWithListOfDicts.model_json_schema())


def test_manual_schema_creation():
    """
    Build Schema objects by hand and ensure correct validation behavior.

    Covers:
    - Valid minimal schema (id + one field)
    - Invalid is_id usage on non-id key
    - Array field requires inner_type
    - Number format defaults to float
    """

    # Valid minimal schema
    minimal = Schema(
        {
            "id": SchemaField(
                type="string",
                format=None,
                size=None,
                optional=False,
                inner_type=None,
                is_id=True,
                is_unique=True,
            ),
            "title": SchemaField(type="string", format=None, size=32, optional=False, inner_type=None),
        }
    )
    d = minimal.as_dict()
    assert d["id"].is_id is True and d["title"].type == "string" and d["title"].size == 32

    # Invalid: is_id on non-id key
    with pytest.raises(SchemaError):
        Schema(
            {
                "name": SchemaField(
                    type="string",
                    format=None,
                    size=10,
                    optional=False,
                    inner_type=None,
                    is_id=True,
                )
            }
        )

    # Invalid: array without inner_type
    with pytest.raises(SchemaError):
        Schema({"arr": SchemaField(type="array", format=None, size=3, optional=False, inner_type=None)})

    # Number format defaults to float
    assert SchemaField(type="number", format=None, optional=False, size=None, inner_type=None).format == "float"
