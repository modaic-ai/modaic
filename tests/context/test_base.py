from typing import Any

import pytest
from pydantic import SerializationInfo, SerializerFunctionWrapHandler, model_serializer

from modaic.context import Context, Text
from modaic.context.base import Hydratable, is_hydratable
from modaic.context.table import Table, TableFile
from modaic.types import Array, Field, Optional, String


class User(Context):
    name: str
    api_key: str = Field(hidden=True)


class CustomContextBase(Context):
    """
    Custom context for Milvus tests, covering all supported types and Optionals.
    """

    field1: str
    field2: int
    field3: bool
    field4: float
    field5: list[str]
    field6: dict[str, int]
    field7: Array[int, 10]
    field8: String[50]
    field9: Text
    field10: Optional[Array[String[50], 10]] = None
    field11: Optional[Array[int, 10]] = None
    field12: Optional[String[50]] = None


def test_schema_creation_returns_simplified_schema():
    sch = User.schema().as_dict()
    assert "id" in sch and sch["id"].type == "string"
    assert sch["name"].type == "string"


def test_chunk_with_and_apply_to_chunks():
    from modaic.context import Text

    t = Text(text="alpha beta gamma")
    t.chunk_text(lambda s: s.split())
    assert [c.text for c in t.chunks] == ["alpha", "beta", "gamma"]
    t.apply_to_chunks(lambda c: c.metadata.update({"len": len(c.text)}))
    assert [c.metadata["len"] for c in t.chunks] == [5, 4, 5]


def test_is_hydratable_protocol_and_helper():
    # TableFile implements hydration
    from modaic.storage import InPlaceFileStore

    store = InPlaceFileStore("tests/artifacts/test_dir_excel")
    tf = TableFile.from_file_store("1st_New_Zealand_Parliament_0.xlsx", store)
    # Protocol check (method presence) may vary; helper ensures Context+protocol
    assert is_hydratable(tf) is True
    assert isinstance(tf, Hydratable) is True


def test_table_is_embeddable_via_markdown():
    import pandas as pd

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    t = Table(df=df, name="numbers")
    md = t.embedme()
    assert isinstance(md, str)
    assert "Table name: numbers" in md


class AnotherContext(Context):
    """
    Another context class for comprehensive equality testing with different field types.
    """

    name: str
    age: int
    is_active: bool
    score: float
    tags: list[str]
    config: dict[str, Any]
    data: Array[int, 5]
    description: String[100]
    content: Text
    optional_field: Optional[str] = None
    optional_array: Optional[Array[String[20], 3]] = None


def test_eq_check():
    """
    Test basic equality checking with CustomContextBase instances.
    """
    # Test same instance
    ctx1 = CustomContextBase(
        field1="test",
        field2=42,
        field3=True,
        field4=3.14,
        field5=["a", "b", "c"],
        field6={"x": 1, "y": 2},
        field7=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        field8="test_string",
        field9=Text(text="sample text"),
        field10=["opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7", "opt8", "opt9", "opt10"],
        field11=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        field12="optional_string",
    )

    # Same instance should be equal to itself
    assert ctx1 == ctx1

    # Test copied instance (preserves ID)
    ctx2 = ctx1.model_copy()
    assert ctx1 == ctx2

    # Test contexts with same explicit ID
    # Create a shared Text object to ensure nested objects have the same ID
    shared_text = Text(text="sample text")

    ctx3 = CustomContextBase(
        id="same-id",
        field1="test",
        field2=42,
        field3=True,
        field4=3.14,
        field5=["a", "b", "c"],
        field6={"x": 1, "y": 2},
        field7=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        field8="test_string",
        field9=shared_text,
        field10=["opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7", "opt8", "opt9", "opt10"],
        field11=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        field12="optional_string",
    )

    ctx4 = CustomContextBase(
        id="same-id",
        field1="test",
        field2=42,
        field3=True,
        field4=3.14,
        field5=["a", "b", "c"],
        field6={"x": 1, "y": 2},
        field7=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        field8="test_string",
        field9=shared_text,
        field10=["opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7", "opt8", "opt9", "opt10"],
        field11=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        field12="optional_string",
    )

    # Contexts with same ID should be equal
    assert ctx3 == ctx4

    # Test different field values (same ID)
    ctx5 = CustomContextBase(
        id="same-id",
        field1="different",
        field2=42,
        field3=True,
        field4=3.14,
        field5=["a", "b", "c"],
        field6={"x": 1, "y": 2},
        field7=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        field8="test_string",
        field9=shared_text,
        field10=["opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7", "opt8", "opt9", "opt10"],
        field11=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        field12="optional_string",
    )

    assert ctx3 != ctx5

    # Test different IDs (same field values)
    ctx6 = CustomContextBase(
        id="different-id",
        field1="test",
        field2=42,
        field3=True,
        field4=3.14,
        field5=["a", "b", "c"],
        field6={"x": 1, "y": 2},
        field7=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        field8="test_string",
        field9=shared_text,
        field10=["opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7", "opt8", "opt9", "opt10"],
        field11=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        field12="optional_string",
    )

    assert ctx3 != ctx6

    # Test with None optional fields (same ID)
    ctx7 = CustomContextBase(
        id="optional-test-id",
        field1="test",
        field2=42,
        field3=True,
        field4=3.14,
        field5=["a", "b", "c"],
        field6={"x": 1, "y": 2},
        field7=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        field8="test_string",
        field9=shared_text,
        field10=["opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7", "opt8", "opt9", "opt10"],
        field11=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        field12=None,
    )

    ctx8 = CustomContextBase(
        id="optional-test-id",
        field1="test",
        field2=42,
        field3=True,
        field4=3.14,
        field5=["a", "b", "c"],
        field6={"x": 1, "y": 2},
        field7=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        field8="test_string",
        field9=shared_text,
        field10=["opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7", "opt8", "opt9", "opt10"],
        field11=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        field12=None,
    )

    assert ctx7 == ctx8


def test_eq_check_hard():
    """
    Comprehensive equality testing covering key edge cases.
    """
    # Create shared Text object to ensure nested objects have the same ID
    shared_text = Text(text="Sample content")

    # Test 1: Same class, identical values with same ID
    ctx1 = AnotherContext(
        id="test-id-1",
        name="Alice",
        age=30,
        is_active=True,
        score=95.5,
        tags=["tag1", "tag2"],
        config={"key1": "value1", "key2": 42},
        data=[1, 2, 3, 4, 5],
        description="A test description",
        content=shared_text,
        optional_field="optional_value",
        optional_array=["a", "b", "c"],
    )

    ctx2 = AnotherContext(
        id="test-id-1",
        name="Alice",
        age=30,
        is_active=True,
        score=95.5,
        tags=["tag1", "tag2"],
        config={"key1": "value1", "key2": 42},
        data=[1, 2, 3, 4, 5],
        description="A test description",
        content=shared_text,
        optional_field="optional_value",
        optional_array=["a", "b", "c"],
    )

    assert ctx1 == ctx2

    # Test 2: Different classes should not be equal (even with same ID)
    custom_ctx = CustomContextBase(
        id="test-id-1",
        field1="Alice",
        field2=30,
        field3=True,
        field4=95.5,
        field5=["tag1", "tag2"],
        field6={"key1": 1, "key2": 42},
        field7=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        field8="A test description",
        field9=shared_text,
    )

    assert ctx1 != custom_ctx

    # Test 3: Different field values (same ID)
    ctx3 = AnotherContext(
        id="test-id-1",
        name="Bob",  # Different name
        age=30,
        is_active=True,
        score=95.5,
        tags=["tag1", "tag2"],
        config={"key1": "value1", "key2": 42},
        data=[1, 2, 3, 4, 5],
        description="A test description",
        content=shared_text,
        optional_field="optional_value",
        optional_array=["a", "b", "c"],
    )

    assert ctx1 != ctx3

    # Test 4: Different array values (same ID)
    ctx4 = AnotherContext(
        id="test-id-1",
        name="Alice",
        age=30,
        is_active=True,
        score=95.5,
        tags=["tag1", "tag2"],
        config={"key1": "value1", "key2": 42},
        data=[1, 2, 3, 4, 6],  # Different last element
        description="A test description",
        content=shared_text,
        optional_field="optional_value",
        optional_array=["a", "b", "c"],
    )

    assert ctx1 != ctx4

    # Test 5: Different Text content (same ID)
    different_text = Text(text="Different content")
    ctx5 = AnotherContext(
        id="test-id-1",
        name="Alice",
        age=30,
        is_active=True,
        score=95.5,
        tags=["tag1", "tag2"],
        config={"key1": "value1", "key2": 42},
        data=[1, 2, 3, 4, 5],
        description="A test description",
        content=different_text,  # Different text
        optional_field="optional_value",
        optional_array=["a", "b", "c"],
    )

    assert ctx1 != ctx5

    # Test 6: Different optional fields (same ID)
    ctx6 = AnotherContext(
        id="test-id-1",
        name="Alice",
        age=30,
        is_active=True,
        score=95.5,
        tags=["tag1", "tag2"],
        config={"key1": "value1", "key2": 42},
        data=[1, 2, 3, 4, 5],
        description="A test description",
        content=shared_text,
        optional_field="different_optional",  # Different optional field
        optional_array=["a", "b", "c"],
    )

    assert ctx1 != ctx6

    # Test 7: Different IDs (same field values)
    ctx7 = AnotherContext(
        id="different-id",
        name="Alice",
        age=30,
        is_active=True,
        score=95.5,
        tags=["tag1", "tag2"],
        config={"key1": "value1", "key2": 42},
        data=[1, 2, 3, 4, 5],
        description="A test description",
        content=shared_text,
        optional_field="optional_value",
        optional_array=["a", "b", "c"],
    )

    assert ctx1 != ctx7

    # Test 8: Both with None optional fields (same ID)
    ctx8 = AnotherContext(
        id="optional-test-id",
        name="Alice",
        age=30,
        is_active=True,
        score=95.5,
        tags=["tag1", "tag2"],
        config={"key1": "value1", "key2": 42},
        data=[1, 2, 3, 4, 5],
        description="A test description",
        content=shared_text,
        # optional_field and optional_array are None by default
    )

    ctx9 = AnotherContext(
        id="optional-test-id",
        name="Alice",
        age=30,
        is_active=True,
        score=95.5,
        tags=["tag1", "tag2"],
        config={"key1": "value1", "key2": 42},
        data=[1, 2, 3, 4, 5],
        description="A test description",
        content=shared_text,
    )

    assert ctx8 == ctx9


class InnerContext(Context):
    name: str
    age: int
    password: str = Field(hidden=True)


class SingleNestedContext(Context):
    link: str
    private: str = Field(hidden=True)
    inner_context: InnerContext


class DoubleNestedContext(Context):
    company: str
    num_employees: int
    key: str = Field(hidden=True)
    single_nested_context: SingleNestedContext


def test_eq_check_double_nested():
    pass


@pytest.fixture(params=[True, False])
def include_hidden(request: pytest.FixtureRequest) -> bool:
    return request.param


HIDDEN_BASE_FIELDS = ["id", "parent", "metadata"]


def test_dump(include_hidden: bool):
    """
    Test dumping works as expected for single nested contexts with and without model_dump(include_hidden=True)
    """
    i = InnerContext(name="John", age=30, password="this should be hidden")
    dump = i.model_dump(include_hidden=include_hidden)

    assert dump["name"] == "John"
    assert dump["age"] == 30
    if include_hidden:
        assert all(field in dump for field in HIDDEN_BASE_FIELDS)
        assert dump["password"] == "this should be hidden"
    else:
        assert not any(field in dump for field in HIDDEN_BASE_FIELDS)
        assert "password" not in dump


def test_dump_single_nested(include_hidden: bool):
    """
    Test dumping works as expected for single nested contexts with and without model_dump(include_hidden=True)
    """
    i = InnerContext(name="John", age=30, password="this should be hidden")
    s = SingleNestedContext(link="https://www.google.com", private="this is private", inner_context=i)
    dump = s.model_dump(include_hidden=include_hidden)
    assert dump["link"] == "https://www.google.com"
    assert dump["inner_context"]["name"] == "John"
    assert dump["inner_context"]["age"] == 30
    if include_hidden:
        assert all(field in dump for field in HIDDEN_BASE_FIELDS)
        assert all(field in dump["inner_context"] for field in HIDDEN_BASE_FIELDS)
        assert dump["private"] == "this is private"
        assert dump["inner_context"]["password"] == "this should be hidden"
    else:
        assert not any(field in dump for field in HIDDEN_BASE_FIELDS)
        assert not any(field in dump["inner_context"] for field in HIDDEN_BASE_FIELDS)
        assert "private" not in dump
        assert "password" not in dump["inner_context"]


def test_dump_double_nested(include_hidden: bool):
    """
    Test dumping works as expected for double nested contexts with and without model_dump(include_hidden=True)
    """
    i = InnerContext(name="John", age=30, password="this should be hidden")
    s = SingleNestedContext(link="https://www.google.com", private="this is private", inner_context=i)
    d = DoubleNestedContext(company="Google", num_employees=100, key="this is hidden", single_nested_context=s)
    dump = d.model_dump(include_hidden=include_hidden)

    assert dump["company"] == "Google"
    assert dump["num_employees"] == 100
    assert dump["single_nested_context"]["link"] == "https://www.google.com"
    assert dump["single_nested_context"]["inner_context"]["name"] == "John"
    assert dump["single_nested_context"]["inner_context"]["age"] == 30
    if include_hidden:
        assert all(field in dump for field in HIDDEN_BASE_FIELDS)
        assert all(field in dump["single_nested_context"] for field in HIDDEN_BASE_FIELDS)
        assert all(field in dump["single_nested_context"]["inner_context"] for field in HIDDEN_BASE_FIELDS)
        assert dump["key"] == "this is hidden"
        assert dump["single_nested_context"]["private"] == "this is private"
        assert dump["single_nested_context"]["inner_context"]["password"] == "this should be hidden"
    else:
        assert not any(field in dump for field in HIDDEN_BASE_FIELDS)
        assert not any(field in dump["single_nested_context"] for field in HIDDEN_BASE_FIELDS)
        assert not any(field in dump["single_nested_context"]["inner_context"] for field in HIDDEN_BASE_FIELDS)
        assert "key" not in dump
        assert "private" not in dump["single_nested_context"]
        assert "password" not in dump["single_nested_context"]["inner_context"]


class WrapInnerContext(Context):
    name: str
    age: int
    password: str = Field(hidden=True)

    @model_serializer(mode="wrap")
    def ser_model(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> dict[str, Any]:
        dump = handler(self)
        dump["name"] = "custom_name_value"
        return dump


class WrapSingleNestedContext(Context):
    link: str
    private: str = Field(hidden=True)
    inner_context: WrapInnerContext


class WrapDoubleNestedContext(Context):
    company: str
    num_employees: int
    key: str = Field(hidden=True)
    single_nested_context: WrapSingleNestedContext


def test_dump_custom_wrap_serializer(include_hidden: bool):
    """
    Test dumping works as expected for custom serializer with and without model_dump(include_hidden=True)
    """
    i = WrapInnerContext(name="John", age=30, password="this should be hidden")
    dump = i.model_dump(include_hidden=include_hidden)

    assert all(field in dump for field in HIDDEN_BASE_FIELDS)
    assert dump["name"] == "custom_name_value"
    assert dump["age"] == 30
    assert dump["password"] == "this should be hidden"


def test_dump_custom_wrap_serializer_single_nested(include_hidden: bool):
    """
    Test dumping works as expected for custom serializer and single nested contexts with and without model_dump(include_hidden=True)
    """
    i = WrapInnerContext(name="John", age=30, password="this should be hidden")
    s = WrapSingleNestedContext(link="https://www.google.com", private="this is private", inner_context=i)
    dump = s.model_dump(include_hidden=include_hidden)

    assert dump["link"] == "https://www.google.com"
    assert all(field in dump["inner_context"] for field in HIDDEN_BASE_FIELDS)
    assert dump["inner_context"]["name"] == "custom_name_value"
    assert dump["inner_context"]["age"] == 30
    assert dump["inner_context"]["password"] == "this should be hidden"

    if include_hidden:
        assert all(field in dump for field in HIDDEN_BASE_FIELDS)
        assert dump["private"] == "this is private"
    else:
        assert not any(field in dump for field in HIDDEN_BASE_FIELDS)
        assert "private" not in dump


def test_dump_custom_wrap_serializer_double_nested(include_hidden: bool):
    """
    Test dumping works as expected for custom serializer and double nested contexts with and without model_dump(include_hidden=True)
    """
    i = WrapInnerContext(name="John", age=30, password="this should be hidden")
    s = WrapSingleNestedContext(link="https://www.google.com", private="this is private", inner_context=i)
    d = WrapDoubleNestedContext(company="Google", num_employees=100, key="this is hidden", single_nested_context=s)
    dump = d.model_dump(include_hidden=include_hidden)

    assert dump["company"] == "Google"
    assert dump["num_employees"] == 100
    assert dump["single_nested_context"]["link"] == "https://www.google.com"
    assert all(field in dump["single_nested_context"]["inner_context"] for field in HIDDEN_BASE_FIELDS)
    assert dump["single_nested_context"]["inner_context"]["name"] == "custom_name_value"
    assert dump["single_nested_context"]["inner_context"]["age"] == 30
    assert dump["single_nested_context"]["inner_context"]["password"] == "this should be hidden"
    if include_hidden:
        assert all(field in dump for field in HIDDEN_BASE_FIELDS)
        assert dump["key"] == "this is hidden"
        assert all(field in dump["single_nested_context"] for field in HIDDEN_BASE_FIELDS)
        assert dump["single_nested_context"]["private"] == "this is private"
    else:
        assert not any(field in dump for field in HIDDEN_BASE_FIELDS)
        assert "key" not in dump
        assert not any(field in dump["single_nested_context"] for field in HIDDEN_BASE_FIELDS)
        assert "private" not in dump["single_nested_context"]


class PlainInnerContext(Context):
    name: str
    age: int
    password: str = Field(hidden=True)

    @model_serializer(mode="plain")
    def ser_model(self) -> dict[str, Any]:
        return {
            "custom_name": "custom_name_value",
            "age": self.age + 5,
            "password": self.password + " and more",
        }


class PlainSingleNestedContext(Context):
    link: str
    private: str = Field(hidden=True)
    inner_context: PlainInnerContext


class PlainDoubleNestedContext(Context):
    company: str
    num_employees: int
    key: str = Field(hidden=True)
    single_nested_context: PlainSingleNestedContext


def test_dump_plain_serializer(include_hidden: bool):
    """
    Test dumping works as expected for custom serializer with and without model_dump(include_hidden=True)
    """
    i = PlainInnerContext(name="John", age=30, password="this should be hidden")
    dump = i.model_dump(include_hidden=include_hidden)
    assert dump == {
        "custom_name": "custom_name_value",
        "age": 35,
        "password": "this should be hidden and more",
    }


def test_dump_plain_serializer_single_nested(include_hidden: bool):
    """
    Test dumping works as expected for custom serializer and single nested contexts with and without model_dump(include_hidden=True)
    """
    i = PlainInnerContext(name="John", age=30, password="this should be hidden")
    s = PlainSingleNestedContext(link="https://www.google.com", private="this is private", inner_context=i)
    dump = s.model_dump(include_hidden=include_hidden)
    assert dump["link"] == "https://www.google.com"
    assert dump["inner_context"] == {
        "custom_name": "custom_name_value",
        "age": 35,
        "password": "this should be hidden and more",
    }
    if include_hidden:
        assert all(field in dump for field in HIDDEN_BASE_FIELDS)
        assert dump["private"] == "this is private"
    else:
        assert not any(field in dump for field in HIDDEN_BASE_FIELDS)
        assert "private" not in dump


def test_dump_plain_serializer_double_nested(include_hidden: bool):
    """
    Test dumping works as expected for custom serializer and double nested contexts with and without model_dump(include_hidden=True)
    """
    i = PlainInnerContext(name="John", age=30, password="this should be hidden")
    s = PlainSingleNestedContext(link="https://www.google.com", private="this is private", inner_context=i)
    d = PlainDoubleNestedContext(company="Google", num_employees=100, key="this is hidden", single_nested_context=s)
    dump = d.model_dump(include_hidden=include_hidden)
    assert dump["company"] == "Google"
    assert dump["num_employees"] == 100
    assert dump["single_nested_context"]["link"] == "https://www.google.com"
    assert dump["single_nested_context"]["inner_context"] == {
        "custom_name": "custom_name_value",
        "age": 35,
        "password": "this should be hidden and more",
    }
    if include_hidden:
        assert all(field in dump for field in HIDDEN_BASE_FIELDS)
        assert all(field in dump["single_nested_context"] for field in HIDDEN_BASE_FIELDS)
        assert dump["key"] == "this is hidden"
        assert dump["single_nested_context"]["private"] == "this is private"
    else:
        assert not any(field in dump for field in HIDDEN_BASE_FIELDS)
        assert not any(field in dump["single_nested_context"] for field in HIDDEN_BASE_FIELDS)
        assert "key" not in dump
        assert "private" not in dump["single_nested_context"]


class BaseContext(Context):
    state: str = "CA"
    weight: int = Field(hidden=True)


class InheritedContext(BaseContext):
    occupation: str
    ssn: str = Field(hidden=True)


class DoubleInheritedContext(InheritedContext):
    favorite_artist: str
    pin: str = Field(hidden=True)
    single_nested_context: SingleNestedContext


def test_dump_inherited_context(include_hidden: bool):
    i = InheritedContext(weight=251, occupation="freelance furry", ssn="123-45-6789")
    dump = i.model_dump(include_hidden=include_hidden)
    assert dump["occupation"] == "freelance furry"
    if include_hidden:
        assert all(field in dump for field in HIDDEN_BASE_FIELDS)
        assert dump["state"] == "CA"
        assert dump["weight"] == 251
        assert dump["ssn"] == "123-45-6789"
    else:
        assert not any(field in dump for field in HIDDEN_BASE_FIELDS)
        assert dump["state"] == "CA"
        assert "weight" not in dump
        assert "ssn" not in dump


def test_dump_double_inherited_context(include_hidden: bool):
    i = InnerContext(name="John", age=30, password="this should be hidden")
    s = SingleNestedContext(link="https://www.google.com", private="this is private", inner_context=i)
    d = DoubleInheritedContext(
        state="CA",
        weight=251,
        occupation="freelance furry",
        ssn="123-45-6789",
        favorite_artist="Sabrina Carpenter",
        pin="1234",
        single_nested_context=s,
    )
    dump = d.model_dump(include_hidden=include_hidden)
    assert dump["state"] == "CA"
    assert dump["occupation"] == "freelance furry"
    assert dump["favorite_artist"] == "Sabrina Carpenter"
    assert dump["single_nested_context"]["link"] == "https://www.google.com"
    assert dump["single_nested_context"]["inner_context"]["name"] == "John"
    assert dump["single_nested_context"]["inner_context"]["age"] == 30

    if include_hidden:  # include_hidden == True
        assert all(field in dump for field in HIDDEN_BASE_FIELDS)
        assert dump["weight"] == 251
        assert dump["ssn"] == "123-45-6789"
        assert dump["pin"] == "1234"
        assert dump["single_nested_context"]["private"] == "this is private"
        assert dump["single_nested_context"]["inner_context"]["password"] == "this should be hidden"
    else:  # include_hidden == False
        assert not any(field in dump for field in HIDDEN_BASE_FIELDS)
        assert "weight" not in dump
        assert "ssn" not in dump
        assert "pin" not in dump
        assert "private" not in dump["single_nested_context"]
        assert "password" not in dump["single_nested_context"]["inner_context"]
