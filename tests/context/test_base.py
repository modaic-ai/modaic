import json
from typing import Any

import pytest
from pydantic import Field

from modaic.context import Context
from modaic.context.base import Hydratable, is_embeddable, is_hydratable, is_multi_embeddable
from modaic.context.table import Table, TableFile


class User(Context):
    name: str
    api_key: str = Field(hidden=True)


def test_model_dump_includes_hidden_when_requested():
    u = User(name="Ada", api_key="SECRET")
    assert "api_key" not in u.model_dump()
    full = u.model_dump(include_hidden=True)
    assert full["api_key"] == "SECRET"
    assert "id" in full and "metadata" in full


def test_model_dump_json_roundtrip():
    u = User(name="Ada", api_key="SECRET")
    s = u.model_dump_json(include_hidden=True)
    assert isinstance(s, (bytes, str))


def test_schema_creation_returns_simplified_schema():
    sch = User.schema().as_dict()
    assert "id" in sch and sch["id"].type == "string"
    assert sch["name"].type == "string"


def test_chunk_with_and_apply_to_chunks():
    from modaic.context import Text

    t = Text(text="alpha beta gamma")
    t.chunk_text(lambda s: s.split())
    print("CHUNLS", t._chunks)
    assert [c.text for c in t.chunks] == ["alpha", "beta", "gamma"]
    t.apply_to_chunks(lambda c: c.metadata.update({"len": len(c.text)}))
    assert [c.metadata["len"] for c in t.chunks] == [5, 4, 5]


def test_is_hydratable_protocol_and_helper(tmp_path):
    # TableFile implements hydration
    from modaic.storage import InPlaceFileStore

    store = InPlaceFileStore("tests/artifacts/test_dir")
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


def test_external_serializer_excludes_hidden():
    # Roundtrip via json should not include hidden fields
    u = User(name="Ada", api_key="SECRET")
    data = json.loads(u.model_dump_json())
    assert "api_key" not in data
