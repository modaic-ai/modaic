import os
import shutil
import uuid
from typing import Optional

import numpy as np
import pytest
from pymilvus import MilvusClient

# Import your real backend + types
from modaic.context import Context, Text
from modaic.databases import IndexConfig, IndexType, MilvusBackend, VectorDatabase, VectorType
from modaic.databases.vector_database.vector_database import VectorDBBackend
from modaic.databases.vector_database.vendors.milvus import mql_to_milvus
from modaic.types import (
    Array,
    String,
    double,
    int8,
    int16,
    int32,
    int64,
)

from ....testing_utils import Membedder


# ---------------------------
# Pytest CLI options
# ---------------------------
def pytest_addoption(parser):  # noqa: ANN001
    group = parser.getgroup("milvus")
    group.addoption("--milvus-uri", action="store", default=None, help="Hosted Milvus URI (e.g. http://host:19530)")
    group.addoption("--milvus-user", action="store", default=None, help="Milvus username")
    group.addoption("--milvus-password", action="store", default=None, help="Milvus password")
    group.addoption("--milvus-db-name", action="store", default=None, help="Milvus database name")
    group.addoption("--milvus-token", action="store", default=None, help="Milvus token")


def _read_hosted_config(pytestconfig):  # noqa: ANN001
    # Prefer CLI flags, fall back to env vars
    def pick(flag, env):  # noqa: ANN001
        return pytestconfig.getoption(flag) or os.environ.get(env)

    return {
        "uri": pick("--milvus-uri", "MILVUS_URI"),
        "user": pick("--milvus-user", "MILVUS_USER"),
        "password": pick("--milvus-password", "MILVUS_PASSWORD"),
        "db_name": pick("--milvus-db-name", "MILVUS_DB_NAME"),
        "token": pick("--milvus-token", "MILVUS_TOKEN"),
    }


# ---------------------------
# Param: which backend flavor
# ---------------------------
@pytest.fixture(params=["lite", "hosted"])
def milvus_mode(request, pytestconfig):  # noqa: ANN001 ANN201
    cfg = _read_hosted_config(pytestconfig)
    if request.param == "hosted" and not cfg["uri"]:
        pytest.skip("No hosted Milvus configured (pass --milvus-uri or set MILVUS_URI).")
    return request.param


# ---------------------------
# URIs & clients for each mode
# ---------------------------
@pytest.fixture(scope="session")
def milvus_lite_dbfile(tmp_path_factory):  # noqa: ANN001 ANN201
    root = tmp_path_factory.mktemp("tests/artifacts/milvus_lite")
    path = root / "test.db"
    yield str(path)
    # cleanup any aux files/directories Milvus Lite may create
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def hosted_cfg(pytestconfig):  # noqa: ANN001 ANN201
    return _read_hosted_config(pytestconfig)


@pytest.fixture
def vector_database(milvus_mode: str, milvus_lite_dbfile: str, hosted_cfg: dict):  # noqa: ANN001 ANN201
    """
    Returns a real pymilvus MilvusClient connected to Lite or Hosted, depending on milvus_mode.
    """
    if milvus_mode == "lite":
        vector_database = VectorDatabase(MilvusBackend.from_local(milvus_lite_dbfile))
    else:
        vector_database = VectorDatabase(
            MilvusBackend(
                uri=hosted_cfg["uri"],
                user=hosted_cfg["user"] or "",
                password=hosted_cfg["password"] or "",
                db_name=hosted_cfg["db_name"] or "",
                token=hosted_cfg["token"] or "",
            )
        )

    # Smoke check: try a harmless op to verify connectivity
    try:
        _ = vector_database.list_collections()
    except Exception as e:
        pytest.skip(f"Milvus connection failed for mode={milvus_mode}: {e}")

    yield vector_database

    # Best-effort cleanup: drop only collections we created in tests
    try:
        for c in vector_database.list_collections():
            vector_database.drop_collection(c)
    except Exception:
        pass


class CustomContext(Context):
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


def test_mql_to_milvus_simple():
    """
    Test simple MQL to Milvus translation for equality, comparison, in/nin/like/exists, logical ops, and JSON path.

    Params:
        None
    """
    expr = mql_to_milvus({"field1": "foo"})
    assert expr == 'field1 == "foo"'

    expr = mql_to_milvus({"field2": {"$gt": 5, "$lte": 10}})
    assert "field2 > 5" in expr
    assert "field2 <= 10" in expr
    assert "AND" in expr

    expr = mql_to_milvus(
        {
            "field1": {"$in": ["a", "b"]},
            "field3": {"$exists": True},
            "field4": {"$like": "red%"},
            "field5": {"$nin": [1, 2]},
        }
    )
    assert 'field1 in ["a", "b"]' in expr
    assert "field3 IS NOT NULL" in expr
    assert 'field4 like "red%"' in expr
    assert "NOT (field5 in [1, 2])" in expr

    expr = mql_to_milvus(
        {
            "$and": [
                {"field2": {"$gte": 1}},
                {"$or": [{"field3": True}, {"field3": False}]},
            ]
        }
    )
    assert "field2 >= 1" in expr
    assert "OR" in expr
    assert "AND" in expr

    expr = mql_to_milvus({"product.model": {"$eq": "JSN-087"}})
    assert 'product["model"] == "JSN-087"' in expr


def test_mql_to_milvus_hard():
    """
    Test advanced MQL to Milvus translation for $expr, arithmetic, in/like/negation.

    Params:
        None
    """
    expr = mql_to_milvus({"$expr": {"$gt": [{"$add": ["$field2", 5]}, 10]}})
    assert "(field2 + 5) > 10" in expr

    expr = mql_to_milvus(
        {
            "$and": [
                {"$expr": {"$in": ["$field1", ["a", "b"]]}},
                {"$expr": {"$like": ["$field4", "red%"]}},
                {"$expr": {"$eq": [{"$neg": {"$sub": [10, 3]}}, -7]}},
            ]
        }
    )
    assert 'field1 in ["a", "b"]' in expr
    assert 'field4 like "red%"' in expr
    assert "-((10 - 3)) == -7" in expr


def test_milvus_implementes_vector_db_backend():
    assert issubclass(MilvusBackend, VectorDBBackend)


# ---------------------------
# Throwaway collection per test
# ---------------------------
@pytest.fixture
def collection_name(milvus_backend: MilvusBackend):  # noqa: ANN001 ANN201
    """
    Yields a unique collection name; drops it after the test if it was created.
    """
    name = f"t_{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        try:
            if milvus_backend.has_collection(name):
                milvus_backend.drop_collection(name)
        except Exception:
            pass


def test_create_collection(milvus_backend: MilvusBackend, collection_name: str):
    milvus_backend.create_collection(collection_name, CustomContext)
    assert milvus_backend.has_collection(collection_name)


def test_drop_collection(milvus_backend: MilvusBackend, collection_name: str):
    milvus_backend.create_collection(collection_name, CustomContext)
    assert milvus_backend.has_collection(collection_name)
    milvus_backend.drop_collection(collection_name)
    assert not milvus_backend.has_collection(collection_name)


def test_list_collections(milvus_backend: MilvusBackend, collection_name: str):
    milvus_backend.create_collection(collection_name, CustomContext)
    assert collection_name in milvus_backend.list_collections()


def test_has_collection(milvus_backend: MilvusBackend, collection_name: str):
    milvus_backend.create_collection(collection_name, CustomContext)
    assert milvus_backend.has_collection(collection_name)
    milvus_backend.drop_collection(collection_name)
    assert not milvus_backend.has_collection(collection_name)


def test_record_ops(milvus_backend: MilvusBackend, collection_name: str):
    milvus_backend.create_collection(collection_name, CustomContext)
    context = CustomContext(
        field1="test",
        field2=1,
        field3=True,
        field4=1.0,
        field5=["test"],
        field6={"test": 1},
        field7=[1, 2, 3],
        field8="test",
        field9=Text(text="test"),
        field10=["hello", "world"],
        field11=None,
        field12="test",
    )
    record = milvus_backend.create_record({"vector": np.array([1, 2, 3])}, context)
    milvus_backend.add_records(collection_name, [record])
    assert milvus_backend.has_collection(collection_name)
    assert milvus_backend.get_records(collection_name, CustomContext, [context.id])[0] == context


def test_search(milvus_backend: MilvusBackend, collection_name: str):
    milvus_backend.create_collection(collection_name, CustomContext)
    context1 = CustomContext(
        field1="test",
        field2=1,
        field3=True,
        field4=1.0,
        field5=["test"],
        field6={"test": 1},
        field7=[1, 2, 3],
        field8="test",
        field9=Text(text="test"),
        field10=["hello", "world"],
        field11=None,
        field12="test",
    )
    context2 = CustomContext(
        field1="test2",
        field2=2,
        field3=False,
        field4=2.0,
        field5=["test2"],
        field6={"test2": 2},
        field7=[4, 5, 6],
        field8="test2",
        field9=Text(text="test2"),
        field10=["hello2", "world2"],
        field11=None,
        field12="test2",
    )
    context3 = CustomContext(
        field1="test3",
        field2=3,
        field3=True,
        field4=3.0,
        field5=["test3"],
        field6={"test3": 3},
        field7=[7, 8, 9],
        field8="test3",
        field9=Text(text="test3"),
        field10=["hello3", "world3"],
        field11=None,
        field12="test3",
    )
    record1 = milvus_backend.create_record([{"vector": np.array([4, 5, 6])}], context1)  # 0.988195
    record2 = milvus_backend.create_record([{"vector": np.array([6, 3, 0])}], context2)  # 0.539969
    record3 = milvus_backend.create_record([{"vector": np.array([1, 0, 0])}], context3)  # 0.329293
    milvus_backend.add_records(collection_name, [record1, record2, record3])
    vector = np.array([3, 5, 7])
    assert milvus_backend.search(collection_name, vector, CustomContext, 1)[0][0].context == context1
    assert milvus_backend.search(collection_name, vector, CustomContext, 1)[0][1].context == context2
    assert milvus_backend.search(collection_name, vector, CustomContext, 1)[0][2].context == context3


def test_search_with_filters(milvus_backend: MilvusBackend, collection_name: str):
    context1 = CustomContext(
        field1="test",
        field2=1,
        field3=True,
        field4=1.0,
        field5=["test"],
        field6={"test": 1},
        field7=[1, 2, 3],
        field8="test",
        field9=Text(text="test"),
        field10=["hello", "world"],
        field11=None,
        field12="test",
    )
    context2 = CustomContext(
        field1="test2",
        field2=2,
        field3=False,
        field4=2.0,
        field5=["test2"],
        field6={"test2": 2},
        field7=[4, 5, 6],
        field8="test2",
        field9=Text(text="test2"),
        field10=["hello2", "world2"],
        field11=None,
        field12="test2",
    )
    context3 = CustomContext(
        field1="test3",
        field2=3,
        field3=True,
        field4=3.0,
        field5=["test3"],
        field6={"test3": 3},
        field7=[7, 8, 9],
        field8="test3",
        field9=Text(text="test3"),
        field10=["hello3", "world3"],
        field11=None,
        field12="test3",
    )
    vector = np.array([3, 5, 7])
    record1 = milvus_backend.create_record([{"vector": np.array([4, 5, 6])}], context1)  # Cosine similarity 0.988195
    record2 = milvus_backend.create_record([{"vector": np.array([6, 3, 0])}], context2)  # Cosine similarity 0.539969
    record3 = milvus_backend.create_record([{"vector": np.array([1, 0, 0])}], context3)  # Cosine similarity 0.329293

    milvus_backend.add_records(collection_name, [record1, record2, record3])
    filter1 = CustomContext.field1 == "test2"
    assert milvus_backend.search(collection_name, vector, CustomContext, 1, filter1.query)[0][0].context == context1

    filter2 = CustomContext.field2 > 2
    assert milvus_backend.search(collection_name, vector, CustomContext, 1, filter2.query)[0][0].context == context3

    filter3 = CustomContext.field4 < 3.0
    assert milvus_backend.search(collection_name, vector, CustomContext, 1, filter3.query)[0][0].context == context1

    filter4 = CustomContext.field12.in_(["test2", "test3"])
    assert milvus_backend.search(collection_name, vector, CustomContext, 1, filter4.query)[0][0].context == context2

    filter5 = CustomContext.field10.not_in(["test", "test2"])
    assert milvus_backend.search(collection_name, vector, CustomContext, 1, filter5.query)[0][0].context == context3

    filter6 = CustomContext.field2 != 2
    assert milvus_backend.search(collection_name, vector, CustomContext, 1, filter6.query)[0][0].context == context1

    filter7 = CustomContext.field3.contains("test2")
    assert milvus_backend.search(collection_name, vector, CustomContext, 1, filter7.query)[0][0].context == context2

    filter8 = CustomContext.field12["test2"] == 2
    assert milvus_backend.search(collection_name, vector, CustomContext, 1, filter8.query)[0][0].context == context2

    filter9 = (CustomContext.field4 < 3.1) & (CustomContext.field4 > 1.9)
    assert milvus_backend.search(collection_name, vector, CustomContext, 1, filter9.query)[0][0].context == context2

    filter10 = (CustomContext.field4 < 3.1) | (CustomContext.field4 > 1.9) & (CustomContext.field2 != 2)
    assert milvus_backend.search(collection_name, vector, CustomContext, 1, filter10.query)[0][0].context == context3
