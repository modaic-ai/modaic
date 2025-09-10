import os
import shutil
import uuid
from typing import Optional

import numpy as np
import pytest

# Import your real backend + types
from modaic.context import Context, Text
from modaic.databases import MilvusBackend, VectorDatabase
from modaic.databases.vector_database.vector_database import VectorDBBackend
from modaic.databases.vector_database.vendors.milvus import mql_to_milvus
from modaic.types import Array, String
from tests.testing_utils import DummyEmbedder, HardcodedEmbedder


def _read_hosted_config():  # noqa: ANN001
    """
    Read hosted Milvus configuration from environment variables.

    Returns:
        dict: Configuration dictionary with uri, user, password, db_name, and token
    """
    return {
        "uri": os.environ.get("MILVUS_URI"),
        "user": os.environ.get("MILVUS_USER"),
        "password": os.environ.get("MILVUS_PASSWORD"),
        "db_name": os.environ.get("MILVUS_DB_NAME"),
        "token": os.environ.get("MILVUS_TOKEN"),
    }


# ---------------------------
# Param: which backend flavor
# ---------------------------
@pytest.fixture(params=["lite", "hosted"])
def milvus_mode(request):  # noqa: ANN001 ANN201
    """
    Fixture that provides Milvus mode (lite or hosted) for parameterized tests.

    Params:
        request: pytest request object

    Returns:
        str: Either "lite" or "hosted"
    """
    cfg = _read_hosted_config()
    if request.param == "hosted" and not cfg["uri"]:
        pytest.skip("No hosted Milvus configured (set MILVUS_URI environment variable).")
    return request.param


# ---------------------------
# URIs & clients for each mode
# ---------------------------
@pytest.fixture(scope="session")
def milvus_lite_dbfile(tmp_path_factory):  # noqa: ANN001 ANN201
    """
    Create a temporary database file for Milvus Lite testing.

    Params:
        tmp_path_factory: pytest temporary path factory

    Returns:
        str: Path to temporary database file
    """
    root = tmp_path_factory.mktemp("milvus_lite")
    path = root / "test.db"
    yield str(path)
    # cleanup any aux files/directories Milvus Lite may create
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def hosted_cfg():  # noqa: ANN001 ANN201
    """
    Provide hosted Milvus configuration for tests.

    Returns:
        dict: Hosted configuration dictionary
    """
    return _read_hosted_config()


@pytest.fixture
def vector_database(milvus_mode: str, milvus_lite_dbfile: str, hosted_cfg: dict):  # noqa: ANN001 ANN201
    """
    Returns a real pymilvus MilvusClient connected to Lite or Hosted, depending on milvus_mode.

    Params:
        milvus_mode: Either "lite" or "hosted"
        milvus_lite_dbfile: Path to database file for lite mode
        hosted_cfg: Configuration dictionary for hosted mode

    Returns:
        VectorDatabase: Configured vector database instance
    """
    # Create a default embedder for testing
    default_embedder = DummyEmbedder()

    if milvus_mode == "lite":
        vector_database = VectorDatabase(MilvusBackend.from_local(milvus_lite_dbfile), embedder=default_embedder)
    else:
        vector_database = VectorDatabase(
            MilvusBackend(
                uri=hosted_cfg["uri"],
                user=hosted_cfg["user"] or "",
                password=hosted_cfg["password"] or "",
                db_name=hosted_cfg["db_name"] or "",
                token=hosted_cfg["token"] or "",
            ),
            embedder=default_embedder,
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


# ---------------------------
# Throwaway collection per test
# ---------------------------
@pytest.fixture
def collection_name(vector_database: VectorDatabase):  # noqa: ANN001 ANN201
    """
    Yields a unique collection name; drops it after the test if it was created.

    Params:
        vector_database: Vector database instance

    Returns:
        str: Unique collection name
    """
    name = f"t_{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        try:
            if vector_database.has_collection(name):
                vector_database.drop_collection(name)
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

    def embedme(self) -> str:
        return self.field9.text


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


def test_milvus_implementes_vector_db_backend(vector_database: VectorDatabase):
    backend = vector_database.ext.backend
    assert isinstance(backend, VectorDBBackend)


def test_create_collection(vector_database: VectorDatabase, collection_name: str):
    vector_database.create_collection(collection_name, CustomContext)
    assert vector_database.has_collection(collection_name)


def test_drop_collection(vector_database: VectorDatabase, collection_name: str):
    vector_database.create_collection(collection_name, CustomContext)
    assert vector_database.has_collection(collection_name)
    vector_database.drop_collection(collection_name)
    assert not vector_database.has_collection(collection_name)


def test_list_collections(vector_database: VectorDatabase, collection_name: str):
    vector_database.create_collection(collection_name, CustomContext)
    assert collection_name in vector_database.list_collections()


def test_has_collection(vector_database: VectorDatabase, collection_name: str):
    vector_database.create_collection(collection_name, CustomContext)
    assert vector_database.has_collection(collection_name)
    vector_database.drop_collection(collection_name)
    assert not vector_database.has_collection(collection_name)


def test_record_ops(vector_database: VectorDatabase, collection_name: str):
    vector_database.create_collection(collection_name, CustomContext, embedder=DummyEmbedder())
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
    vector_database.add_records(collection_name, [context])
    assert vector_database.has_collection(collection_name)
    assert vector_database.get_records(collection_name, [context.id])[0] == context


def test_search(vector_database: VectorDatabase, collection_name: str):
    hardcoded_embedder = HardcodedEmbedder()
    vector_database.create_collection(collection_name, CustomContext, embedder=hardcoded_embedder)
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
    # CAVEAT: these lines set hardcoded embedder to return the given embeddings when run with the same strings.
    hardcoded_embedder("query", np.array([3, 5, 7]))
    hardcoded_embedder("record1", np.array([4, 5, 6]))  # 0.988195
    hardcoded_embedder("record2", np.array([6, 3, 0]))  # 0.539969
    hardcoded_embedder("record3", np.array([1, 0, 0]))  # 0.329293
    # raise ValueError("faile here")

    vector_database.add_records(collection_name, [("record1", context1), ("record2", context2), ("record3", context3)])
    print("returned context", vector_database.search(collection_name, "query", k=1)[0][0].context)
    print("expected context", context1)

    assert vector_database.search(collection_name, "query", k=1)[0][0].context == context1
    assert vector_database.search(collection_name, "query", k=1)[0][1].context == context2
    assert vector_database.search(collection_name, "query", k=1)[0][2].context == context3


def test_search_with_filters(vector_database: VectorDatabase[MilvusBackend], collection_name: str):
    hardcoded_embedder = HardcodedEmbedder()
    vector_database.create_collection(collection_name, CustomContext, embedder=hardcoded_embedder)
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
    hardcoded_embedder("record1", np.array([4, 5, 6]))  # Cosine similarity 0.988195
    hardcoded_embedder("record2", np.array([6, 3, 0]))  # Cosine similarity 0.539969
    hardcoded_embedder("record3", np.array([1, 0, 0]))  # Cosine similarity 0.329293

    vector_database.add_records(collection_name, [("record1", context1), ("record2", context2), ("record3", context3)])
    filter1 = CustomContext.field1 == "test2"
    assert vector_database.search(collection_name, vector, CustomContext, 1, filter1.query)[0][0].context == context1

    filter2 = CustomContext.field2 > 2
    assert vector_database.search(collection_name, vector, CustomContext, 1, filter2.query)[0][0].context == context3

    filter3 = CustomContext.field4 < 3.0
    assert vector_database.search(collection_name, vector, CustomContext, 1, filter3.query)[0][0].context == context1

    filter4 = CustomContext.field12.in_(["test2", "test3"])
    assert vector_database.search(collection_name, vector, CustomContext, 1, filter4.query)[0][0].context == context2

    filter5 = CustomContext.field10.not_in(["test", "test2"])
    assert vector_database.search(collection_name, vector, CustomContext, 1, filter5.query)[0][0].context == context3

    filter6 = CustomContext.field2 != 2
    assert vector_database.search(collection_name, vector, CustomContext, 1, filter6.query)[0][0].context == context1

    filter7 = CustomContext.field3.contains("test2")
    assert vector_database.search(collection_name, vector, CustomContext, 1, filter7.query)[0][0].context == context2

    filter8 = CustomContext.field12["test2"] == 2
    assert vector_database.search(collection_name, vector, CustomContext, 1, filter8.query)[0][0].context == context2

    filter9 = (CustomContext.field4 < 3.1) & (CustomContext.field4 > 1.9)
    assert vector_database.search(collection_name, vector, CustomContext, 1, filter9.query)[0][0].context == context2

    filter10 = (CustomContext.field4 < 3.1) | (CustomContext.field4 > 1.9) & (CustomContext.field2 != 2)
    assert vector_database.search(collection_name, vector, CustomContext, 1, filter10.query)[0][0].context == context3
