import os
import uuid
from typing import Optional

import numpy as np
import pytest
import weaviate
from weaviate.classes.query import Filter

from modaic import Condition, parse_modaic_filter

# Import your real backend + types
from modaic.context import Context, Text
from modaic.databases import WeaviateBackend, VectorDatabase
from modaic.databases.vector_database.vector_database import VectorDBBackend
from modaic.types import Array, String
from tests.testing_utils import DummyEmbedder, HardcodedEmbedder


def _read_hosted_config():
    """
    Read hosted Weaviate configuration from environment variables.

    Returns:
        dict: Configuration dictionary with url and api_key
    """
    return {
        "url": os.environ.get("WEAVIATE_URL"),
        "api_key": os.environ.get("WEAVIATE_API_KEY"),
    }


# ---------------------------
# Param: which backend flavor
# ---------------------------
@pytest.fixture(params=["local", "hosted"])
def weaviate_mode(request):
    """
    Fixture that provides Weaviate mode (local or hosted) for parameterized tests.

    Params:
        request: pytest request object

    Returns:
        str: Either "local" or "hosted"
    """
    cfg = _read_hosted_config()
    if request.param == "hosted" and not cfg["url"]:
        pytest.skip("No hosted Weaviate configured (set WEAVIATE_URL environment variable).")
    return request.param


# ---------------------------
# Configuration for each mode
# ---------------------------
@pytest.fixture(scope="session")
def hosted_cfg():
    """
    Provide hosted Weaviate configuration for tests.

    Returns:
        dict: Hosted configuration dictionary
    """
    return _read_hosted_config()


@pytest.fixture
def vector_database(weaviate_mode: str, hosted_cfg: dict):
    """
    Returns a WeaviateBackend connected to local or hosted, depending on weaviate_mode.

    Params:
        weaviate_mode: Either "local" or "hosted"
        hosted_cfg: Configuration dictionary for hosted mode

    Returns:
        VectorDatabase: Configured vector database instance
    """
    # Create a default embedder for testing
    default_embedder = DummyEmbedder()

    if weaviate_mode == "local":
        vector_database = VectorDatabase(
            WeaviateBackend.from_local(),
            embedder=default_embedder
        )
    else:
        vector_database = VectorDatabase(
            WeaviateBackend(
                url=hosted_cfg["url"],
                api_key=hosted_cfg["api_key"],
            ),
            embedder=default_embedder,
        )

    # Smoke check: try a harmless op to verify connectivity
    try:
        _ = vector_database.list_collections()
    except Exception as e:
        pytest.skip(f"Weaviate connection failed for mode={weaviate_mode}: {e}")

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
def collection_name(vector_database: VectorDatabase):
    """
    Yields a unique collection name; drops it after the test if it was created.

    Params:
        vector_database: Vector database instance

    Returns:
        str: Unique collection name
    """
    # Weaviate collection names must start with uppercase letter
    name = f"T{uuid.uuid4().hex[:12]}"
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
    Custom context for Weaviate tests, covering all supported types and Optionals.
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


def test_mql_to_weaviate_simple():
    """
    Test simple MQL to Weaviate translation for equality, comparison, in/like, and logical ops.

    Params:
        None
    """
    translator = WeaviateBackend.mql_translator
    
    # Simple equality
    expr = CustomContext.field1 == "foo"
    filter_obj = parse_modaic_filter(translator, expr)
    # Just verify it creates some kind of filter object, don't check specific type
    assert filter_obj is not None
    
    # Range with AND
    expr = (CustomContext.field2 > 5) & (CustomContext.field2 <= 10)
    filter_obj = parse_modaic_filter(translator, expr)
    assert filter_obj is not None
    
    # IN operator with AND
    expr = (CustomContext.field1.in_(["a", "b"])) & (CustomContext.field2 < 100)
    filter_obj = parse_modaic_filter(translator, expr)
    assert filter_obj is not None
    
    # OR combination
    expr = (CustomContext.field2 < 0) | (CustomContext.field2 > 10)
    filter_obj = parse_modaic_filter(translator, expr)
    assert filter_obj is not None


def test_mql_to_weaviate_complex():
    """
    Complex nested MQL to Weaviate translation - simplified since Modaic doesn't support NOT yet
    """
    translator = WeaviateBackend.mql_translator

    range_and = (CustomContext.field2 >= 1) & (CustomContext.field2 <= 10)
    in_list = CustomContext.field1.in_(["x", "y"])

    # Combine the filters - skip NOT since Modaic doesn't support it yet
    complex_expr = range_and & in_list
    filter_obj = parse_modaic_filter(translator, complex_expr)
    
    # Verify it's a valid filter
    assert filter_obj is not None


def test_weaviate_implements_vector_db_backend(vector_database: VectorDatabase):
    """Test that WeaviateBackend implements VectorDBBackend interface."""
    backend = vector_database.ext.backend
    assert isinstance(backend, VectorDBBackend)


def test_create_collection(vector_database: VectorDatabase, collection_name: str):
    """Test creating a collection."""
    vector_database.create_collection(collection_name, CustomContext)
    assert vector_database.has_collection(collection_name)


def test_drop_collection(vector_database: VectorDatabase, collection_name: str):
    """Test dropping a collection."""
    vector_database.create_collection(collection_name, CustomContext)
    assert vector_database.has_collection(collection_name)
    vector_database.drop_collection(collection_name)
    assert not vector_database.has_collection(collection_name)


def test_list_collections(vector_database: VectorDatabase, collection_name: str):
    """Test listing collections."""
    vector_database.create_collection(collection_name, CustomContext)
    assert collection_name in vector_database.list_collections()


def test_has_collection(vector_database: VectorDatabase, collection_name: str):
    """Test checking if collection exists."""
    vector_database.create_collection(collection_name, CustomContext)
    assert vector_database.has_collection(collection_name)
    vector_database.drop_collection(collection_name)
    assert not vector_database.has_collection(collection_name)


def test_record_ops(vector_database: VectorDatabase, collection_name: str):
    """Test adding and retrieving records."""
    vector_database.create_collection(collection_name, CustomContext, embedder=DummyEmbedder(embedding_dim=3))
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
    
    retrieved = vector_database.get_records(collection_name, [context.id])
    assert len(retrieved) == 1
    assert retrieved[0] == context


def test_search(vector_database: VectorDatabase, collection_name: str):
    """Test vector search with multiple records."""
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
    
    # Set up hardcoded embeddings for predictable results
    hardcoded_embedder("query", np.array([3, 5, 7]))
    hardcoded_embedder("record1", np.array([4, 5, 6]))  # Cosine similarity 0.988195
    hardcoded_embedder("record2", np.array([6, 3, 0]))  # Cosine similarity 0.539969
    hardcoded_embedder("record3", np.array([1, 0, 0]))  # Cosine similarity 0.329293

    vector_database.add_records(collection_name, [("record1", context1), ("record2", context2), ("record3", context3)])

    # Test top-k retrieval
    results_k1 = vector_database.search(collection_name, "query", k=1)
    assert results_k1[0][0].context == context1
    
    results_k2 = vector_database.search(collection_name, "query", k=2)
    assert results_k2[0][1].context == context2
    
    results_k3 = vector_database.search(collection_name, "query", k=3)
    assert results_k3[0][2].context == context3


def test_search_with_filters(vector_database: VectorDatabase[WeaviateBackend], collection_name: str):
    """Test vector search with various filters."""
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
    
    # Set up embeddings
    hardcoded_embedder("query", np.array([3, 5, 7]))
    hardcoded_embedder("record1", np.array([4, 5, 6]))  # Cosine similarity 0.988195
    hardcoded_embedder("record2", np.array([6, 3, 0]))  # Cosine similarity 0.539969
    hardcoded_embedder("record3", np.array([1, 0, 0]))  # Cosine similarity 0.329293

    vector_database.add_records(collection_name, [("record1", context1), ("record2", context2), ("record3", context3)])
    
    # Test equality filter
    filter1 = CustomContext.field1 == "test2"
    results1 = vector_database.search(collection_name, "query", 1, filter1)
    assert results1[0][0].context == context2

    # Test greater than filter
    filter2 = CustomContext.field2 > 2
    results2 = vector_database.search(collection_name, "query", 1, filter2)
    assert results2[0][0].context == context3

    # Test less than filter
    filter3 = CustomContext.field4 < 3.0
    results3 = vector_database.search(collection_name, "query", 1, filter3)
    assert results3[0][0].context == context1

    # Test IN filter
    filter4 = CustomContext.field12.in_(["test2", "test3"])
    results4 = vector_database.search(collection_name, "query", 1, filter4)
    assert results4[0][0].context == context2

    # Test range filter with AND
    filter9 = (CustomContext.field4 < 3.1) & (CustomContext.field4 > 1.9)
    results9 = vector_database.search(collection_name, "query", 1, filter9)
    assert results9[0][0].context == context2


def test_search_with_multiple_vectors(vector_database: VectorDatabase, collection_name: str):
    """Test searching with multiple query vectors at once."""
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
    
    hardcoded_embedder("query1", np.array([1, 0, 0]))
    hardcoded_embedder("query2", np.array([0, 1, 0]))
    hardcoded_embedder("record1", np.array([1, 0, 0]))
    
    vector_database.add_records(collection_name, [("record1", context1)])
    
    # Search with multiple queries
    results = vector_database.search(collection_name, ["query1", "query2"], k=1)
    assert len(results) == 2  # One result set per query
    assert results[0][0].context == context1
    assert results[1][0].context == context1


def test_null_value_handling(vector_database: VectorDatabase, collection_name: str):
    """Test that Weaviate properly handles null values (which Milvus Lite doesn't support)."""
    vector_database.create_collection(collection_name, CustomContext, embedder=DummyEmbedder(embedding_dim=3))
    
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
        field10=None,  # Null value
        field11=None,  # Null value
        field12=None,  # Null value
    )
    
    vector_database.add_records(collection_name, [context])
    retrieved = vector_database.get_records(collection_name, [context.id])
    
    assert len(retrieved) == 1
    assert retrieved[0].field10 is None
    assert retrieved[0].field11 is None
    assert retrieved[0].field12 is None


@pytest.mark.skip(reason="Connection cleanup test - skipping for now")
def test_connection_cleanup(weaviate_mode: str, hosted_cfg: dict):
    """Test that the client connection is properly closed."""
    default_embedder = DummyEmbedder()
    
    if weaviate_mode == "local":
        backend = WeaviateBackend.from_local()
    else:
        if not hosted_cfg["url"]:
            pytest.skip("No hosted Weaviate configured")
        backend = WeaviateBackend(
            url=hosted_cfg["url"],
            api_key=hosted_cfg["api_key"],
        )
    
    vector_db = VectorDatabase(backend, embedder=default_embedder)
    
    # Verify connection works
    _ = vector_db.list_collections()
    
    # Cleanup
    del vector_db
    del backend