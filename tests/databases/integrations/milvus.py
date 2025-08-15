import pytest
from modaic.databases import (
    VectorDatabase,
    MilvusVDBConfig,
)
from modaic.indexing import Embedder
import numpy as np
from modaic.context import ContextSchema, TextSchema
from modaic.databases.integrations.milvus import mql_to_milvus


class CustomContextSchema(ContextSchema):
    field1: str
    field2: int
    field3: bool
    field4: float
    field5: list[str]
    field6: dict[str, int]
    field7: TextSchema


@pytest.fixture
def milvus_vdb():
    # Create and return the object
    dummy_func = lambda x: np.random.rand(len(x), 10)  # noqa: E731
    dummy_embedder = Embedder(dummy_func)
    vdb_config = MilvusVDBConfig.from_local("tests/artifacts/index2.db")
    vdb = VectorDatabase(
        config=vdb_config,
        embedder=dummy_embedder,
        payload_schema=CustomContextSchema,
    )
    return vdb


def test_mql_simple_eq():
    expr = mql_to_milvus({"field1": "foo"})
    assert expr == 'field1 == "foo"'


def test_mql_comparison_ops():
    expr = mql_to_milvus({"field2": {"$gt": 5, "$lte": 10}})
    # order within AND is deterministic because we iterate items; allow either ordering by checking substrings
    assert "field2 > 5" in expr
    assert "field2 <= 10" in expr
    assert "AND" in expr


def test_mql_in_nin_like_exists():
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


def test_mql_logical_ops():
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


def test_mql_json_path_translation():
    expr = mql_to_milvus({"product.model": {"$eq": "JSN-087"}})
    assert 'product["model"] == "JSN-087"' in expr


def test_mql_expr_arithmetic_and_compare():
    # $expr with arithmetic: field2 + 5 > 10
    expr = mql_to_milvus({"$expr": {"$gt": [{"$add": ["$field2", 5]}, 10]}})
    assert "(field2 + 5) > 10" in expr


def test_mql_expr_in_like_and_neg():
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
