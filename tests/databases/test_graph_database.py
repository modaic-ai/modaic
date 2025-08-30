from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Optional

import gqlalchemy
import pytest

from modaic.databases.graph_database import GraphDatabase
from modaic.context.base import ContextSchema, Relation


class SampleNodeSchema(ContextSchema):
    """Minimal schema for testing node CRUD operations."""

    value: int


class SampleRelation(Relation):
    """Minimal relation for testing relationship CRUD operations."""

    weight: int | None = None


class FakeClient:
    """In-memory fake client implementing the subset of the gqlalchemy DatabaseClient API used by GraphDatabase."""

    def __init__(self, **_: Any):
        self.executed: list[tuple[str, Dict[str, Any]]] = []
        self.indexes: list[Any] = []
        self.constraints: list[Any] = []
        self.dropped_db: bool = False
        self.saved_nodes: list[gqlalchemy.Node] = []
        self.created_nodes: list[gqlalchemy.Node] = []
        self.saved_relationships: list[gqlalchemy.Relationship] = []
        self.created_relationships: list[gqlalchemy.Relationship] = []

    # Query execution
    def execute_and_fetch(self, query: str) -> List[Dict[str, Any]]:
        self.executed.append((query, {}))
        return [{"result": 1}]

    def execute(
        self,
        query: str,
        parameters: Dict[str, Any] | None = None,
        connection: Optional[gqlalchemy.Connection] = None,
    ) -> None:
        self.executed.append((query, parameters or {}))

    # Indexes
    def create_index(self, index: Any) -> None:
        self.indexes.append(index)

    def drop_index(self, index: Any) -> None:
        self.indexes = [i for i in self.indexes if i != index]

    def get_indexes(self) -> List[Any]:
        return list(self.indexes)

    def ensure_indexes(self, indexes: List[Any]) -> None:
        for idx in indexes:
            if idx not in self.indexes:
                self.indexes.append(idx)

    def drop_indexes(self) -> None:
        self.indexes.clear()

    # Constraints
    def create_constraint(self, constraint: Any) -> None:
        self.constraints.append(constraint)

    def drop_constraint(self, constraint: Any) -> None:
        self.constraints = [c for c in self.constraints if c != constraint]

    def get_constraints(self) -> List[Any]:
        return list(self.constraints)

    def get_exists_constraints(self) -> List[Any]:
        return [c for c in self.constraints if getattr(c, "kind", "") == "exists"]

    def get_unique_constraints(self) -> List[Any]:
        return [c for c in self.constraints if getattr(c, "kind", "") == "unique"]

    def ensure_constraints(self, constraints: List[Any]) -> None:
        for c in constraints:
            if c not in self.constraints:
                self.constraints.append(c)

    # DB and connections
    def drop_database(self) -> None:
        self.dropped_db = True

    def new_connection(self) -> "gqlalchemy.Connection":
        return object()  # type: ignore[return-value]

    def get_variable_assume_one(
        self, query_result: Iterator[Dict[str, Any]], variable_name: str
    ) -> Any:
        first = next(iter(query_result))
        return first[variable_name]

    # Nodes
    def create_node(self, node: gqlalchemy.Node) -> gqlalchemy.Node:
        self.created_nodes.append(node)
        return node

    def save_node(self, node: gqlalchemy.Node) -> gqlalchemy.Node:
        self.saved_nodes.append(node)
        return node

    def save_nodes(self, nodes: List[gqlalchemy.Node]) -> List[gqlalchemy.Node]:
        self.saved_nodes.extend(nodes)
        return nodes

    def save_node_with_id(self, node: gqlalchemy.Node) -> gqlalchemy.Node:
        self.saved_nodes.append(node)
        return node

    def load_node(self, node: gqlalchemy.Node) -> gqlalchemy.Node:
        return node

    def load_node_with_all_properties(self, node: gqlalchemy.Node) -> gqlalchemy.Node:
        return node

    def load_node_with_id(self, node: gqlalchemy.Node) -> gqlalchemy.Node:
        return node

    # Relationships
    def load_relationship(
        self, relationship: gqlalchemy.Relationship
    ) -> gqlalchemy.Relationship:
        return relationship

    def load_relationship_with_id(
        self, relationship: gqlalchemy.Relationship
    ) -> gqlalchemy.Relationship:
        return relationship

    def load_relationship_with_start_node_id_and_end_node_id(
        self, relationship: gqlalchemy.Relationship
    ) -> gqlalchemy.Relationship:
        return relationship

    def save_relationship(
        self, relationship: gqlalchemy.Relationship
    ) -> gqlalchemy.Relationship:
        self.saved_relationships.append(relationship)
        return relationship

    def save_relationships(self, relationships: List[gqlalchemy.Relationship]) -> None:
        self.saved_relationships.extend(relationships)

    def save_relationship_with_id(
        self, relationship: gqlalchemy.Relationship
    ) -> gqlalchemy.Relationship:
        self.saved_relationships.append(relationship)
        return relationship

    def create_relationship(
        self, relationship: gqlalchemy.Relationship
    ) -> gqlalchemy.Relationship:
        self.created_relationships.append(relationship)
        return relationship


@dataclass
class FakeConfig:
    """Dataclass config that fits GraphDatabase expectations."""

    host: str = "fake"
    port: int = 0

    _client_class: ClassVar[type[FakeClient]] = FakeClient


@pytest.fixture()
def db() -> GraphDatabase:
    """GraphDatabase wired with an in-memory FakeClient."""
    return GraphDatabase(FakeConfig())


def make_sample_node(value: int = 1) -> SampleNodeSchema:
    """Factory for a sample node schema instance."""
    return SampleNodeSchema(value=value)


def make_sample_rel(
    start: int = 1, end: int = 2, weight: int | None = 5
) -> SampleRelation:
    """Factory for a sample relation instance with int endpoints."""
    return SampleRelation(start_node=start, end_node=end, weight=weight)


def test_execute_and_fetch(db: GraphDatabase) -> None:
    """execute_and_fetch should delegate and return list of dicts."""
    assert db.execute_and_fetch("RETURN 1 AS result") == [{"result": 1}]


def test_execute(db: GraphDatabase) -> None:
    """execute should delegate without returning a value."""
    assert db.execute("CREATE ()") is None


def test_index_lifecycle(db: GraphDatabase) -> None:
    """Index management should reflect create, ensure, list and drop behaviors."""
    idx_a = {"index": "A"}
    idx_b = {"index": "B"}
    db.create_index(idx_a)
    db.ensure_indexes([idx_a, idx_b])
    assert idx_a in db.get_indexes()
    assert idx_b in db.get_indexes()
    db.drop_index(idx_a)
    assert idx_a not in db.get_indexes()
    db.drop_indexes()
    assert db.get_indexes() == []


def test_constraint_lifecycle(db: GraphDatabase) -> None:
    """Constraint management should reflect create, ensure, list, unique/exists and drop behaviors."""
    c_exists = type("C", (), {"name": "c1", "kind": "exists"})()
    c_unique = type("C", (), {"name": "c2", "kind": "unique"})()
    other = {"name": "c3"}
    db.create_constraint(c_exists)
    db.ensure_constraints([c_unique, other])
    constraints = db.get_constraints()
    assert c_exists in constraints and c_unique in constraints and other in constraints
    assert db.get_exists_constraints() == [c_exists]
    assert db.get_unique_constraints() == [c_unique]
    db.drop_constraint(other)
    assert other not in db.get_constraints()


def test_drop_database_and_connection(db: GraphDatabase) -> None:
    """Dropping database sets flag; new_connection returns a non-null object."""
    db.drop_database()
    assert db.new_connection() is not None


def test_get_variable_assume_one(db: GraphDatabase) -> None:
    """Should delegate to client and pick variable from first row."""
    data: Iterable[Dict[str, Any]] = iter([{"x": 42}])
    assert db.get_variable_assume_one(data, "x") == 42


def test_create_node_roundtrip(db: GraphDatabase) -> None:
    """create_node should convert to gqlalchemy.Node and back to ContextSchema."""
    node = make_sample_node(10)
    created = db.create_node(node)
    assert isinstance(created, ContextSchema)
    assert isinstance(created, SampleNodeSchema)
    assert created.value == 10


def test_save_node_roundtrip(db: GraphDatabase) -> None:
    """save_node should convert to gqlalchemy.Node and back to ContextSchema."""
    node = make_sample_node(11)
    saved = db.save_node(node)
    assert isinstance(saved, SampleNodeSchema)
    assert saved.value == 11


def test_save_nodes_roundtrip(db: GraphDatabase) -> None:
    """save_nodes should return a list of ContextSchema instances."""
    nodes = [make_sample_node(1), make_sample_node(2)]
    results = db.save_nodes(nodes)
    assert isinstance(results, list)
    assert [n.value for n in results] == [1, 2]


def test_save_node_with_id_roundtrip(db: GraphDatabase) -> None:
    """save_node_with_id should return ContextSchema when client echoes the node."""
    node = make_sample_node(12)
    out = db.save_node_with_id(node)
    assert isinstance(out, SampleNodeSchema)
    assert out.value == 12


def test_load_node_variants(db: GraphDatabase) -> None:
    """load_node, load_node_with_all_properties, and load_node_with_id return ContextSchema or None."""
    node = make_sample_node(13)
    assert isinstance(db.load_node(node), SampleNodeSchema)
    assert isinstance(db.load_node_with_all_properties(node), SampleNodeSchema)
    assert isinstance(db.load_node_with_id(node), SampleNodeSchema)


def test_relationship_load_variants(db: GraphDatabase) -> None:
    """load_relationship* should return Relation instances when client echoes the relationship."""
    rel = make_sample_rel(1, 2, 7)
    assert isinstance(db.load_relationship(rel), Relation)
    assert isinstance(db.load_relationship_with_id(rel), Relation)
    assert isinstance(
        db.load_relationship_with_start_node_id_and_end_node_id(rel), Relation
    )


def test_relationship_save_variants(db: GraphDatabase) -> None:
    """save_relationship*, create_relationship should return Relation instances or None for save_relationships."""
    rel = make_sample_rel(3, 4, 9)
    assert isinstance(db.save_relationship(rel), Relation)
    assert db.save_relationships([rel, make_sample_rel(4, 5, 10)]) is None
    assert isinstance(db.save_relationship_with_id(rel), Relation)
    assert isinstance(db.create_relationship(rel), Relation)
