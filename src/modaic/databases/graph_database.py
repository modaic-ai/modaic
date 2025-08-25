from typing import (
    ClassVar,
    Dict,
    Any,
    Type,
    Protocol,
    TYPE_CHECKING,
    List,
    Optional,
    Iterator,
)
from dataclasses import dataclass, asdict
from ..context.base import ContextSchema, Relation

if TYPE_CHECKING:
    import gqlalchemy
    # from gqlalchemy.vendors.database_client import DatabaseClient
    # from gqlalchemy.connection import Connection
    # from gqlalchemy.models import Index, Constraint


class GraphDBConfig(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]]
    _client_class: ClassVar[Type["gqlalchemy.DatabaseClient"]]


class GraphDatabase:
    """
    A database that stores context objects and relationships between them in a graph database.
    """

    def __init__(self, config: GraphDBConfig, **kwargs):
        self.config = config
        self._client = self.config._client_class(**asdict(self.config))

    def execute_and_fetch(self, query: str) -> List[Dict[str, Any]]:
        return self._client.execute_and_fetch(query)

    def execute(
        self,
        query: str,
        parameters: Dict[str, Any] = {},
        connection: Optional["gqlalchemy.Connection"] = None,
    ) -> None:
        self._client.execute(query, parameters, connection)

    def create_index(self, index: "gqlalchemy.Index") -> None:
        self._client.create_index(index)

    def drop_index(self, index: "gqlalchemy.Index") -> None:
        self._client.drop_index(index)

    def get_indexes(self) -> List["gqlalchemy.Index"]:
        return self._client.get_indexes()

    def ensure_indexes(self, indexes: List["gqlalchemy.Index"]) -> None:
        self._client.ensure_indexes(indexes)

    def drop_indexes(self) -> None:
        self._client.drop_indexes()

    def create_constraint(self, constraint: "gqlalchemy.Constraint") -> None:
        self._client.create_constraint(constraint)

    def drop_constraint(self, constraint: "gqlalchemy.Constraint") -> None:
        self._client.drop_constraint(constraint)
    
    def get_constraints(self) -> List["gqlalchemy.Constraint"]:
        return self._client.get_constraints()
    
    def get_exists_constraints(self) -> List["gqlalchemy.Constraint"]:
        return self._client.get_exists_constraints()
    
    def get_unique_constraints(self) -> List["gqlalchemy.Constraint"]:
        return self._client.get_unique_constraints()
    
    def ensure_constraints(self, constraints: List["gqlalchemy.Constraint"]) -> None:
        self._client.ensure_constraints(constraints)
    
    def drop_database(self) -> None:
        self._client.drop_database()
    
    def new_connection(self) -> "gqlalchemy.Connection":
        return self._client.new_connection()
    
    def get_variable_assume_one(self, query_result: Iterator[Dict[str, Any]], variable_name: str) -> Any:
        return self._client.get_variable_assume_one(query_result, variable_name)
    
    def create_node(self, node: ContextSchema) -> None:
        
        
    def save_node(self, node: Node) -> None: ...
    def save_nodes(self, nodes: List[Node]) -> None: ...
    def save_node_with_id(self, node: Node) -> None: ...
    def load_node(self, node: Node) -> Optional[Node]: ...
    def load_node_with_all_properties(self, node: Node) -> Optional[Node]: ...
    def load_node_with_id(self, node: Node) -> Optional[Node]: ...
    def load_relationship(
        self, relationship: Relationship
    ) -> Optional[Relationship]: ...
    def load_relationship_with_id(
        self, relationship: Relationship
    ) -> Optional[Relationship]: ...
    def load_relationship_with_start_node_id_and_end_node_id(
        self, relationship: Relationship
    ) -> Optional[Relationship]: ...

    def save_relationship(
        self, relationship: Relationship
    ) -> Optional[Relationship]: ...

    def save_relationships(self, relationships: List[Relationship]) -> None: ...
    def save_relationship_with_id(
        self, relationship: Relationship
    ) -> Optional[Relationship]: ...
    def create_relationship(
        self, relationship: Relationship
    ) -> Optional[Relationship]: ...


class Neo4jConfig:
    host: str
    port: int
    username: str
    password: str
    database: str
    driver: str
    driver_args: Dict[str, Any]
    driver_kwargs: Dict[str, Any]
    driver_kwargs: Dict[str, Any]
