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

if TYPE_CHECKING:
    from gqlalchemy.vendors.database_client import DatabaseClient
    from gqlalchemy.connection import Connection
    from gqlalchemy.models import Index, Constraint


class GraphDBConfig(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]]
    _backend_class: ClassVar[Type["DatabaseClient"]]


class GraphDatabase:
    """
    A database that stores context objects and relationships between them in a graph database.
    """

    def __init__(self, config: GraphDBConfig, **kwargs):
        self.config = config
        self._backend = self.config._backend_class(**asdict(self.config))

    def execute_and_fetch(self, query: str) -> List[Dict[str, Any]]:
        return self._backend.execute_and_fetch(query)

    def execute(
        self,
        query: str,
        parameters: Dict[str, Any] = {},
        connection: Optional["Connection"] = None,
    ) -> None: ...

    def create_index(self, index: Index) -> None:
        self._backend.create_index(index)

    def drop_index(index: Index) -> None: ...
    def get_indexes(self) -> List[Index]: ...
    def ensure_indexes(self, indexes: List[Index]) -> None: ...
    def drop_indexes() -> None: ...
    def create_constraint(index: Constraint) -> None: ...
    def drop_constraint(index: Constraint) -> None: ...
    def get_constraints(self) -> List[Constraint]: ...
    def ensure_constraints(self, constraints: List[Constraint]) -> None: ...
    def drop_database() -> None: ...
    def new_connection(self) -> "Connection": ...
    def get_variable_assume_one(
        self, query_result: Iterator[Dict[str, Any]], variable_name: str
    ) -> Any: ...
    def create_node(self, node: Node) -> None: ...
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
