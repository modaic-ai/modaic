import os
from dataclasses import asdict, dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Type,
)

from dotenv import find_dotenv, load_dotenv

from ..context.base import Context, Relation
from ..observability import Trackable, track_modaic_obj

env_file = find_dotenv(usecwd=True)
load_dotenv(env_file)


if TYPE_CHECKING:
    import gqlalchemy


class GraphDBConfig(Protocol):
    _client_class: ClassVar[Type["gqlalchemy.DatabaseClient"]]
    # CAVEAT: checking for this attribute is currently the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]]


class GraphDatabase(Trackable):
    """
    A database that stores context objects and relationships between them in a graph database.
    """

    def __init__(self, config: GraphDBConfig, **kwargs):
        Trackable.__init__(self, **kwargs)
        self.config = config
        if getattr(self.config, "_client_class", None) is None:
            raise ImportError("gqlalchemy is not installed. Install 'gqlalchemy' to use GraphDatabase backends.")
        self._client = self.config._client_class(**asdict(self.config))

    @track_modaic_obj
    def execute_and_fetch(self, query: str) -> List[Dict[str, Any]]:
        return self._client.execute_and_fetch(query)

    def execute(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        connection: Optional["gqlalchemy.Connection"] = None,
    ) -> None:
        self._client.execute(query, parameters or {}, connection)

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

    def create_node(self, node: Context) -> Optional[Context]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        node = node.to_gqlalchemy()
        created_node = self._client.create_node(node)
        if created_node is not None:
            return Context.from_gqlalchemy(created_node)

    def save_node(self, node: Context) -> "gqlalchemy.Node":
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        node = node.to_gqlalchemy(self)
        result = self._client.save_node(node)
        return result

    def save_nodes(self, nodes: List[Context]) -> None:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        nodes = [node.to_gqlalchemy(self) for node in nodes]
        self._client.save_nodes(nodes)

    def save_node_with_id(self, node: Context) -> Optional["gqlalchemy.Node"]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        node = node.to_gqlalchemy(self)
        result = self._client.save_node_with_id(node)
        return result

    def load_node(self, node: Context) -> Optional["gqlalchemy.Node"]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        node = node.to_gqlalchemy(self)
        result = self._client.load_node(node)
        return result

    def load_node_with_all_properties(self, node: Context) -> Optional["gqlalchemy.Node"]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        node = node.to_gqlalchemy(self)
        result = self._client.load_node_with_all_properties(node)
        return result

    def load_node_with_id(self, node: Context) -> Optional["gqlalchemy.Node"]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        node = node.to_gqlalchemy(self)
        result = self._client.load_node_with_id(node)
        return result

    def load_relationship(self, relationship: Relation) -> Optional["gqlalchemy.Relationship"]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        relationship = relationship.to_gqlalchemy(self)
        result = self._client.load_relationship(relationship)
        return result

    def load_relationship_with_id(self, relationship: Relation) -> Optional["gqlalchemy.Relationship"]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        relationship = relationship.to_gqlalchemy(self)
        result = self._client.load_relationship_with_id(relationship)
        return result

    def load_relationship_with_start_node_id_and_end_node_id(
        self, relationship: Relation
    ) -> Optional["gqlalchemy.Relationship"]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        relationship = relationship.to_gqlalchemy(self)
        result = self._client.load_relationship_with_start_node_id_and_end_node_id(relationship)
        return result

    def save_relationship(self, relationship: Relation) -> Optional["gqlalchemy.Relationship"]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        relationship = relationship.to_gqlalchemy(self)

        result = self._client.save_relationship(relationship)
        return result

    def save_relationships(self, relationships: List[Relation]) -> None:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        relationships = [relationship.to_gqlalchemy(self) for relationship in relationships]
        self._client.save_relationships(relationships)

    def save_relationship_with_id(self, relationship: Relation) -> Optional["gqlalchemy.Relationship"]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        relationship = relationship.to_gqlalchemy(self)
        result = self._client.save_relationship_with_id(relationship)
        return result

    def create_relationship(self, relationship: Relation) -> Optional["gqlalchemy.Relationship"]:
        """
        <Danger>This method is not thread safe. We are actively working on a solution to make it thread safe.</Danger>
        """
        relationship = relationship.to_gqlalchemy(self)
        result = self._client.create_relationship(relationship)
        return result


NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT = int(os.getenv("NEO4J_PORT", "7687"))
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")
NEO4J_ENCRYPTED = os.getenv("NEO4J_ENCRYPT", "false").lower() == "true"
NEO4J_CLIENT_NAME = os.getenv("NEO4J_CLIENT_NAME", "neo4j")


def load_neo4j() -> Optional[Type[Any]]:
    try:
        import gqlalchemy
    except ImportError:
        return None
    return gqlalchemy.Neo4j


def load_memgraph() -> Optional[Type[Any]]:
    try:
        import gqlalchemy
    except ImportError:
        return None
    return gqlalchemy.Memgraph


@dataclass
class Neo4jConfig:
    host: str = NEO4J_HOST
    port: int = NEO4J_PORT
    username: str = NEO4J_USERNAME
    password: str = NEO4J_PASSWORD
    encrypted: bool = (NEO4J_ENCRYPTED,)
    client_name: str = (NEO4J_CLIENT_NAME,)

    _client_class: ClassVar[Optional[Type[Any]]] = load_neo4j()


MG_HOST = os.getenv("MG_HOST", "127.0.0.1")
MG_PORT = int(os.getenv("MG_PORT", "7687"))
MG_USERNAME = os.getenv("MG_USERNAME", "")
MG_PASSWORD = os.getenv("MG_PASSWORD", "")
MG_ENCRYPTED = os.getenv("MG_ENCRYPT", "false").lower() == "true"
MG_CLIENT_NAME = os.getenv("MG_CLIENT_NAME", "GQLAlchemy")
MG_LAZY = os.getenv("MG_LAZY", "false").lower() == "true"


@dataclass
class MemgraphConfig:
    host: str = MG_HOST
    port: int = MG_PORT
    username: str = MG_USERNAME
    password: str = MG_PASSWORD
    encrypted: bool = MG_ENCRYPTED
    client_name: str = MG_CLIENT_NAME
    lazy: bool = MG_LAZY

    _client_class: ClassVar[Optional[Type[Any]]] = load_memgraph()
