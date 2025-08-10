from pydantic import BaseModel, ValidationError
from typing import Optional, Type, Literal, List, ClassVar, Iterable, Tuple
import dspy
from dataclasses import dataclass
from ..context.base import Context, SerializedContext
import numpy as np
import importlib
import inspect
from tqdm import tqdm
from .. import Embedder
from ..types import pydantic_to_modaic_schema
from aenum import AutoNumberEnum


class Vector(AutoNumberEnum):
    _init_ = "supported_libraries"
    # name | supported_libraries
    FLOAT = ["milvus", "qdrant", "mongo", "pinecone"]
    VECTOR = ["milvus", "qdrant", "mongo"]
    FLOAT16VECTOR = ["milvus"]


class IndexType(AutoNumberEnum):
    _init_ = "supported_libraries"
    # name | supported_libraries
    DEFAULT = ["milvus", "qdrant", "mongo", "pinecone"]
    HNSW = ["milvus", "qdrant", "mongo"]
    FLAT = ["milvus", "redis"]
    IVF_FLAT = ["milvus"]
    IVF_SQ8 = ["milvus"]
    IVF_PQ = ["milvus"]
    IVF_RABITQ = ["milvus"]
    GPU_IVF_FLAT = ["milvus"]
    GPU_IVF_PQ = ["milvus"]
    DISKANN = ["milvus"]
    BIN_FLAT = ["milvus"]
    BIN_IVF_FLAT = ["milvus"]
    MINHASH_LSH = ["milvus"]
    SPARSE_INVERTED_INDEX = ["milvus"]
    INVERTED = ["milvus"]
    BITMAP = ["milvus"]
    TRIE = ["milvus"]
    STL_SORT = ["milvus"]


class Metric(AutoNumberEnum):
    _init_ = "supported_libraries"
    EUCLIDIAN = {
        "milvus": "L2",
        "qdrant": "Euclid",
        "mongo": "euclidean",
        "pinecone": "euclidian",
    }
    DOT_PRODUCT = {
        "milvus": "IP",
        "qdrant": "Dot",
        "mongo": "dotProduct",
        "pinecone": "dotproduct",
    }
    COSINE = {
        "milvus": "COSINE",
        "qdrant": "Cosine",
        "mongo": "cosine",
        "pinecone": "cosine",
    }
    MANHATTAN = {
        "qdrant": "Manhattan",
        "mongo": "manhattan",
        "pinecone": "manhattan",
    }


class VectorDatabaseConfig:
    """
    Base class for vector database configurations.
    Each subclass must implement the `_module` class variable.
    """

    _module: ClassVar[str] = NotImplemented

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._module is NotImplemented:
            raise AssertionError(
                f"Subclass {cls.__name__} must implement the _module class variable"
            )


@dataclass
class IndexConfig:
    name: str
    vector_type: Type[Vector]
    index: Index = Index.HNSW
    metric: Metric = Metric.L2


class VectorDatabase:
    def __init__(
        self,
        config: VectorDatabaseConfig,
        embedder: Embedder,
        payload_schema: Type[BaseModel] = None,
        **kwargs,
    ):
        """
        Initialize a vanilla vector database. This is a base class for all vector databases. If you need more functionality from a specific vector database, you should use a specific subclass.

        Args:
            config: The configuration for the vector database
            embedder: The embedder to use for the vector database
            payload_schema: The Pydantic schema for validating context metadata
            **kwargs: Additional keyword arguments
        """
        self.config = config
        self.embedder = embedder
        self.payload_schema = payload_schema

        # CAVEAT: this loads a module from /integrations, which implements custom logic for a specific vector database provider.
        # CAVEAT It should be noted that some functions will raise NotImplementedErrors if the provider does not support the functionality.
        try:
            self.module = importlib.import_module(config._module)
        except ImportError as e:
            raise ImportError(f"""Unable to use the {config._module}, integration. Please make sure to install the module as an extra dependency for modaic.
                              You can install the module by running: pip install modaic[{config._module}]
                              OriginalError: {e}""")
        self.client = self.module._init(config)

    def drop_collection(self, collection_name: str):
        self.client.drop_collection(collection_name)

    def create_collection(
        self,
        collection_name: str,
        payload_schema: Optional[Type[BaseModel]] = None,
        embedding_dim: Optional[int] = None,
        index: Optional[Index] = None,
        metric: Optional[Metric] = None,
        exists_behavior: Literal["fail", "replace", "append"] = "replace",
    ):
        """
        Create a collection in the vector database.

        Args:
            collection_name: The name of the collection to create
            payload_schema: The schema of the collection
            exists_behavior: The behavior when the collection already exists
        """
        # Check if collection exists
        collection_exists = self.module.has_collection(self.client, collection_name)

        if collection_exists:
            if exists_behavior == "fail":
                raise ValueError(
                    f"Collection '{collection_name}' already exists and exists_behavior is set to 'fail'"
                )
            elif exists_behavior == "replace":
                self.module.drop_collection(self.client, collection_name)
            elif exists_behavior == "append":
                # If appending, just return as collection already exists
                return

        embedding_dim = (
            self.embedder.embedding_dim if embedding_dim is None else embedding_dim
        )
        payload_schema = (
            self.payload_schema if payload_schema is None else payload_schema
        )
        modaic_schema = pydantic_to_modaic_schema(payload_schema)
        # Create the collection
        self.module.create_collection(
            self.client, collection_name, modaic_schema, embedding_dim
        )

    def add_records(
        self,
        collection_name: str,
        records: Iterable[Context | Tuple[str, SerializedContext]],
        batch_size: Optional[int] = None,
    ):
        """
        Add items to a collection in the vector database.
        Uses the Context's get_embed_context() method and the embedder to create embeddings.

        Args:
            collection_name: The name of the collection to add records to
            records: The records to add to the collection
            batch_size: Optional batch size for processing records
        """
        if not records:
            return

        # Extract embed contexts from all items
        embedmes = []
        serialized_contexts = []

        for i, item in tqdm(
            enumerate(records), desc="Adding records to vector database"
        ):
            match item:
                case Context() as context:
                    embedme = context.embedme()
                    embedmes.append(embedme)
                    serialized_contexts.append(context.serialize())
                case (str() as embedme, SerializedContext() as serialized_context):
                    embedmes.append(embedme)
                    serialized_contexts.append(serialized_context)
                case _:
                    raise ValueError(
                        f"Unsupported VectorDatabase record format: {item}"
                    )

            if batch_size is not None and i % batch_size == 0:
                print("Adding chunk")
                self._embed_and_add_records(
                    collection_name, embedmes, serialized_contexts
                )
                embedmes = []
                serialized_contexts = []

        if embedmes:
            self._embed_and_add_records(collection_name, embedmes, serialized_contexts)

    def _embed_and_add_records(
        self,
        collection_name: str,
        embedmes: List[str],
        serialized_contexts: List[SerializedContext],
    ):
        print("Embedding records")
        try:
            embeddings = self.embedder(embedmes)

            # CAVEAT: Ensure embeddings is a 2D array (DSPy returns 1D for single strings, 2D for lists)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
        except Exception as e:
            raise ValueError(f"Failed to create embeddings: {e}")

        data_to_insert = []
        for i, (embedding, item) in tqdm(
            enumerate(zip(embeddings, serialized_contexts)),
            desc="Validating payloads",
        ):
            if (
                self.payload_schema is not None
                and type(item) is not self.payload_schema
            ):
                raise ValueError(
                    f"Expected item {i} to be a {self.payload_schema.__name__}, got {type(item)}"
                )

            # Create a record with embedding and validated metadata
            record = self.module._create_record(embedding, item)

            data_to_insert.append(record)
        print("Adding records to vector database (final call)")
        self.module.add_records(self.client, collection_name, data_to_insert)

    def search(
        self,
        collection_name: str,
        vector: np.ndarray | List[int],
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> List[SerializedContext]:
        """
        Retrieve records from the vector database.

        Args:
            collection_name: The name of the collection to search
            vector: The vector to search with
            k: The number of results to return
            filter: Optional filter to apply to the search

        Returns:
            List of serialized contexts matching the search.
        """
        return self.module.search(self.client, collection_name, vector, k, filter)

    def get_record(self, collection_name: str, record_id: str) -> SerializedContext:
        """
        Get a record from the vector database.

        Args:
            collection_name: The name of the collection
            record_id: The ID of the record to retrieve

        Returns:
            The serialized context record.
        """
        raise NotImplementedError(
            "get_record is not implemented for this vector database"
        )

    def query(
        self, query: str, k: int = 10, filter: Optional[dict] = None
    ) -> List[SerializedContext]:
        """
        Query the vector database.

        Args:
            query: The query string
            k: The number of results to return
            filter: Optional filter to apply to the query

        Returns:
            List of serialized contexts matching the query.
        """
        raise NotImplementedError("query is not implemented for this vector database")

    @staticmethod
    def print_available_functions(config_type: Type[VectorDatabaseConfig]):
        """
        Print the available functions for a given vector database configuration type.

        Args:
            config_type: The vector database configuration type to check
        """
        module = importlib.import_module(config_type._module)
        print(f"Available functions for {config_type._module} vector database:")
        for name in dir(module):
            if not name.startswith("_"):
                try:
                    inspect.signature(getattr(module, name))
                    print(f"- {name} (available) ✅")
                except NotImplementedError:
                    print(f"- {name} (Not implemented) ❌")


class InMemoryVectorDatabase(VectorDatabase):
    def __init__(
        self, embedder: dspy.Embedder, payload_schema: Type[BaseModel] = None, **kwargs
    ):
        from .integrations.milvus import MilvusVDBConfig

        in_memory_config = MilvusVDBConfig()
        super().__init__(in_memory_config, embedder, payload_schema, **kwargs)
        self.data_map = {}

    def add_records(
        self,
        collection_name: str,
        records: Iterable[Context | SerializedContext],
        batch_size: Optional[int] = None,
    ):
        serialized_records = []
        for record in records:
            serialized_record = record.serialize()
            serialized_records.append(serialized_record)
            self.data_map[serialized_record] = record
        super().add_records(collection_name, serialized_records)

    def retrieve(
        self, query: str, k: int = 10, filter: Optional[dict] = None
    ) -> List[Context]:
        pass
