from pydantic import BaseModel
from typing import (
    Optional,
    Type,
    Literal,
    List,
    ClassVar,
    Iterable,
    Tuple,
    Dict,
    Any,
    TypedDict,
    Protocol,
)
import dspy
from dataclasses import dataclass
from ...context.base import ContextSchema
import numpy as np
import importlib
import inspect
from tqdm.auto import tqdm
from ... import Embedder
from ...types import pydantic_to_modaic_schema
from aenum import AutoNumberEnum
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, Dict, Protocol, Any
from dataclasses import is_dataclass

# import psutil, os, time


class SearchResult(TypedDict):
    id: str
    distance: float
    context_schema: ContextSchema


# TODO: Add casting logic
class VectorType(AutoNumberEnum):
    _init_ = "supported_libraries"
    # name | supported_libraries
    FLOAT = ["milvus", "qdrant", "mongo", "pinecone"]  # float32
    FLOAT16 = ["milvus", "qdrant"]
    BFLOAT16 = ["milvus"]
    INT8 = ["milvus", "mongo"]
    UINT8 = ["qdrant"]
    BINARY = ["milvus", "mongo"]
    MULTI = ["qdrant"]
    FLOAT_SPARSE = ["milvus", "qdrant", "pinecone"]
    FLOAT16_SPARSE = ["qdrant"]
    INT8_SPARSE = ["qdrant"]


class IndexType(AutoNumberEnum):
    """
    The ANN or ENN algorithm to use for an index. IndexType.DEFAULT is IndexType.HNSW for most vector databases (milvus, qdrant, mongo).
    """

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
    _init_ = "supported_libraries"  # mapping of the library that supports the metric and the name the library uses to refer to it
    EUCLIDEAN = {
        "milvus": "L2",
        "qdrant": "Euclid",
        "mongo": "euclidean",
        "pinecone": "euclidean",
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
    }
    HAMMING = {"milvus": "HAMMING"}
    JACCARD = {"milvus": "JACCARD"}
    MHJACCARD = {"milvus": "MHJACCARD"}
    BM25 = {"milvus": "BM25"}


@dataclass
class IndexConfig:
    """
    Configuration for a VDB index.

    Args:
        name: The name of the index. For backends that support multiple indexes, this will also be the name of the vector field.
        vector_type: The type of vector used by the index.
        index_type: The type of index to use. see IndexType for available options.
        metric: The metric to use for the index. see Metric for available options.
        embedder: The embedder to use for the index. If not provided, will use the VectorDatabase's embedder.
    """

    name: str = "vector"
    vector_type: VectorType = VectorType.FLOAT
    index_type: IndexType = IndexType.DEFAULT
    metric: Metric = Metric.COSINE
    embedder: Optional[Embedder] = None


class VectorDatabase:
    def __init__(
        self,
        config: "VectorDBConfig",
        embedder: Optional[Embedder] = None,
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
        if not is_dataclass(config):
            raise TypeError("config must be a dataclass instance")
        self.config = config
        self.default_embedder = embedder
        self.payload_schema = payload_schema

        self._backend = self.config._backend_class(self.config)
        self.indexes = defaultdict(dict)
        self._schemas = {}  # collection_name -> modaic_schema

        self.ext = ExtHub(self._backend)

    def drop_collection(self, collection_name: str):
        self._backend.drop_collection(collection_name)

    # TODO: Signature looks good but some things about how the class will need to change to support this.
    def load_collection(
        self,
        collection_name: str,
        payload_schema: Type[BaseModel],
        embedder: Optional[Embedder | Dict[str, Embedder]] = None,
    ):
        """
        Load collection information into the vector database.
        Args:
            collection_name: The name of the collection to load
            payload_schema: The schema of the collection
            index: The index configuration for the collection
        """
        if not self._backend.has_collection(collection_name):
            raise ValueError(
                f"Collection {collection_name} does not exist in the vector database"
            )

        self._schemas[collection_name] = payload_schema

        raise NotImplementedError

    def create_collection(
        self,
        collection_name: str,
        payload_schema: Type[BaseModel],
        index: IndexConfig | List[IndexConfig] = IndexConfig(),
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
        collection_exists = self._backend.has_collection(collection_name)

        if collection_exists:
            if exists_behavior == "fail":
                raise ValueError(
                    f"Collection '{collection_name}' already exists and exists_behavior is set to 'fail'"
                )
            elif exists_behavior == "replace":
                self._backend.drop_collection(collection_name)

        self._schemas[collection_name] = payload_schema
        # payload_schema = (
        #     self.payload_schema if payload_schema is None else payload_schema
        # )
        modaic_schema = pydantic_to_modaic_schema(payload_schema)

        if isinstance(index, IndexConfig):
            index = [index]

        for index_config in index:
            if index_config.embedder is None and self.default_embedder is not None:
                index_config.embedder = self.default_embedder
            elif index_config.embedder is None:
                raise ValueError(
                    f"Failed to create collection: No embedder provided for index {index_config.name}"
                )

            self.indexes[collection_name][index_config.name] = index_config
        # TODO: Might want some more sophisticated logic here to check if the index already exists and has different parameters (diff vector type, index type, embedding dim, payload schema) This only really applies to milvus. and for now its okay to just let the backend SDKs handle this with their error messages.
        if (
            collection_exists and exists_behavior == "append"
        ):  # return before creating the collection
            return

        # Create the collection
        self._backend.create_collection(collection_name, modaic_schema, index)

    def add_records(
        self,
        collection_name: str,
        records: Iterable[ContextSchema | Tuple[str, ContextSchema]],
        batch_size: Optional[int] = None,
    ):
        """
        Add items to a collection in the vector database.
        Uses the ContextSchema's get_embed_context() method and the embedder to create embeddings.

        Args:
            collection_name: The name of the collection to add records to
            records: The records to add to the collection
            batch_size: Optional batch size for processing records
        """
        if not records:
            return
        # process = psutil.Process(os.getpid())

        # Extract embed contexts from all items
        embedmes = []
        serialized_contexts = []
        # TODO: add multi-processing here as well. Make sure that `_embed_and_add_records` runs on a single process though.
        with tqdm(
            total=len(records), desc="Adding records to vector database", position=0
        ) as pbar:
            for i, item in tqdm(
                enumerate(records),
                desc="Adding records to vector database",
                position=0,
                leave=False,
            ):
                match item:
                    case ContextSchema() as context:
                        embedme = context.embedme()
                        embedmes.append(embedme)
                        serialized_contexts.append(context.serialize())
                    case (str() as embedme, ContextSchema() as context_schema):
                        embedmes.append(embedme)
                        serialized_contexts.append(context_schema)
                    case _:
                        raise ValueError(
                            f"Unsupported VectorDatabase record format: {item}"
                        )

                if batch_size is not None and i % batch_size == 0:
                    self._embed_and_add_records(
                        collection_name, embedmes, serialized_contexts
                    )
                    embedmes = []
                    serialized_contexts = []
                    # mem = process.memory_info().rss / (1024 * 1024)
                    # pbar.set_postfix(mem=f"{mem:.2f} MB")
                    pbar.update(batch_size)

        if embedmes:
            self._embed_and_add_records(collection_name, embedmes, serialized_contexts)

    def _embed_and_add_records(
        self,
        collection_name: str,
        embedmes: List[str],
        serialized_contexts: List[ContextSchema],
    ):
        # print("Embedding records")
        # TODO: could add functionality for multiple embedmes per context (e.g. you want to embed both an image and a text description of an image)
        all_embeddings = {}
        assert collection_name in self.indexes, (
            f"Collection {collection_name} not found in VectorDatabase's indexes, Please use VectorDatabase.create_collection() to create a collection first. You can use VectorDatabase.create_collection() with exists_behavior='append' to add records to an existing collection."
        )
        try:
            for index_name, index_config in self.indexes[collection_name].items():
                embeddings = index_config.embedder(embedmes)

                # CAVEAT: Ensure embeddings is a 2D array (DSPy returns 1D for single strings, 2D for lists)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                all_embeddings[index_name] = embeddings
        except Exception as e:
            raise ValueError(
                f"Failed to create embeddings for index: {index_name}: {e}"
            )

        data_to_insert = []
        # TODO: add multi-processing here
        for i, item in enumerate(serialized_contexts):
            if (
                self.payload_schema is not None
                and type(item) is not self.payload_schema
            ):
                raise ValueError(
                    f"Expected item {i} to be a {self.payload_schema.__name__}, got {type(item)}"
                )
            embedding_map = {}
            for index_name, embedding in all_embeddings.items():
                embedding_map[index_name] = embedding[i]

            # Create a record with embedding and validated metadata
            record = self._backend.create_record(embedding_map, item)
            data_to_insert.append(record)

        self._backend.add_records(collection_name, data_to_insert)
        del data_to_insert

    # TODO: maybe better way of handling telling the integration module which ContextSchema class to return
    # TODO: add support for storage contexts. Where the payload is stored in a context and is mapped to the data via id
    # TODO: add support for multiple searches at once (i.e. accept a list of vectors)
    def search(
        self,
        collection_name: str,
        vector: np.ndarray | List[int],
        k: int = 10,
        filter: Optional[dict] = None,
        index_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Retrieve records from the vector database.
        Returns a list of SearchResult dictionaries
        SearchResult is a TypedDict with the following keys:
        - id: The id of the record
        - distance: The distance of the record
        - context_schema: The serialized context of the record

        Args:
            collection_name: The name of the collection to search
            vector: The vector to search with
            k: The number of results to return
            filter: Optional filter to apply to the search

        Returns:
            results: List of SearchResult dictionaries matching the search.

        """
        indexes = self.indexes[collection_name]
        if index_name is None:
            if len(indexes) > 1:
                raise ValueError(
                    f"Collection {collection_name} has multiple indexes, please specify which index to search with index_name"
                )
            elif len(indexes) == 1:
                index_name = list(indexes.keys())[0]
        # CAVEAT: Allowing index_name to be None for libraries that don't care. Integration module should handle this behavior on their own.
        return self._backend.search(
            collection_name,
            vector,
            self._schemas[collection_name],
            k,
            filter,
            index_name,
        )

    def get_record(self, collection_name: str, record_id: str) -> ContextSchema:
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

    def hybrid_search(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        index_names: List[str],
        k: int = 10,
    ) -> List[ContextSchema]:
        """
        Hybrid search the vector database.
        """
        raise NotImplementedError(
            "hybrid_search is not implemented for this vector database"
        )

    def query(
        self, query: str, k: int = 10, filter: Optional[dict] = None
    ) -> List[ContextSchema]:
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


class VectorDBConfig(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]]
    _backend_class: ClassVar[Type["VectorDBBackend"]]


class VectorDBBackend(Protocol):
    _name: ClassVar[str]
    _client: Any

    def __init__(self, config: VectorDBConfig) -> Any: ...
    def create_record(
        self, embedding_map: Dict[str, np.ndarray], scontext: ContextSchema
    ) -> Any: ...
    def add_records(self, collection_name: str, records: List[Any]) -> None: ...
    def drop_collection(self, collection_name: str) -> None: ...
    def create_collection(
        self,
        collection_name: str,
        payload_schema: Type[BaseModel],
        index: IndexConfig | List[IndexConfig] = IndexConfig(),
    ) -> None: ...
    def has_collection(self, collection_name: str) -> bool: ...
    def search(
        self,
        collection_name: str,
        vector: np.ndarray,
        payload_schema: Type[BaseModel],
        k: int,
        filter: Optional[dict],
        index_name: Optional[str],
    ) -> List[SearchResult]: ...
    def get_record(self, collection_name: str, record_id: str) -> ContextSchema: ...
    def hybrid_search(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        index_names: List[str],
        k: int,
    ) -> List[ContextSchema]: ...
    def query(
        self, query: str, k: int, filter: Optional[dict]
    ) -> List[ContextSchema]: ...
    def print_available_functions(self, config_type: Type[VectorDBConfig]) -> None: ...


COMMON_EXT = {
    "reindex",
}


class ExtHub:
    def __init__(self, backend: VectorDBBackend):
        self._backend = backend

    def __getattr__(self, name: str):
        if name == "client":
            return self._backend._client
        if name == self._backend._name:
            return self._backend
        if name in COMMON_EXT:
            try:
                return getattr(self._backend, name)
            except AttributeError:
                raise AttributeError(
                    f"""{self._backend._name} does not support the function {name}.
                    
                    Available functions: {self.available()}
                    """
                )
        raise AttributeError(f"VectorDatabase extension {name} not found")

    def has(self, op: str) -> bool:
        fn = getattr(self, op, None)
        return callable(fn)

    def available(self) -> List[str]:
        return [op for op in COMMON_EXT if self.has(op)]
