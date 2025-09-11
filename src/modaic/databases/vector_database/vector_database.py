from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NoReturn,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    overload,
    runtime_checkable,
)

import immutables
import numpy as np
from aenum import AutoNumberEnum
from langchain_core.structured_query import Visitor
from more_itertools import peekable
from PIL import Image
from tqdm.auto import tqdm

from ... import Embedder
from ...context.base import Context, Embeddable
from ...observability import Trackable, track_modaic_obj
from ...query_language import Condition, parse_modaic_filter

DEFAULT_INDEX_NAME = "default"


class SearchResult(NamedTuple):
    id: str
    score: float
    context: Context


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


# TODO Make this support non-vector indexes like full-text search maybe?
@dataclass
class IndexConfig:
    """
    Configuration for a VDB index.

    Args:
        vector_type: The type of vector used by the index.
        index_type: The type of index to use. see IndexType for available options.
        metric: The metric to use for the index. see Metric for available options.
        embedder: The embedder to use for the index. If not provided, will use the VectorDatabase's embedder.
    """

    vector_type: Optional[VectorType] = VectorType.FLOAT
    index_type: Optional[IndexType] = IndexType.DEFAULT
    metric: Optional[Metric] = Metric.COSINE
    embedder: Optional[Embedder] = None


@dataclass
class CollectionConfig:
    payload_class: Type[Context]
    indexes: Dict[str, IndexConfig] = field(default_factory=dict)


TBackend = TypeVar("TBackend", bound="VectorDBBackend")


class VectorDatabase(Generic[TBackend], Trackable):
    ext: "VDBExtensions[TBackend]"
    collections: Dict[str, CollectionConfig]
    default_payload_class: Optional[Type[Context]] = None
    default_embedder: Optional[Embedder] = None

    def __init__(
        self,
        backend: TBackend,
        embedder: Optional[Embedder] = None,
        payload_class: Optional[Type[Context]] = None,
        **kwargs,
    ):
        """
        Initialize a vanilla vector database. This is a base class for all vector databases. If you need more functionality from a specific vector database, you should use a specific subclass.

        Args:
            config: The configuration for the vector database
            embedder: The embedder to use for the vector database
            payload_class: The default context class for collections
            **kwargs: Additional keyword arguments
        """

        Trackable.__init__(self, **kwargs)
        if isinstance(payload_class, type) and not issubclass(payload_class, Context):
            raise TypeError(f"payload_class must be a subclass of Context, got {payload_class}")

        self.ext = VDBExtensions(backend)
        self.collections = {}
        self.default_payload_class = payload_class
        self.default_embedder = embedder

    def drop_collection(self, collection_name: str):
        self.ext.backend.drop_collection(collection_name)

    # TODO: Signature looks good but some things about how the class will need to change to support this.
    def load_collection(
        self,
        collection_name: str,
        payload_class: Type[Context],
        embedder: Optional[Embedder | Dict[str, Embedder]] = None,
    ):
        """
        Load collection information into the vector database.
        Args:
            collection_name: The name of the collection to load
            payload_class: The context class of the context objects stored in the collection
            index: The index configuration for the collection
        """
        if not issubclass(payload_class, Context):
            raise TypeError(f"payload_class must be a subclass of Context, got {payload_class}")
        if not self.ext.backend.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist in the vector database")

        index_cfg = IndexConfig(
            vector_type=None,
            index_type=None,
            metric=None,
            embedder=embedder or self.default_embedder,
        )
        self.collections[collection_name] = CollectionConfig(
            indexes={DEFAULT_INDEX_NAME: index_cfg},
            payload_class=payload_class,
        )

    def create_collection(
        self,
        collection_name: str,
        payload_class: Type[Context],
        metric: Metric = Metric.COSINE,
        index_type: IndexType = IndexType.DEFAULT,
        vector_type: VectorType = VectorType.FLOAT,
        embedder: Optional[Embedder] = None,
        exists_behavior: Literal["fail", "replace"] = "replace",
    ):
        """
        Create a collection in the vector database.

        Args:
            collection_name: The name of the collection to create
            payload_class: The class of the context objects stored in the collection
            exists_behavior: The behavior when the collection already exists
        """
        if not issubclass(payload_class, Context):
            raise TypeError(f"payload_class must be a subclass of Context, got {payload_class}")
        collection_exists = self.ext.backend.has_collection(collection_name)

        if collection_exists:
            if exists_behavior == "fail":
                raise ValueError(
                    f"Collection '{collection_name}' already exists and exists_behavior is set to 'fail', if you would like ti load the collection instead use load_collection()"
                )
            elif exists_behavior == "replace":
                self.ext.backend.drop_collection(collection_name)

        index_cfg = IndexConfig(
            vector_type=vector_type,
            index_type=index_type,
            metric=metric,
            embedder=embedder or self.default_embedder,
        )
        self.collections[collection_name] = CollectionConfig(
            indexes={DEFAULT_INDEX_NAME: index_cfg},
            payload_class=payload_class,
        )

        self.ext.backend.create_collection(collection_name, payload_class, index_cfg)

    def list_collections(self) -> List[str]:
        return self.ext.backend.list_collections()

    def benchmark_add_records(
        self,
        collection_name: str,
        func: Callable,
        records: Iterable[Embeddable | Tuple[str | Image.Image, Context]],
        batch_size: Optional[int] = None,
        embedme_scope: Literal["auto", "context", "index"] = "auto",
    ):
        func(self, collection_name, records, batch_size, embedme_scope)

    def add_records(
        self,
        collection_name: str,
        records: Iterable[Embeddable | Tuple[str | Image.Image, Context]],
        batch_size: Optional[int] = None,
        embedme_scope: Literal["auto", "context", "index"] = "auto",
        tqdm_total: Optional[int] = None,
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

        # NOTE: Make embedmes compatible with the ext's hybrid search function
        if embedme_scope == "auto":
            if _items_have_multiple_embedmes(records):
                embedme_scope = "index"
            else:
                embedme_scope = "context"

        if embedme_scope == "index":
            embedmes: Dict[str, List[str | Image.Image]] = {
                k: [] for k in self.collections[collection_name].indexes.keys()
            }
        else:
            # CAVEAT: We make embedmes a dict with None as opposed to a list so we don't have to type check it
            embedmes: Dict[None, List[str | Image.Image]] = {None: []}

        serialized_contexts = []
        # TODO: add multi-processing/multi-threading here, just ensure that the backend is thread-safe. Maybe we add a class level parameter to check if the vendor is thread-safe. Embedding will still need to happen on a single thread
        for item in tqdm(
            records,
            desc="Adding records to vector database",
            disable=tqdm_total is None,
            total=tqdm_total or 0,
        ):
            cntxt = _add_ebedmes_and_return_context(embedmes, item)
            serialized_contexts.append(cntxt)

            if batch_size is not None and len(serialized_contexts) == batch_size:
                self._embed_and_add_records(collection_name, embedmes, serialized_contexts)
                if embedme_scope == "index":
                    embedmes = {k: [] for k in embedmes.keys()}
                else:
                    embedmes = {None: []}
                serialized_contexts = []

        if embedmes:
            self._embed_and_add_records(collection_name, embedmes, serialized_contexts)

    def has_collection(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the vector database.

        Args:
            collection_name: The name of the collection to check

        Returns:
            True if the collection exists, False otherwise
        """
        return self.ext.backend.has_collection(collection_name)

    def _embed_and_add_records(
        self,
        collection_name: str,
        embedmes: Dict[str, List[str | Image.Image]] | Dict[None, List[str | Image.Image]],
        contexts: List[Context],
    ):
        # TODO: could add functionality for multiple embedmes per context (e.g. you want to embed both an image and a text description of an image)
        all_embeddings = {}
        if collection_name not in self.collections:
            raise ValueError(
                f"Collection {collection_name} not found in VectorDatabase's collections, Please use VectorDatabase.create_collection() to create a collection first. Alternatively, you can use VectorDatabase.load_collection() to add records to an existing collection."
            )
        try:
            # NOTE: get embeddings for each index
            for index_name, index_config in self.collections[collection_name].indexes.items():
                # If dict is {None: embeddings} then we use the same embeddings for all indexes. Otherwise lookup embeddinsg for each index
                key = None if None in embedmes else index_name
                embeddings = index_config.embedder(embedmes[key])

                # NOTE: Ensure embeddings is a 2D array (DSPy returns 1D for single strings, 2D for lists)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)

                all_embeddings[index_name] = embeddings
        except Exception as e:
            raise ValueError(f"Failed to create embeddings for index: {index_name}") from e

        data_to_insert: List[immutables.Map[str, np.ndarray]] = []
        # FIXME Probably should add type checking to ensure context matches schema, not sure how to do this efficiently
        for i, item in enumerate(contexts):
            embedding_map: dict[str, np.ndarray] = {}
            for index_name, embeddings in all_embeddings.items():
                embedding_map[index_name] = embeddings[i]

            # Create a record with embedding and validated metadata
            record = self.ext.backend.create_record(embedding_map, item)

            data_to_insert.append(record)

        self.ext.backend.add_records(collection_name, data_to_insert)
        del data_to_insert

    # TODO: maybe better way of handling telling the integration module which Context class to return
    # TODO: add support for storage contexts. Where the payload is stored in a context and is mapped to the data via id
    # TODO: add support for multiple searches at once (i.e. accept a list of vectors)
    @track_modaic_obj
    def search(
        self,
        collection_name: str,
        query: str | Image.Image | List[str] | List[Image.Image],
        k: int = 10,
        filter: Optional[Condition] = None,
    ) -> List[List[SearchResult]]:
        """
        Retrieve records from the vector database.
        Returns a list of SearchResult dictionaries
        SearchResult is a NamedTuple with the following keys:
        - id: The id of the record
        - distance: The distance of the record
        - context: The context object (unhydrated if its hydratable)

        Args:
            collection_name: The name of the collection to search
            query: The vector to search with
            k: The number of results to return
            filter: Optional filter to apply to the search

        Returns:
            results: List of SearchResult dictionaries matching the search.

        Example:
            ```python
            results = vdb.search("collection 1", "How do I bake an apple pie?", k=10)
            print(results[0][0].context)
            >>> <Context: Text(text="apple pie recipe is 2 cups of flour, 1 cup of sugar, 1 cup of milk, 1 cup of eggs, 1 cup of butter")>
            ```

        """
        if filter is not None:
            filter = parse_modaic_filter(self.ext.backend.mql_translator, filter)
        indexes = self.collections[collection_name].indexes
        if len(indexes) > 1:
            raise ValueError(
                f"Collection {collection_name} has multiple indexes, please use VectorDatabase.ext.hybrid_search with an index_name"
            )
        query = [query] if isinstance(query, (str, Image.Image)) else query
        vectors = indexes[DEFAULT_INDEX_NAME].embedder(query)
        vectors = [vectors] if vectors.ndim == 1 else list(vectors)
        # CAVEAT: Allowing index_name to be None for libraries that don't care. Integration module should handle this behavior on their own.
        return self.ext.backend.search(
            collection_name,
            vectors,
            self.collections[collection_name].payload_class,
            k,
            filter,
        )

    def get_records(self, collection_name: str, record_id: List[str]) -> List[Context]:
        """
        Get a record from the vector database.

        Args:
            collection_name: The name of the collection
            record_id: The ID of the record to retrieve

        Returns:
            The serialized context record.
        """
        return self.ext.backend.get_records(collection_name, self.collections[collection_name].payload_class, record_id)

    def hybrid_search(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        index_names: List[str],
        k: int = 10,
    ) -> List[Context]:
        """
        Hybrid search the vector database.
        """
        raise NotImplementedError("hybrid_search is not implemented for this vector database")

    def query(self, query: str, k: int = 10, filter: Optional[dict] = None) -> List[Context]:
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

    def set_embedder(self, embedder: Embedder):
        self.default_embedder = embedder

    def upsert_records(self, collection_name: str, records: Iterable[Context]):
        """
        Upsert a record into the vector database.
        """
        raise NotImplementedError("upsert_record is not implemented for this vector database")

    def delete_records(self, collection_name: str, context_ids: Iterable[str]):
        """
        Delete a record from the vector database.
        """
        raise NotImplementedError("delete_record is not implemented for this vector database")


@runtime_checkable
class VectorDBBackend(Protocol):
    _name: ClassVar[str]
    _client: Any
    mql_translator: Visitor

    def __init__(self, *args, **kwargs) -> Any: ...
    def create_record(self, embedding_map: Dict[str, np.ndarray], context: Context) -> Any: ...
    def add_records(self, collection_name: str, records: List[Any]) -> None: ...
    def drop_collection(self, collection_name: str) -> None: ...
    def create_collection(
        self,
        collection_name: str,
        payload_class: Type[Context],
        index: IndexConfig = IndexConfig(),  # noqa: B008
    ) -> None: ...
    def list_collections(self) -> List[str]: ...
    def has_collection(self, collection_name: str) -> bool: ...
    def search(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        payload_class: Type[Context],
        k: int,
        filter: Optional[Any],  # Any the backend's native filtering language
    ) -> List[List[SearchResult]]: ...
    def get_records(
        self, collection_name: str, payload_class: Type[Context], record_ids: List[str]
    ) -> List[Context]: ...


COMMON_EXT = {
    "reindex",
}


@runtime_checkable
class SupportsBM25(VectorDBBackend, Protocol):
    def bm25_search(
        self,
        collection_name: str,
        query: str,
        k: int,
    ) -> List[Context]: ...
    def create_bm25_collection(
        self,
        collection_name: str,
        payload_class: Type[Context],
        exists_behavior: Literal["fail", "replace"] = "replace",
    ) -> List[Context]: ...
    def load_bm25_collection(
        self,
        collection_name: str,
        payload_class: Type[Context],
    ) -> List[Context]: ...


@runtime_checkable
class SupportsHybridSearch(VectorDBBackend, Protocol):
    def hybrid_search(
        self,
        collection_name: str,
        vectors: Dict[str, np.ndarray],
        k: int,
    ) -> List[Context]: ...
    def create_hybrid_collection(
        self,
        collection_name: str,
        payload_class: Type[Context],
        indexes: Dict[str, IndexConfig],
        exists_behavior: Literal["fail", "replace"] = "replace",
    ) -> List[Context]: ...
    def load_hybrid_collection(
        self,
        collection_name: str,
        payload_class: Type[Context],
        indexes: Dict[str, IndexConfig],
    ) -> List[Context]: ...


class VDBExtensions(Generic[TBackend]):
    backend: TBackend

    def __init__(self, backend: TBackend):
        self.backend = backend

    @property
    def client(self) -> Any:
        return self.backend._client

    # Use constrained TypeVars so intersection Protocols bind correctly
    TSupportsBM25 = TypeVar("TSupportsBM25", bound=SupportsBM25)
    TSupportsHybridSearch = TypeVar("TSupportsHybridSearch", bound=SupportsHybridSearch)

    @overload
    def hybrid_search(
        self: "VDBExtensions[TSupportsHybridSearch]",
        collection_name: str,
        vectors: Dict[str, np.ndarray],
        k: int,
    ) -> List[Context]: ...

    @overload
    def hybrid_search(
        self: "VDBExtensions[TBackend]",
        collection_name: str,
        vectors: Dict[str, np.ndarray],
        k: int,
    ) -> NoReturn: ...

    def hybrid_search(
        self: "VDBExtensions[TBackend]",
        collection_name: str,
        vectors: Dict[str, np.ndarray],
        k: int,
    ):
        if not isinstance(self.backend, SupportsHybridSearch):
            raise AttributeError(
                f"""{self.backend._name} does not support the function reindex.

                    Available functions: {self.available()}
                    """
            )
        return self.backend.hybrid_search(collection_name, vectors, k)

    @overload
    def bm25_search(
        self: "VDBExtensions[TSupportsBM25]",
        collection_name: str,
        vectors: List[np.ndarray],
        index_names: List[str],
        k: int,
    ) -> List[Context]: ...

    @overload
    def bm25_search(
        self: "VDBExtensions[TBackend]",
        collection_name: str,
        vectors: List[np.ndarray],
        index_names: List[str],
        k: int,
    ) -> NoReturn: ...

    def bm25_search(
        self: "VDBExtensions[TBackend]",
        collection_name: str,
        vectors: List[np.ndarray],
        index_names: List[str],
        k: int,
    ) -> List[Context]:
        if not isinstance(self.backend, SupportsBM25):
            raise AttributeError(
                f"""{self.backend._name} does not support the function hybrid_search.

                    Available functions: {self.available()}
                    """
            )
        return self.backend.hybrid_search(collection_name, vectors, index_names, k)

    @overload
    def create_hybrid_collection(
        self: "VDBExtensions[TSupportsHybridSearch]",
        query: str,
        k: int,
        filter: Optional[dict],
    ) -> List[Context]: ...

    @overload
    def create_hybrid_collection(
        self: "VDBExtensions[TBackend]", query: str, k: int, filter: Optional[dict]
    ) -> NoReturn: ...

    def create_hybrid_collection(
        self: "VDBExtensions[TBackend]", query: str, k: int, filter: Optional[dict]
    ) -> List[Context]:
        if not isinstance(self.backend, SupportsHybridSearch):
            raise AttributeError(
                f"""{self.backend._name} does not support the function query.

                    Available functions: {self.available()}
                    """
            )
        return self.backend.query(query, k, filter)

    def has(self, op: str) -> bool:
        fn = getattr(self, op, None)
        return callable(fn)

    def available(self) -> List[str]:
        return [op for op in COMMON_EXT if self.has(op)]


def _add_ebedmes_and_return_context(
    embedmes: Dict[str | None, List[str | Image.Image]],
    item: Embeddable | Tuple[str | Image.Image, Context],
) -> Context:
    """
    Adds all embedmes to the embedmes dictionary and returns the context.
    """
    # Fast type check for tuple
    if type(item) is tuple:
        embedme = item[0]
        for index in embedmes.keys():
            embedmes[index].append(embedme)
        return item[1]
    elif _has_multiple_embedmes(item):
        # CAVEAT: Context objects that implement Embeddable protocol and take in an index name as a parameter also accept None as the default index.
        for index in embedmes.keys():
            embedmes[index].append(item.embedme(index))
        return item
    else:
        for index in embedmes.keys():
            embedmes[index].append(item.embedme())
        return item


def _has_multiple_embedmes(
    item: Embeddable,
):
    """
    Check if the item has multiple embedmes.
    """
    return item.embedme.__code__.co_argcount == 2


def _items_have_multiple_embedmes(
    records: Iterable[Embeddable | Tuple[str | Image.Image, Context]],
):
    """
    Check if the first record has multiple embedmes.
    """
    p = peekable(records)
    first_item = p.peek()
    if isinstance(first_item, Embeddable) and _has_multiple_embedmes(first_item):
        return True
    return False
