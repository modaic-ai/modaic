# Table of Contents

* [vector\_database](#modaic.databases.vector_database)
  * [VectorType](#modaic.databases.vector_database.VectorType)
    * [FLOAT](#modaic.databases.vector_database.VectorType.FLOAT)
  * [IndexType](#modaic.databases.vector_database.IndexType)
  * [Metric](#modaic.databases.vector_database.Metric)
    * [\_init\_](#modaic.databases.vector_database.Metric._init_)
  * [VectorDatabaseConfig](#modaic.databases.vector_database.VectorDatabaseConfig)
  * [IndexConfig](#modaic.databases.vector_database.IndexConfig)
  * [VectorDatabase](#modaic.databases.vector_database.VectorDatabase)
    * [\_\_init\_\_](#modaic.databases.vector_database.VectorDatabase.__init__)
    * [load\_collection](#modaic.databases.vector_database.VectorDatabase.load_collection)
    * [create\_collection](#modaic.databases.vector_database.VectorDatabase.create_collection)
    * [add\_records](#modaic.databases.vector_database.VectorDatabase.add_records)
    * [search](#modaic.databases.vector_database.VectorDatabase.search)
    * [get\_record](#modaic.databases.vector_database.VectorDatabase.get_record)
    * [hybrid\_search](#modaic.databases.vector_database.VectorDatabase.hybrid_search)
    * [query](#modaic.databases.vector_database.VectorDatabase.query)
    * [print\_available\_functions](#modaic.databases.vector_database.VectorDatabase.print_available_functions)

<a id="modaic.databases.vector_database"></a>

# vector\_database

<a id="modaic.databases.vector_database.VectorType"></a>

## VectorType Objects

```python
class VectorType(AutoNumberEnum)
```

<a id="modaic.databases.vector_database.VectorType.FLOAT"></a>

#### FLOAT

float32

<a id="modaic.databases.vector_database.IndexType"></a>

## IndexType Objects

```python
class IndexType(AutoNumberEnum)
```

The ANN or ENN algorithm to use for an index. IndexType.DEFAULT is IndexType.HNSW for most vector databases (milvus, qdrant, mongo).

<a id="modaic.databases.vector_database.Metric"></a>

## Metric Objects

```python
class Metric(AutoNumberEnum)
```

<a id="modaic.databases.vector_database.Metric._init_"></a>

#### \_init\_

mapping of the library that supports the metric and the name the library uses to refer to it

<a id="modaic.databases.vector_database.VectorDatabaseConfig"></a>

## VectorDatabaseConfig Objects

```python
class VectorDatabaseConfig()
```

Base class for vector database configurations.
Each subclass must implement the `_module` class variable.

<a id="modaic.databases.vector_database.IndexConfig"></a>

## IndexConfig Objects

```python
@dataclass
class IndexConfig()
```

Configuration for a VDB index.

**Arguments**:

- `name` - The name of the index. For backends that support multiple indexes, this will also be the name of the vector field.
- `vector_type` - The type of vector used by the index.
- `index_type` - The type of index to use. see IndexType for available options.
- `metric` - The metric to use for the index. see Metric for available options.
- `embedder` - The embedder to use for the index. If not provided, will use the VectorDatabase's embedder.

<a id="modaic.databases.vector_database.VectorDatabase"></a>

## VectorDatabase Objects

```python
class VectorDatabase()
```

<a id="modaic.databases.vector_database.VectorDatabase.__init__"></a>

#### \_\_init\_\_

```python
def __init__(config: VectorDatabaseConfig,
             embedder: Optional[Embedder] = None,
             payload_schema: Type[BaseModel] = None,
             **kwargs)
```

Initialize a vanilla vector database. This is a base class for all vector databases. If you need more functionality from a specific vector database, you should use a specific subclass.

**Arguments**:

- `config` - The configuration for the vector database
- `embedder` - The embedder to use for the vector database
- `payload_schema` - The Pydantic schema for validating context metadata
- `**kwargs` - Additional keyword arguments

<a id="modaic.databases.vector_database.VectorDatabase.load_collection"></a>

#### load\_collection

```python
def load_collection(collection_name: str,
                    payload_schema: Type[BaseModel],
                    embedder: Optional[Embedder | Dict[str, Embedder]] = None)
```

Load collection information into the vector database.

**Arguments**:

- `collection_name` - The name of the collection to load
- `payload_schema` - The schema of the collection
- `index` - The index configuration for the collection

<a id="modaic.databases.vector_database.VectorDatabase.create_collection"></a>

#### create\_collection

```python
def create_collection(collection_name: str,
                      payload_schema: Type[BaseModel],
                      index: IndexConfig | List[IndexConfig] = IndexConfig(),
                      exists_behavior: Literal["fail", "replace",
                                               "append"] = "replace")
```

Create a collection in the vector database.

**Arguments**:

- `collection_name` - The name of the collection to create
- `payload_schema` - The schema of the collection
- `exists_behavior` - The behavior when the collection already exists

<a id="modaic.databases.vector_database.VectorDatabase.add_records"></a>

#### add\_records

```python
def add_records(collection_name: str,
                records: Iterable[Context | Tuple[str, ContextSchema]],
                batch_size: Optional[int] = None)
```

Add items to a collection in the vector database.
Uses the Context's get_embed_context() method and the embedder to create embeddings.

**Arguments**:

- `collection_name` - The name of the collection to add records to
- `records` - The records to add to the collection
- `batch_size` - Optional batch size for processing records

<a id="modaic.databases.vector_database.VectorDatabase.search"></a>

#### search

```python
def search(collection_name: str,
           vector: np.ndarray | List[int],
           k: int = 10,
           filter: Optional[dict] = None,
           index_name: Optional[str] = None) -> List[SearchResult]
```

Retrieve records from the vector database.
Returns a list of SearchResult dictionaries
SearchResult is a TypedDict with the following keys:
- id: The id of the record
- distance: The distance of the record
- context_schema: The serialized context of the record

**Arguments**:

- `collection_name` - The name of the collection to search
- `vector` - The vector to search with
- `k` - The number of results to return
- `filter` - Optional filter to apply to the search
  

**Returns**:

- `results` - List of SearchResult dictionaries matching the search.

<a id="modaic.databases.vector_database.VectorDatabase.get_record"></a>

#### get\_record

```python
def get_record(collection_name: str, record_id: str) -> ContextSchema
```

Get a record from the vector database.

**Arguments**:

- `collection_name` - The name of the collection
- `record_id` - The ID of the record to retrieve
  

**Returns**:

  The serialized context record.

<a id="modaic.databases.vector_database.VectorDatabase.hybrid_search"></a>

#### hybrid\_search

```python
def hybrid_search(collection_name: str,
                  vectors: List[np.ndarray],
                  index_names: List[str],
                  k: int = 10) -> List[ContextSchema]
```

Hybrid search the vector database.

<a id="modaic.databases.vector_database.VectorDatabase.query"></a>

#### query

```python
def query(query: str,
          k: int = 10,
          filter: Optional[dict] = None) -> List[ContextSchema]
```

Query the vector database.

**Arguments**:

- `query` - The query string
- `k` - The number of results to return
- `filter` - Optional filter to apply to the query
  

**Returns**:

  List of serialized contexts matching the query.

<a id="modaic.databases.vector_database.VectorDatabase.print_available_functions"></a>

#### print\_available\_functions

```python
@staticmethod
def print_available_functions(config_type: Type[VectorDatabaseConfig])
```

Print the available functions for a given vector database configuration type.

**Arguments**:

- `config_type` - The vector database configuration type to check

