# Vector Database Integrations Guide
This is a guide for how to integrate a new vector database with Modaic. All vector databases are implemented as python modules in the /integrations. The VectorDatabase class will automatically load the correct module based ont he config class passed into its constructor. You will need to define each of the following for your integration interface.
## Concepts

### VectorType
This is an enum that defines the type of vector to use for an index. Most vector databases only support float32. When creating your integration be sure to update the enum's supported_libraries attribute to define which vector types are supported by your integration.

### IndexType
This is an enum that defines the type of index to use for an index. The default index type is IndexType.DEFAULT which is AUTOINDEX for milvus, and HNSW for most others (e.g.qdrant, mongo). Be sure to update the enum's supported_libraries attribute to define which index types are supported by your integration.

### Metric
This is an enum that defines the metric to use for an index. The default metric is Metric.COSINE. Be sure to update the enum's supported_libraries attribute to define which metrics are supported by your integration and map it to the name your integration uses to refer to that metric.

### IndexConfig
This is a dataclass that defines the configuration for an index. It has the following attributes:
```python
@dataclass
class IndexConfig:
    name: str = "vector"
    vector_type: VectorType = VectorType.FLOAT # The type of vector to use for the index.
    index_type: IndexType = IndexType.DEFAULT # The type of index to use for the index.
    metric: Metric = Metric.COSINE # The metric to use for the index.
    embedder: Optional[Embedder] = None # The embedder to use for the index. If not provided, will use the VectorDatabase's embedder.
```
You can access the dim of the expected embeddings via embedder.embedding_dim.

## Modaic Schema, SchemaField, and Modaic_Type
All modaic SerializedContexts are pydantic BaseModels under the hood. There are also extra types we have defined in the modaic.types module such as (int32, Array, and String) to make it easier to define the schema of a collection, ecspecially for libraries like milvus that optimize indexes based on a strict payload structure. VectorDatabase.create_collection() will automatically convert the pydantic BaseModel to a Modaic Schema which is a dictionary of field names and their SchemaField dataclasses using the modaic.types.pydantic_to_modaic_schema() function. The SchemaField dataclass has the following attributes:
```python
# All the types as strings that can be used in a SchemaField
Modaic_Type = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "bool",
    "String",
    "Array",
]

@dataclass
class InnerField:
    type: Modaic_Type # a string representing the type of the inner field see Modaic_Type above
    size: Optional[int] = None # the size of the inner field (if applicable) (e.g. String[100])


@dataclass
class SchemaField:
    type: Modaic_Type # a string representing the type of the field see Modaic_Type above
    optional: bool = False # whether the field is optional (i.e. can be None)
    size: Optional[int] = None # the size of the field (if applicable) (e.g. Array[int,100] or String[100])
    inner_type: Optional[InnerField] = None # the inner type of the field (if applicable) (e.g. for Array[String, 100] the inner_type would be String)
```
Note: for fields like Array and String the size may be None indicating the user just wants to use the max size. Your integration should handle this case.




## `VectorDatabaseConfig`
Make a subclass of VectorDatabaseConfig and set the _module class variable to the name of your module.
Example:
```python
@dataclass
class MilvusVDBConfig(VectorDatabaseConfig):
    _module: ClassVar[str] = "modaic.databases.integrations.milvus"

    uri: str = "http://localhost:19530"
    user: str = ""
    password: str = ""
    db_name: str = ""
    token: str = ""
    timeout: Optional[float] = None
    kwargs: dict = field(default_factory=dict)

    @staticmethod
    def from_local(file_path: str):
        return MilvusVDBConfig(uri=file_path)
```

## `_init(config: VectorDatabaseConfig) -> <Integration's Client Class>:`
This is the function that will be called to initialize the vector database client. It should accept the integration's config class as an argument and return the client object for your integration. This client object will be passed in to other functions in your integration module.
Example:
```python
def _init(config: MilvusVDBConfig) -> MilvusClient:
    """
    Initialize a Milvus vector database.
    """
    return MilvusClient(
        uri=config.uri,
        user=config.user,
        password=config.password,
        db_name=config.db_name,
        token=config.token,
        timeout=config.timeout,
        **config.kwargs,
    )
```

## `_create_record(embedding: np.ndarray, scontext: SerializedContext)-> Any:`
This fucntion will be used to turn embeddings and SerializedContexts into a format that can be added to your vector database. It can return any type of object just make sure that your definition of `add_records` accepts it.
Example:
```python
def _create_record(embedding: np.ndarray, scontext: SerializedContext) -> Any:
    """
    Convert a SerializedContext to a record for Milvus.
    """
    record = scontext.model_dump(mode="json")
    record["vector"] = embedding
    return record
```

## `add_records(client: <Integration's Client Class>, collection_name: str, records: List[Dict[str, Any]])`
This will be used to add records created by `_create_record` to your vector database in a specific collection. It should accept your integration's client object, the collection name, and a list of dictionaries of records. The keys of the dictionary will be the index names and the values will be the embedding for that index.
Example:
```python
def add_records(client: MilvusClient, collection_name: str, records: List[Any]):
    """
    Add records to a Milvus collection.
    """
    client.insert(collection_name, records)
```

## `create_collection(client: <Integration's Client Class>, collection_name: str, payload_schema: Dict[str, SchemaField], index: List[IndexConfig])`
This will be used to create a collection in your vector database. It should accept your integration's client object, the collection name, the payload schema, and a list of index configurations.
Its okay if your integration doesn't support multiple indexes. In that case raise an error stating "{integration_name} does not support multiple indexes" if there are multiple indexes in the list.


## `drop_collection(client: <Integration's Client Class>, collection_name: str)`
This will be used to drop a collection from your vector database. It should accept your integration's client object and the collection name.
Example:
```python
def drop_collection(client: MilvusClient, collection_name: str):
    """
    Drop a Milvus collection.
    """
    client.drop_collection(collection_name)
```

## `search(client: <Integration's Client Class>, collection_name: str, vector: np.ndarray, k: int, filter: Optional[Any] = None)`
This will be used to search for the k most similar records to a given vector in a specific collection. It should accept your integration's client object, the collection name, the vector to search for, the number of results to return, and an optional filter.
Note: we are currently working on a singular filter interface for all vector databases. For now you can expect that the filter should be compatible with the filter interface of your integration.
Example:
```python
def search(client: MilvusClient, collection_name: str, vector: np.ndarray, k: int, filter: Optional[str] = None, index_name: Optional[str] = None):
    """
    Search for the k most similar records to a given vector in a Milvus collection.
    """
    client.search(collection_name, vector, k, filter)
```
