# Table of Contents

* [callback](#modaic.callback)
* [directory\_database](#modaic.databases.directory_database)
  * [DirectoryDatabase](#modaic.databases.directory_database.DirectoryDatabase)
* [database](#modaic.databases.database)
  * [ContextDatabase](#modaic.databases.database.ContextDatabase)
  * [RAGDatabase](#modaic.databases.database.RAGDatabase)
* [databases](#modaic.databases)
* [bucket\_database](#modaic.databases.bucket_database)
  * [BucketDatabase](#modaic.databases.bucket_database.BucketDatabase)
* [sql\_database](#modaic.databases.sql_database)
  * [SQLDatabaseConfig](#modaic.databases.sql_database.SQLDatabaseConfig)
  * [SQLServerConfig](#modaic.databases.sql_database.SQLServerConfig)
  * [SQLiteConfig](#modaic.databases.sql_database.SQLiteConfig)
  * [SQLDatabase](#modaic.databases.sql_database.SQLDatabase)
    * [drop\_table](#modaic.databases.sql_database.SQLDatabase.drop_table)
    * [list\_tables](#modaic.databases.sql_database.SQLDatabase.list_tables)
    * [get\_table\_schema](#modaic.databases.sql_database.SQLDatabase.get_table_schema)
    * [get\_table\_metadata](#modaic.databases.sql_database.SQLDatabase.get_table_metadata)
    * [from\_dir](#modaic.databases.sql_database.SQLDatabase.from_dir)
    * [connect](#modaic.databases.sql_database.SQLDatabase.connect)
    * [open\_persistent\_connection](#modaic.databases.sql_database.SQLDatabase.open_persistent_connection)
    * [close](#modaic.databases.sql_database.SQLDatabase.close)
    * [begin](#modaic.databases.sql_database.SQLDatabase.begin)
    * [connect\_and\_begin](#modaic.databases.sql_database.SQLDatabase.connect_and_begin)
* [milvus](#modaic.databases.integrations.milvus)
  * [add\_records](#modaic.databases.integrations.milvus.add_records)
  * [drop\_collection](#modaic.databases.integrations.milvus.drop_collection)
  * [create\_collection](#modaic.databases.integrations.milvus.create_collection)
  * [has\_collection](#modaic.databases.integrations.milvus.has_collection)
  * [search](#modaic.databases.integrations.milvus.search)
* [pinecone](#modaic.databases.integrations.pinecone)
* [qdrant](#modaic.databases.integrations.qdrant)
* [mongodb](#modaic.databases.integrations.mongodb)
* [all](#modaic.databases.integrations.all)
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
* [graph\_database](#modaic.databases.graph_database)
  * [GraphDatabase](#modaic.databases.graph_database.GraphDatabase)
* [dtype\_mapping](#modaic.context.dtype_mapping)
* [context](#modaic.context)
* [text](#modaic.context.text)
  * [LongText](#modaic.context.text.LongText)
    * [chunk\_text](#modaic.context.text.LongText.chunk_text)
* [table](#modaic.context.table)
  * [Table](#modaic.context.table.Table)
    * [\_\_init\_\_](#modaic.context.table.Table.__init__)
    * [get\_col](#modaic.context.table.Table.get_col)
    * [get\_schema\_with\_samples](#modaic.context.table.Table.get_schema_with_samples)
    * [query](#modaic.context.table.Table.query)
    * [markdown](#modaic.context.table.Table.markdown)
    * [readme](#modaic.context.table.Table.readme)
    * [embedme](#modaic.context.table.Table.embedme)
  * [MultiTabbedTable](#modaic.context.table.MultiTabbedTable)
    * [init\_sql](#modaic.context.table.MultiTabbedTable.init_sql)
    * [close\_sql](#modaic.context.table.MultiTabbedTable.close_sql)
    * [query](#modaic.context.table.MultiTabbedTable.query)
  * [sanitize\_name](#modaic.context.table.sanitize_name)
  * [is\_valid\_table\_name](#modaic.context.table.is_valid_table_name)
* [base](#modaic.context.base)
  * [Source](#modaic.context.base.Source)
    * [model\_dump](#modaic.context.base.Source.model_dump)
    * [model\_dump\_json](#modaic.context.base.Source.model_dump_json)
  * [ContextSchema](#modaic.context.base.ContextSchema)
  * [Context](#modaic.context.base.Context)
    * [embedme](#modaic.context.base.Context.embedme)
    * [readme](#modaic.context.base.Context.readme)
    * [serialize](#modaic.context.base.Context.serialize)
    * [deserialize](#modaic.context.base.Context.deserialize)
    * [set\_source](#modaic.context.base.Context.set_source)
    * [set\_metadata](#modaic.context.base.Context.set_metadata)
    * [add\_metadata](#modaic.context.base.Context.add_metadata)
    * [from\_dict](#modaic.context.base.Context.from_dict)
  * [Atomic](#modaic.context.base.Atomic)
  * [Molecular](#modaic.context.base.Molecular)
    * [chunk\_with](#modaic.context.base.Molecular.chunk_with)
    * [apply\_to\_chunks](#modaic.context.base.Molecular.apply_to_chunks)
* [auto\_agent](#modaic.auto_agent)
  * [AutoConfig](#modaic.auto_agent.AutoConfig)
  * [AutoAgent](#modaic.auto_agent.AutoAgent)
    * [from\_precompiled](#modaic.auto_agent.AutoAgent.from_precompiled)
  * [git\_snapshot](#modaic.auto_agent.git_snapshot)
* [types](#modaic.types)
  * [Array](#modaic.types.Array)
  * [String](#modaic.types.String)
  * [unpack\_type](#modaic.types.unpack_type)
  * [pydantic\_to\_modaic\_schema](#modaic.types.pydantic_to_modaic_schema)
* [context\_store](#modaic.storage.context_store)
* [indexing](#modaic.indexing)
  * [Reranker](#modaic.indexing.Reranker)
    * [\_\_call\_\_](#modaic.indexing.Reranker.__call__)
  * [Embedder](#modaic.indexing.Embedder)
* [precompiled\_agent](#modaic.precompiled_agent)
  * [PrecompiledConfig](#modaic.precompiled_agent.PrecompiledConfig)
    * [save\_precompiled](#modaic.precompiled_agent.PrecompiledConfig.save_precompiled)
    * [from\_precompiled](#modaic.precompiled_agent.PrecompiledConfig.from_precompiled)
    * [from\_dict](#modaic.precompiled_agent.PrecompiledConfig.from_dict)
    * [from\_json](#modaic.precompiled_agent.PrecompiledConfig.from_json)
  * [PrecompiledAgent](#modaic.precompiled_agent.PrecompiledAgent)
    * [forward](#modaic.precompiled_agent.PrecompiledAgent.forward)
    * [save\_precompiled](#modaic.precompiled_agent.PrecompiledAgent.save_precompiled)
    * [from\_precompiled](#modaic.precompiled_agent.PrecompiledAgent.from_precompiled)
    * [push\_to\_hub](#modaic.precompiled_agent.PrecompiledAgent.push_to_hub)

<a id="modaic.callback"></a>

# callback

<a id="modaic.databases.directory_database"></a>

# directory\_database

<a id="modaic.databases.directory_database.DirectoryDatabase"></a>

## DirectoryDatabase Objects

```python
class DirectoryDatabase(ContextDatabase)
```

A database that stores context objects in a local file system directory. Not to be confused with the BucketDatabase in local mode.
This database is designed to be used in-place and in tandem with a user's local folder.

<a id="modaic.databases.database"></a>

# database

<a id="modaic.databases.database.ContextDatabase"></a>

## ContextDatabase Objects

```python
class ContextDatabase(ABC)
```

A database that can store context objects.

<a id="modaic.databases.database.RAGDatabase"></a>

## RAGDatabase Objects

```python
class RAGDatabase(ABC)
```

A database used for RAG

<a id="modaic.databases"></a>

# databases

<a id="modaic.databases.bucket_database"></a>

# bucket\_database

<a id="modaic.databases.bucket_database.BucketDatabase"></a>

## BucketDatabase Objects

```python
class BucketDatabase(ContextDatabase)
```

A database that stores context objects in a bucket like S3.

<a id="modaic.databases.sql_database"></a>

# sql\_database

<a id="modaic.databases.sql_database.SQLDatabaseConfig"></a>

## SQLDatabaseConfig Objects

```python
@dataclass
class SQLDatabaseConfig()
```

Base class for SQL database configurations.
Each subclass must implement the `url` property.

<a id="modaic.databases.sql_database.SQLServerConfig"></a>

## SQLServerConfig Objects

```python
@dataclass
class SQLServerConfig(SQLDatabaseConfig)
```

Configuration for a SQL served over a port or remote connection. (MySQL, PostgreSQL, etc.)

**Arguments**:

- `user` - The username to connect to the database.
- `password` - The password to connect to the database.
- `host` - The host of the database.
- `database` - The name of the database.
- `port` - The port of the database.

<a id="modaic.databases.sql_database.SQLiteConfig"></a>

## SQLiteConfig Objects

```python
@dataclass
class SQLiteConfig(SQLDatabaseConfig)
```

Configuration for a SQLite database.

**Arguments**:

- `db_path` - Path to the SQLite database file.
- `in_memory` - Whether to create an in-memory SQLite database.
- `query_params` - Query parameters to pass to the database.

<a id="modaic.databases.sql_database.SQLDatabase"></a>

## SQLDatabase Objects

```python
class SQLDatabase()
```

<a id="modaic.databases.sql_database.SQLDatabase.drop_table"></a>

#### drop\_table

```python
def drop_table(name: str, must_exist: bool = False)
```

Drop a table from the database and remove its metadata.

**Arguments**:

- `name` - The name of the table to drop

<a id="modaic.databases.sql_database.SQLDatabase.list_tables"></a>

#### list\_tables

```python
def list_tables()
```

List all tables currently in the database.

**Returns**:

  List of table names in the database.

<a id="modaic.databases.sql_database.SQLDatabase.get_table_schema"></a>

#### get\_table\_schema

```python
def get_table_schema(name: str)
```

Return column schema for a given table.

**Arguments**:

- `name` - The name of the table to get schema for
  

**Returns**:

  Column schema information for the table.

<a id="modaic.databases.sql_database.SQLDatabase.get_table_metadata"></a>

#### get\_table\_metadata

```python
def get_table_metadata(name: str) -> dict
```

Get metadata for a specific table.

**Arguments**:

- `name` - The name of the table to get metadata for
  

**Returns**:

  Dictionary containing the table's metadata, or empty dict if not found.

<a id="modaic.databases.sql_database.SQLDatabase.from_dir"></a>

#### from\_dir

```python
@classmethod
def from_dir(cls, dir_path: str, config: SQLDatabaseConfig)
```

Initializes a new SQLDatabase from a directory of files.

**Arguments**:

- `dir_path` - Path to the directory containing files to load
- `config` - SQL database configuration
  

**Returns**:

  New SQLDatabase instance loaded with data from the directory.

<a id="modaic.databases.sql_database.SQLDatabase.connect"></a>

#### connect

```python
@contextmanager
def connect()
```

Context manager for database connections.
Reuses existing connection if available, otherwise creates a temporary one.

<a id="modaic.databases.sql_database.SQLDatabase.open_persistent_connection"></a>

#### open\_persistent\_connection

```python
def open_persistent_connection()
```

Opens a persistent connection that will be reused across operations.
Call close() to close the persistent connection.

<a id="modaic.databases.sql_database.SQLDatabase.close"></a>

#### close

```python
def close()
```

Closes the current connection if one exists.

<a id="modaic.databases.sql_database.SQLDatabase.begin"></a>

#### begin

```python
@contextmanager
def begin()
```

Context manager for database transactions using existing connection.
Requires an active connection. Commits on success, rolls back on exception.

**Raises**:

- `RuntimeError` - If no active connection exists

<a id="modaic.databases.sql_database.SQLDatabase.connect_and_begin"></a>

#### connect\_and\_begin

```python
@contextmanager
def connect_and_begin()
```

Context manager that establishes a connection and starts a transaction.
Reuses existing connection if available, otherwise creates a temporary one.
Commits on success, rolls back on exception.

<a id="modaic.databases.integrations.milvus"></a>

# milvus

<a id="modaic.databases.integrations.milvus.add_records"></a>

#### add\_records

```python
def add_records(client: MilvusClient, collection_name: str,
                records: List[Any])
```

Add records to a Milvus collection.

<a id="modaic.databases.integrations.milvus.drop_collection"></a>

#### drop\_collection

```python
def drop_collection(client: MilvusClient, collection_name: str)
```

Drop a Milvus collection.

<a id="modaic.databases.integrations.milvus.create_collection"></a>

#### create\_collection

```python
def create_collection(client: MilvusClient,
                      collection_name: str,
                      payload_schema: Dict[str, SchemaField],
                      index: List[IndexConfig] = IndexConfig())
```

Create a Milvus collection.

<a id="modaic.databases.integrations.milvus.has_collection"></a>

#### has\_collection

```python
def has_collection(client: MilvusClient, collection_name: str) -> bool
```

Check if a collection exists in Milvus.

Params:
client: The Milvus client instance
collection_name: The name of the collection to check

**Returns**:

- `bool` - True if the collection exists, False otherwise

<a id="modaic.databases.integrations.milvus.search"></a>

#### search

```python
def search(client: MilvusClient,
           collection_name: str,
           vector: np.ndarray | List[int],
           payload_schema: Type[BaseModel],
           k: int = 10,
           filter: Optional[dict] = None,
           index_name: Optional[str] = None) -> List[SearchResult]
```

Retrieve records from the vector database.

<a id="modaic.databases.integrations.pinecone"></a>

# pinecone

<a id="modaic.databases.integrations.qdrant"></a>

# qdrant

<a id="modaic.databases.integrations.mongodb"></a>

# mongodb

<a id="modaic.databases.integrations.all"></a>

# all

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

<a id="modaic.databases.graph_database"></a>

# graph\_database

<a id="modaic.databases.graph_database.GraphDatabase"></a>

## GraphDatabase Objects

```python
class GraphDatabase(ContextDatabase)
```

A database that stores context objects in a graph database.

<a id="modaic.context.dtype_mapping"></a>

# dtype\_mapping

<a id="modaic.context"></a>

# context

<a id="modaic.context.text"></a>

# text

<a id="modaic.context.text.LongText"></a>

## LongText Objects

```python
class LongText(Molecular)
```

<a id="modaic.context.text.LongText.chunk_text"></a>

#### chunk\_text

```python
def chunk_text(
        chunk_fn: Callable[[str], List[str | tuple[str, dict]]]) -> List[Text]
```

Chunk the text into smaller Context objects.

**Arguments**:

- `chunk_fn` - A function that takes in a string and returns a list of strings or string-metadata pairs.
  

**Returns**:

  A list of Context objects.

<a id="modaic.context.table"></a>

# table

<a id="modaic.context.table.Table"></a>

## Table Objects

```python
class Table(Molecular)
```

A molecular context object that represents a table. Can be queried with SQL.

<a id="modaic.context.table.Table.__init__"></a>

#### \_\_init\_\_

```python
def __init__(df: pd.DataFrame,
             name: str,
             prepare_for_sql: bool = True,
             **kwargs)
```

Initializes a Table context object.

**Arguments**:

- `df` - The dataframe to represent as a table.
- `name` - The name of the table.
- `prepare_for_sql` - Whether to prepare the table for SQL queries.
- `**kwargs` - Additional keyword arguments to pass to the Molecular context object.

<a id="modaic.context.table.Table.get_col"></a>

#### get\_col

```python
def get_col(col_name: str) -> pd.Series
```

Gets a single column from the table.

**Arguments**:

- `col_name` - Name of the column to get
  

**Returns**:

  The specified column as a pandas Series.

<a id="modaic.context.table.Table.get_schema_with_samples"></a>

#### get\_schema\_with\_samples

```python
def get_schema_with_samples()
```

Returns a dictionary of mapping column names to dictionaries containing the column type and sample values.

**Example**:

    ```python
    >>> df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    >>> table = Table(df, name="table")
    >>> table.get_schema_with_samples()
    {"Column1": {"type": "INT", "sample_values": [1, 2, 3]}, "Column2": {"type": "INT", "sample_values": [4, 5, 6]}, "Column3": {"type": "INT", "sample_values": [7, 8, 9]}}
    ```

<a id="modaic.context.table.Table.query"></a>

#### query

```python
def query(query: str)
```

Queries the table. All queries run should refer to the table as `this` or `This`

<a id="modaic.context.table.Table.markdown"></a>

#### markdown

```python
def markdown() -> str
```

Converts the table to markdown format.
Returns a markdown representation of the table with the table name as header.

<a id="modaic.context.table.Table.readme"></a>

#### readme

```python
def readme()
```

readme method for table. Returns a markdown representation of the table.

**Example**:

            ```python
            >>> df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
            >>> table = Table(df, name="table")
            >>> table.readme()
            "Table name: table
"
            " | Column1 | Column2 | Column3 | 
"
            " | --- | --- | --- | 
"
            " | 1 | 2 | 3 | 
"
            " | 4 | 5 | 6 | 
"
            " | 7 | 8 | 9 | 
"
            ```

<a id="modaic.context.table.Table.embedme"></a>

#### embedme

```python
def embedme()
```

embedme method for table. Returns a markdown representation of the table.

<a id="modaic.context.table.MultiTabbedTable"></a>

## MultiTabbedTable Objects

```python
class MultiTabbedTable(Molecular)
```

<a id="modaic.context.table.MultiTabbedTable.init_sql"></a>

#### init\_sql

```python
def init_sql()
```

Initilizes and in memory sql database for querying

<a id="modaic.context.table.MultiTabbedTable.close_sql"></a>

#### close\_sql

```python
def close_sql()
```

Closes the in memory sql database

<a id="modaic.context.table.MultiTabbedTable.query"></a>

#### query

```python
def query(query: str)
```

Queries the in memory sql database

<a id="modaic.context.table.sanitize_name"></a>

#### sanitize\_name

```python
def sanitize_name(original_name: str) -> str
```

Sanitizes names of files and directories.

Rules:
1. Remove file extension
2. Replace illegal characters with underscores
3. Replace consecutive consecutive underscores/illegal charachters with a single underscore
4. Replace - with _
5. no caps
4. remove leading/trailing underscores
5. if name starts with a number, add t_
6. if name is longer than 64 chars, truncate and add a hash suffix

**Arguments**:

- `original_name` - The name to sanitize.
  

**Returns**:

  The sanitized name.

<a id="modaic.context.table.is_valid_table_name"></a>

#### is\_valid\_table\_name

```python
def is_valid_table_name(name: str) -> bool
```

Checks if a name is a valid table name.

**Arguments**:

- `name` - The name to validate.
  

**Returns**:

  True if the name is valid, False otherwise.

<a id="modaic.context.base"></a>

# base

<a id="modaic.context.base.Source"></a>

## Source Objects

```python
class Source(BaseModel)
```

<a id="modaic.context.base.Source.model_dump"></a>

#### model\_dump

```python
def model_dump(**kwargs)
```

Override model_dump method to exclude _parent field

<a id="modaic.context.base.Source.model_dump_json"></a>

#### model\_dump\_json

```python
def model_dump_json(**kwargs)
```

Override model_dump_json method to exclude _parent field

<a id="modaic.context.base.ContextSchema"></a>

## ContextSchema Objects

```python
class ContextSchema(BaseModel)
```

Base class used to define the schema of a context object when they are serialized.

**Attributes**:

- `context_class` - The class of the context object that this serialized context is for.
- `id` - The id of the serialized context.
- `source` - The source of the context object.
- `metadata` - The metadata of the context object.
  

**Example**:

  In this example, `CaptionImageSchema` stores the caption and the caption embedding while `CaptionedImage` is the `Context` class that is used to store the context object.
  Note that the image is loaded dynamically in the `CaptionedImage` class and is not serialized to `CaptionImageSchema`.
    ```python
    from modaic.context import ContextSchema
    from modaic.types import String, Vector, Float16Vector

    class CaptionImageSchema(ContextSchema):
        caption: String[100]
        caption_embedding: Float16Vector[384]
        image_path: String[100]

    class CaptionedImage(Atomic):
        schema = CaptionImageSchema

        def __init__(self, image_path: str, caption: str, caption_embedding: np.ndarray, **kwargs):
            super().__init__(**kwargs)
            self.caption = caption
            self.caption_embedding = caption_embedding
            self.image_path = image_path
            self.image = PIL.Image.open(image_path)

        def embedme(self) -> PIL.Image.Image:
            return self.image
    ```

<a id="modaic.context.base.Context"></a>

## Context Objects

```python
class Context(ABC)
```

<a id="modaic.context.base.Context.embedme"></a>

#### embedme

```python
@abstractmethod
def embedme() -> str | PIL.Image.Image
```

Abstract method defined by all subclasses of `Context` to define how embedding modeles should embed the context.

**Returns**:

  The string or image that should be used to embed the context.

<a id="modaic.context.base.Context.readme"></a>

#### readme

```python
def readme() -> str | pydantic.BaseModel
```

How LLMs should read the context. By default returns self.serialize()

**Returns**:

  LLM readable format of the context.

<a id="modaic.context.base.Context.serialize"></a>

#### serialize

```python
def serialize() -> ContextSchema
```

Serializes the context object into its associated `ContextSchema` object. Defined at self.schema.

**Returns**:

  The serialized context object.

<a id="modaic.context.base.Context.deserialize"></a>

#### deserialize

```python
@classmethod
def deserialize(cls, serialized: ContextSchema | dict, **kwargs)
```

Deserializes a `ContextSchema` object into a `Context` object.

**Arguments**:

- `serialized` - The serialized context object or a dict.
- `**kwargs` - Additional keyword arguments to pass to the Context object's constructor. (will overide any attributes set in the ContextSchema object)
  

**Returns**:

  The deserialized context object.

<a id="modaic.context.base.Context.set_source"></a>

#### set\_source

```python
def set_source(source: Source, copy: bool = False)
```

Sets the source of the context object.

**Arguments**:

- `source` - Source - The source of the context object.
- `copy` - bool - Whether to copy the source object to make it safe to mutate.

<a id="modaic.context.base.Context.set_metadata"></a>

#### set\_metadata

```python
def set_metadata(metadata: dict, copy: bool = False)
```

Sets the metadata of the context object.

**Arguments**:

- `metadata` - The metadata of the context object.
- `copy` - Whether to copy the metadata object to make it safe to mutate.

<a id="modaic.context.base.Context.add_metadata"></a>

#### add\_metadata

```python
def add_metadata(metadata: dict)
```

Adds metadata to the context object.

**Arguments**:

- `metadata` - The metadata to add to the context object.

<a id="modaic.context.base.Context.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, d: dict, **kwargs)
```

Deserializes a dict into a `Context` object.

**Arguments**:

- `d` - The dict to deserialize.
- `**kwargs` - Additional keyword arguments to pass to the Context object's constructor. (will overide any attributes set in the dict)
  

**Returns**:

  The deserialized context object.

<a id="modaic.context.base.Atomic"></a>

## Atomic Objects

```python
class Atomic(Context)
```

Base class for all Atomic Context objects. Atomic objects represent context at its finest granularity and are not chunkable.

**Example**:

  In this example, `CaptionedImage` is an `Atomic` context object that stores the caption and the caption embedding.
    ```python
    from modaic.context import ContextSchema
    from modaic.types import String, Vector, Float16Vector

    class CaptionImageSchema(ContextSchema):
        caption: String[100]
        caption_embedding: Float16Vector[384]
        image_path: String[100]

    class CaptionedImage(Atomic):
        schema = CaptionImageSchema

        def __init__(self, image_path: str, caption: str, caption_embedding: np.ndarray, **kwargs):
            super().__init__(**kwargs)
            self.caption = caption
            self.caption_embedding = caption_embedding
            self.image_path = image_path
            self.image = PIL.Image.open(image_path)

        def embedme(self) -> PIL.Image.Image:
            return self.image
    ```

<a id="modaic.context.base.Molecular"></a>

## Molecular Objects

```python
class Molecular(Context)
```

Base class for all `Molecular` Context objects. `Molecular` context objects represent context that can be chunked into smaller `Molecular` or `Atomic` context objects.

**Example**:

  In this example, `MarkdownDoc` is a `Molecular` context object that stores a markdown document.
    ```python
    from modaic.context import Molecular
    from modaic.types import String, Vector, Float16Vector
    from langchain_text_splitters import MarkdownTextSplitter
    from modaic.context import Text

    class MarkdownDocSchema(ContextSchema):
        markdown: String

    class MarkdownDoc(Molecular):
        schema = MarkdownDocSchema

        def chunk(self):
            # Split the markdown into chunks of 1000 characters
            splitter = MarkdownTextSplitter()
            chunk_fn = lambda mdoc: [Text(text=t) for t in splitter.split_text(mdoc.markdown)]
            self.chunk_with(chunk_fn)

        def __init__(self, markdown: str, **kwargs):
            super().__init__(**kwargs)
            self.markdown = markdown
    ```

<a id="modaic.context.base.Molecular.chunk_with"></a>

#### chunk\_with

```python
def chunk_with(chunk_fn: str | Callable[[Context], List[Context]],
               set_source: bool = True,
               **kwargs)
```

Chunk the context object into smaller Context objects.

**Arguments**:

- `chunk_fn` - The function to use to chunk the context object. The function should take in a specific type of Context object and return a list of Context objects.
- `set_source` - bool - Whether to automatically set the source of the chunks using the Context object. (sets chunk.source to self.source, sets chunk.source.parent to self, and updates the chunk.source.metadata with the chunk_id)
- `**kwargs` - dict - Additional keyword arguments to pass to the chunking function.

<a id="modaic.context.base.Molecular.apply_to_chunks"></a>

#### apply\_to\_chunks

```python
def apply_to_chunks(apply_fn: Callable[[Context], None], **kwargs)
```

Applies apply_fn to each chunk in chunks.

**Arguments**:

- `apply_fn` - The function to apply to each chunk. Function should take in a Context object and mutate it.
- `**kwargs` - Additional keyword arguments to pass to apply_fn.

<a id="modaic.auto_agent"></a>

# auto\_agent

<a id="modaic.auto_agent.AutoConfig"></a>

## AutoConfig Objects

```python
class AutoConfig()
```

Config for AutoAgent.

<a id="modaic.auto_agent.AutoAgent"></a>

## AutoAgent Objects

```python
class AutoAgent()
```

The AutoAgent class used to dynamically load agent frameworks at the given Modaic Hub path

<a id="modaic.auto_agent.AutoAgent.from_precompiled"></a>

#### from\_precompiled

```python
@staticmethod
def from_precompiled(repo_id, **kw)
```

Load a compiled agent from the given path. AutoAgent will automatically determine the correct Agent class.

**Arguments**:

- `repo_id` - The path to the compiled agent.
- `**kw` - Additional keyword arguments to pass to the Agent class.
  

**Returns**:

  An instance of the Agent class.

<a id="modaic.auto_agent.git_snapshot"></a>

#### git\_snapshot

```python
def git_snapshot(url: str, *, rev: str | None = "main") -> str
```

Clone / update a public Git repo into a local cache and return the path.

**Arguments**:

- `url` - Git repository URL (e.g., https://github.com/user/repo.git)
- `rev` - Branch, tag, or full commit SHA; default is 'main'
  

**Returns**:

  Path to the local cached repository.

<a id="modaic.types"></a>

# types

<a id="modaic.types.Array"></a>

## Array Objects

```python
class Array(List, metaclass=ArrayMeta)
```

Array field type for `ContextSchema`. Must be created with Array[dtype, max_size]

**Arguments**:

- `dtype` _Type_ - The type of the elements in the array.
- `max_size` _int_ - The maximum size of the array.
  

**Example**:

  A `EmailSchema` for `Email` context class that stores an email's content and recipients.
    ```python
    from modaic.types import Array
    from modaic.context import ContextSchema

    class EmailSchema(ContextSchema):
        content: str
        recipients: Array[str, 100]
    ```

<a id="modaic.types.String"></a>

## String Objects

```python
class String(str, metaclass=StringMeta)
```

String type that can be parameterized with max_length constraint.

**Arguments**:

- `max_size` _int_ - The maximum length of the string.
  

**Example**:

    ```python
    from modaic.types import String
    from modaic.context import ContextSchema

    class EmailSchema(ContextSchema
        subject: String[100]
        content: str
        recipients: Array[str, 100]
    ```

<a id="modaic.types.unpack_type"></a>

#### unpack\_type

```python
def unpack_type(field_type: Type) -> SchemaField
```

Unpacks a type into a compatible modaic schema field.
Modaic schema fields can be any of the following for type:
- Array
- Vector, Float16Vector, Float32Vector, Float64Vector, BFloat16Vector, BinaryVector
- String
- int8, int16, int32, int64, float32, float64, double(float64), bool, float(float64), int(int64)

The function will return a SchemaField dataclass with the following fields:

SchemaField - a dataclass with the following fields:
optional (bool): Whether the field is optional.
type (Type): The type of the field.
size (int | None): The size of the field.
inner_type (InnerField | None): The inner type of the field.

InnerField - a dataclass with the following fields:
type (Type): The type of the inner field.
size (int | None): The size of the inner field.

**Arguments**:

- `field_type` _Type_ - The type to unpack.
  

**Returns**:

  SchemaField - a dataclass containing information to serialize the type.

<a id="modaic.types.pydantic_to_modaic_schema"></a>

#### pydantic\_to\_modaic\_schema

```python
def pydantic_to_modaic_schema(
        pydantic_model: Type[BaseModel]) -> Dict[str, SchemaField]
```

Unpacks a type into a dictionary of compatible modaic schema fields.
Modaic schema fields can be any of the following for type:
- Array
- Vector, Float16Vector, Float32Vector, Float64Vector, BFloat16Vector, BinaryVector
- String
- int8, int16, int32, int64, float32, float64, double(float64), bool, float(float64), int(int64)

The function will return a dictionary mapping field names to SchemaField dataclasses.

SchemaField - a dataclass with the following fields:
optional (bool): Whether the field is optional.
type (Type): The type of the field.
size (int | None): The size of the field.
inner_type (InnerField | None): The inner type of the field.

InnerField is a dataclass with the following fields:
type (Type): The type of the inner field.
size (int | None): The size of the inner field.

**Arguments**:

- `pydantic_model` - The pydantic model to unpack.
  

**Returns**:

- `schema` - A dictionary mapping field names to SchemaField dataclasses.

<a id="modaic.storage.context_store"></a>

# context\_store

<a id="modaic.indexing"></a>

# indexing

<a id="modaic.indexing.Reranker"></a>

## Reranker Objects

```python
class Reranker(ABC)
```

<a id="modaic.indexing.Reranker.__call__"></a>

#### \_\_call\_\_

```python
def __call__(query: str,
             options: List[Context | Tuple[str, Context | ContextSchema]],
             k: int = 10,
             **kwargs) -> List[Tuple[Context | ContextSchema, float]]
```

Reranks the options based on the query.

**Arguments**:

- `query` - The query to rerank the options for.
- `options` - The options to rerank. Each option is a Context or tuple of (embedme_string, Context/ContextSchema).
- `k` - The number of options to return.
- `**kwargs` - Additional keyword arguments to pass to the reranker.
  

**Returns**:

  A list of tuples, where each tuple is (Context | ContextSchema, score). The Context or ContextSchema type depends on whichever was passed as an option for that index.

<a id="modaic.indexing.Embedder"></a>

## Embedder Objects

```python
class Embedder(dspy.Embedder)
```

A wrapper around dspy.Embedder that automatically determines the output size of the model.

<a id="modaic.precompiled_agent"></a>

# precompiled\_agent

<a id="modaic.precompiled_agent.PrecompiledConfig"></a>

## PrecompiledConfig Objects

```python
class PrecompiledConfig()
```

<a id="modaic.precompiled_agent.PrecompiledConfig.save_precompiled"></a>

#### save\_precompiled

```python
def save_precompiled(path: str) -> None
```

Saves the config to a config.json file in the given path.

**Arguments**:

- `path` - The path to save the config to.

<a id="modaic.precompiled_agent.PrecompiledConfig.from_precompiled"></a>

#### from\_precompiled

```python
@classmethod
def from_precompiled(cls, path: str) -> "PrecompiledConfig"
```

Loads the config from a config.json file in the given path.

**Arguments**:

- `path` - The path to load the config from.
  

**Returns**:

  An instance of the PrecompiledConfig class.

<a id="modaic.precompiled_agent.PrecompiledConfig.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, dict: Dict) -> "PrecompiledConfig"
```

Loads the config from a dictionary.

**Arguments**:

- `dict` - A dictionary containing the config.
  

**Returns**:

  An instance of the PrecompiledConfig class.

<a id="modaic.precompiled_agent.PrecompiledConfig.from_json"></a>

#### from\_json

```python
@classmethod
def from_json(cls, path: str) -> "PrecompiledConfig"
```

Loads the config from a json file.

**Arguments**:

- `path` - The path to load the config from.
  

**Returns**:

  An instance of the PrecompiledConfig class.

<a id="modaic.precompiled_agent.PrecompiledAgent"></a>

## PrecompiledAgent Objects

```python
class PrecompiledAgent(dspy.Module)
```

Bases: `dspy.Module`

<a id="modaic.precompiled_agent.PrecompiledAgent.forward"></a>

#### forward

```python
def forward(**kwargs) -> str
```

Forward pass for the agent.

**Arguments**:

- `**kwargs` - Additional keyword arguments.
  

**Returns**:

  Forward pass result.

<a id="modaic.precompiled_agent.PrecompiledAgent.save_precompiled"></a>

#### save\_precompiled

```python
def save_precompiled(path: str) -> None
```

Saves the agent and the config to the given path.

**Arguments**:

- `path` - The path to save the agent and config to. Must be a local path.

<a id="modaic.precompiled_agent.PrecompiledAgent.from_precompiled"></a>

#### from\_precompiled

```python
@classmethod
def from_precompiled(cls, path: str, **kwargs) -> "PrecompiledAgent"
```

Loads the agent and the config from the given path.

**Arguments**:

- `path` - The path to load the agent and config from. Can be a local path or a path on Modaic Hub.
- `**kwargs` - Additional keyword arguments.
  

**Returns**:

  An instance of the PrecompiledAgent class.

<a id="modaic.precompiled_agent.PrecompiledAgent.push_to_hub"></a>

#### push\_to\_hub

```python
def push_to_hub(repo_id: str) -> None
```

Pushes the agent and the config to the given repo_id.

**Arguments**:

- `repo_id` - The path on Modaic hub to save the agent and config to.

