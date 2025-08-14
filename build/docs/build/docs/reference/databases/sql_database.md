# Table of Contents

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

