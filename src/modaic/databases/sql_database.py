from sqlalchemy import (
    create_engine,
    MetaData,
    inspect,
    Table as SQLTable,
    Column,
    String,
    Text,
    text,
    CursorResult,
)
from sqlalchemy.orm import sessionmaker
from ..context.table import Table
from typing import Optional, Literal, List, Tuple, Iterable
import pandas as pd
from dataclasses import dataclass
import os
from tqdm import tqdm
from urllib.parse import urlencode
import json
from sqlalchemy.sql.compiler import IdentifierPreparer
from sqlalchemy.dialects import sqlite
from contextlib import contextmanager


@dataclass
class SQLDatabaseConfig:
    """
    Base class for SQL database configurations.
    Each subclass must implement the `url` property.
    """

    @property
    def url(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")


@dataclass
class SQLServerConfig(SQLDatabaseConfig):
    """
    Configuration for a SQL served over a port or remote connection. (MySQL, PostgreSQL, etc.)

    Args:
        user: The username to connect to the database.
        password: The password to connect to the database.
        host: The host of the database.
        database: The name of the database.
        port: The port of the database.
    """

    user: str
    password: str
    host: str
    database: str
    port: Optional[str] = None
    dialect: str = "mysql"
    driver: Optional[str] = None
    query_params: Optional[dict] = None

    @property
    def url(self) -> str:
        port = f":{self.port}" if self.port else ""
        driver = f"+{self.driver}" if self.driver else ""
        query = f"?{urlencode(self.query_params)}" if self.query_params else ""
        return f"{self.dialect}{driver}://{self.user}:{self.password}@{self.host}{port}/{self.database}{query}"


@dataclass
class SQLiteConfig(SQLDatabaseConfig):
    """
    Configuration for a SQLite database.

    Args:
        db_path: Path to the SQLite database file.
        in_memory: Whether to create an in-memory SQLite database.
        query_params: Query parameters to pass to the database.
    """

    db_path: Optional[str] = None
    in_memory: bool = False
    query_params: Optional[dict] = None

    @property
    def url(self) -> str:
        base = "sqlite:///:memory:" if self.in_memory else f"sqlite:///{self.db_path}"
        query = f"?{urlencode(self.query_params)}" if self.query_params else ""
        return f"{base}{query}"


class SQLDatabase:
    def __init__(
        self,
        config: SQLDatabaseConfig | str,
        engine_kwargs: dict = {},  # TODO: This may not be a smart idea, may want to enforce specific kwargs
        session_kwargs: dict = {},  # TODO: This may not be a smart idea
    ):
        self.url = config.url if isinstance(config, SQLDatabaseConfig) else config
        self.engine = create_engine(self.url, **engine_kwargs)
        self.metadata = MetaData()
        self.session = sessionmaker(bind=self.engine, **session_kwargs)
        self.inspector = inspect(self.engine)
        self.preparer = IdentifierPreparer(sqlite.dialect())

        # Create metadata table to store table metadata
        self.metadata_table = SQLTable(
            "metadata",
            self.metadata,
            Column("table_name", String(255), primary_key=True),
            Column("metadata_json", Text),
        )
        self.metadata.create_all(self.engine)
        self.connection = None
        self._in_transaction = False

    def add_table(
        self,
        table: Table,
        if_exists: Literal["fail", "replace", "append"] = "replace",
        schema: str = None,
    ):
        # TODO: support batch inserting for large dataframes
        with self.connect() as connection:
            # Use the connection for to_sql to respect transaction context
            table._df.to_sql(table.name, connection, if_exists=if_exists, index=False)

            # Remove existing metadata for this table if it exists
            connection.execute(
                self.metadata_table.delete().where(
                    self.metadata_table.c.table_name == table.name
                )
            )

            # Insert new metadata
            connection.execute(
                self.metadata_table.insert().values(
                    table_name=table.name,
                    metadata_json=json.dumps(table.metadata),
                )
            )
            if self._should_commit():
                connection.commit()

    def add_tables(
        self,
        tables: Iterable[Table],
        if_exists: Literal["fail", "replace", "append"] = "replace",
        schema: str = None,
    ):
        for table in tables:
            self.add_table(table, if_exists, schema)

    def drop_table(self, name: str, must_exist: bool = False):
        """
        Drop a table from the database and remove its metadata.

        Args:
            name: The name of the table to drop
        """
        if_exists = "IF EXISTS" if not must_exist else ""
        safe_name = self.preparer.quote(name)
        with self.connect() as connection:
            command = text(f"DROP TABLE {if_exists} {safe_name}")
            connection.execute(command)
            # Also remove metadata for this table
            connection.execute(
                self.metadata_table.delete().where(
                    self.metadata_table.c.table_name == name
                )
            )
            if self._should_commit():
                connection.commit()

    def drop_tables(self, names: Iterable[str], must_exist: bool = False):
        for name in names:
            self.drop_table(name, must_exist)

    def list_tables(self):
        """
        List all tables currently in the database.

        Returns:
            List of table names in the database.
        """
        # Refresh the inspector to ensure we get current table list
        self.inspector = inspect(self.engine)
        return self.inspector.get_table_names()

    def get_table(self, name: str) -> Table:
        df = pd.read_sql_table(name, self.engine)

        return Table(df, name=name, metadata=self.get_table_metadata(name))

    def get_table_schema(self, name: str):
        """
        Return column schema for a given table.

        Args:
            name: The name of the table to get schema for

        Returns:
            Column schema information for the table.
        """
        return self.inspector.get_columns(name)

    def get_table_metadata(self, name: str) -> dict:
        """
        Get metadata for a specific table.

        Args:
            name: The name of the table to get metadata for

        Returns:
            Dictionary containing the table's metadata, or empty dict if not found.
        """
        with self.connect() as connection:
            result = connection.execute(
                self.metadata_table.select().where(
                    self.metadata_table.c.table_name == name
                )
            ).fetchone()

        if result:
            return json.loads(result.metadata_json)
        return {}

    def query(self, query: str) -> CursorResult:
        with self.connect() as connection:
            result = connection.execute(text(query))
        return result

    def fetchall(self, query: str) -> List[Tuple]:
        result = self.query(query)
        return result.fetchall()

    def fetchone(self, query: str) -> Tuple:
        result = self.query(query)
        return result.fetchone()

    @classmethod
    def from_dir(cls, dir_path: str, config: SQLDatabaseConfig):
        # TODO: support batch inserting and parallel processing
        """
        Initializes a new SQLDatabase from a directory of files.

        Args:
            dir_path: Path to the directory containing files to load
            config: SQL database configuration

        Returns:
            New SQLDatabase instance loaded with data from the directory.
        """
        # TODO: make sure the loaded sql database is empty if not raise error and tell user to use __init__ for an already existing database
        instance = cls(config)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"File not found: {dir_path}")
        for file_name in tqdm(
            os.listdir(dir_path), desc="Uploading files to SQL database"
        ):
            full_path = os.path.join(dir_path, file_name)
            if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                table = Table.from_excel(full_path)
                instance.add_table(table, if_exists="fail")
            elif file_name.endswith(".csv"):
                table = Table.from_csv(full_path)
                instance.add_table(table, if_exists="fail")
        return instance

    @contextmanager
    def connect(self):
        """
        Context manager for database connections.
        Reuses existing connection if available, otherwise creates a temporary one.
        """
        connection_existed = self.connection is not None
        if not connection_existed:
            self.connection = self.engine.connect()

        try:
            yield self.connection
        finally:
            # Only close if we created the connection for this operation
            if not connection_existed:
                self.close()

    def open_persistent_connection(self):
        """
        Opens a persistent connection that will be reused across operations.
        Call close() to close the persistent connection.
        """
        if self.connection is None:
            self.connection = self.engine.connect()

    def close(self):
        """
        Closes the current connection if one exists.
        """
        if self.connection:
            self.connection.close()
            self.connection = None

    def _should_commit(self) -> bool:
        """
        Returns True if operations should commit immediately.
        Returns False if we're within an explicit transaction context.
        """
        return not self._in_transaction

    @contextmanager
    def begin(self):
        """
        Context manager for database transactions using existing connection.
        Requires an active connection. Commits on success, rolls back on exception.

        Raises:
            RuntimeError: If no active connection exists
        """
        if self.connection is None:
            raise RuntimeError(
                "No active connection. Use connect_and_begin() or open a connection first."
            )

        transaction = self.connection.begin()
        old_in_transaction = self._in_transaction
        self._in_transaction = True

        try:
            yield self.connection
            transaction.commit()
        except Exception:
            transaction.rollback()
            raise
        finally:
            self._in_transaction = old_in_transaction

    @contextmanager
    def connect_and_begin(self):
        """
        Context manager that establishes a connection and starts a transaction.
        Reuses existing connection if available, otherwise creates a temporary one.
        Commits on success, rolls back on exception.
        """
        connection_existed = self.connection is not None
        if not connection_existed:
            self.connection = self.engine.connect()

        transaction = self.connection.begin()
        old_in_transaction = self._in_transaction
        self._in_transaction = True

        try:
            yield self.connection
            transaction.commit()
        except Exception:
            transaction.rollback()
            raise
        finally:
            self._in_transaction = old_in_transaction
            # Only close if we created the connection for this operation
            if not connection_existed:
                self.close()


class MultiTenantSQLDatabase:
    def __init__(self):
        raise NotImplementedError("Not implemented")


# def parse_excel_file_and_insert_to_db(excel_file_outer_dir: str):
#     if not os.path.exists(excel_file_outer_dir):
#         raise FileNotFoundError(f"File not found: {excel_file_outer_dir}")


#     for file_name in tqdm(os.listdir(excel_file_outer_dir)):
#         if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
#             full_path = os.path.join(excel_file_outer_dir, file_name)
#             df = pd.read_excel(full_path)

#             df_convert = df.apply(infer_and_convert)
#             df_convert = transfer_df_columns(df_convert)

#             schema_dict, table_name = generate_schema_info(df_convert, file_name)

#             # 确保目录存在
#             if not os.path.exists(SCHEMA_DIR):
#                 os.makedirs(SCHEMA_DIR)

#             with open(f"{SCHEMA_DIR}/{table_name}.json", 'w', encoding='utf-8') as f:
#                 json.dump(schema_dict, f, ensure_ascii=False)

#             sql_alchemy_helper.insert_dataframe_batch(df_convert, table_name)
