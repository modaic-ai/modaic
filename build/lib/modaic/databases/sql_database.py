import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
from sqlalchemy import (
    JSON,
    Column,
    CursorResult,
    Index,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy import Table as SQLTable
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.compiler import IdentifierPreparer
from tqdm import tqdm

from ..context.table import BaseTable, Table, TableFile
from ..storage import FileStore


@dataclass
class SQLDatabaseBackend:
    """
    Base class for SQL database backends.
    Each subclass must implement the `url` property.
    """

    @property
    def url(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")


@dataclass
class SQLServerBackend(SQLDatabaseBackend):
    """
    Backend configuration for a SQL served over a port or remote connection. (MySQL, PostgreSQL, etc.)

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
class SQLiteBackend(SQLDatabaseBackend):
    """
    Backend configuration for a SQLite database.

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
    METADATA_TABLE_NAME = "modaic_metadata"

    def __init__(
        self,
        backend: SQLDatabaseBackend | str,
        engine_kwargs: dict = None,  # TODO: This may not be a smart idea, may want to enforce specific kwargs
        session_kwargs: dict = None,  # TODO: This may not be a smart idea, may want to enforce specific kwargs
        track_metadata: bool = False,
    ):
        self.url = backend.url if isinstance(backend, SQLDatabaseBackend) else backend
        self.engine = create_engine(self.url, **(engine_kwargs or {}))
        self.metadata = MetaData()
        self.session = sessionmaker(bind=self.engine, **(session_kwargs or {}))
        self.inspector = inspect(self.engine)
        self.preparer = self.engine.dialect.identifier_preparer

        # Create metadata table to store table metadata
        if track_metadata:
            self._ensure_metadata_table()
        self.metadata.reflect(bind=self.engine)
        self.metadata_table: Optional[Table] = (
            self.metadata.tables[self.METADATA_TABLE_NAME] if track_metadata else None
        )
        self.connection = None
        self._in_transaction = False

    def _ensure_metadata_table(self) -> None:
        """Create the metadata table if missing."""
        if not self.inspector.has_table(self.METADATA_TABLE_NAME):
            SQLTable(
                self.METADATA_TABLE_NAME,
                self.metadata,
                Column("table_name", String(255), primary_key=True),
                Column("metadata_json", Text),
            )
            self.metadata.create_all(self.engine)

    def add_table(
        self,
        table: BaseTable,
        if_exists: Literal["fail", "replace", "append"] = "replace",
        schema: str = None,
    ):
        # TODO: support batch inserting for large dataframes
        with self.connect() as connection:
            # Use the connection for to_sql to respect transaction context
            table._df.to_sql(table.name, connection, if_exists=if_exists, index=False)
            if self.metadata_table is not None:
                # Remove existing metadata for this table if it exists
                connection.execute(self.metadata_table.delete().where(self.metadata_table.c.table_name == table.name))

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
        tables: Iterable[BaseTable],
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
            if self.metadata_table is not None:
                connection.execute(self.metadata_table.delete().where(self.metadata_table.c.table_name == name))
            if self._should_commit():
                connection.commit()

    def drop_tables(self, names: Iterable[str], must_exist: bool = False):
        for name in names:
            self.drop_table(name, must_exist)

    def list_tables(self) -> List[str]:
        """
        List all tables currently in the database.

        Returns:
            List of table names in the database.
        """
        # Refresh the inspector to ensure we get current table list
        self.inspector = inspect(self.engine)
        return self.inspector.get_table_names()

    def get_table(self, name: str) -> BaseTable:
        df = pd.read_sql_table(name, self.engine)

        return Table(df=df, name=name, metadata=self.get_table_metadata(name))

    def get_table_schema(self, name: str) -> List[Column]:
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
        if self.metadata_table is None:
            raise ValueError(
                "Metadata table is not enabled. Please enable metadata tracking when initializing the SQLDatabase. with track_metadata=True."
            )
        with self.connect() as connection:
            result = connection.execute(
                self.metadata_table.select().where(self.metadata_table.c.table_name == name)
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
    def from_file_store(
        cls,
        file_store: FileStore,
        backend: SQLDatabaseBackend,
        folder: Optional[str] = None,
        table_created_hook: Optional[Callable[[TableFile], Any]] = None,
    ) -> "SQLDatabase":
        # TODO: support batch inserting and parallel processing
        """
        Initializes a new SQLDatabase from a file store.

        Args:
            file_store: File store containing files to load
            backend: SQL database backend
            folder: Folder in the file store to load

        Returns:
            New SQLDatabase instance loaded with data from the file store.
        """
        # TODO: make sure the loaded sql database is empty if not raise error and tell user to use __init__ for an already existing database
        instance = cls(backend)
        instance.add_file_store(file_store, folder, table_created_hook)
        return instance

    def add_file_store(
        self,
        file_store: FileStore,
        folder: Optional[str] = None,
        table_created_hook: Optional[Callable[[TableFile], Any]] = None,
    ):
        with self.begin():
            for key, _ in tqdm(file_store.items(folder), desc="Uploading files to SQL database"):
                table = TableFile.from_file_store(key, file_store)
                self.add_table(table, if_exists="fail")
                if table_created_hook:
                    table_created_hook(table)

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
            raise RuntimeError("No active connection. Use connect_and_begin() or open a connection first.")

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
