import os
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.types import BIGINT

from modaic.context.table import Table
from modaic.databases.sql_database import SQLDatabase, SQLiteBackend, SQLServerBackend


def _read_server_cfg() -> dict:
    """
    Read server SQL configuration from environment variables.

    Returns:
        dict: Configuration with required/optional fields.
    """
    return {
        "dialect": os.environ.get("SQL_DIALECT"),
        "driver": os.environ.get("SQL_DRIVER"),
        "host": os.environ.get("SQL_HOST"),
        "port": os.environ.get("SQL_PORT"),
        "database": os.environ.get("SQL_DATABASE"),
        "user": os.environ.get("SQL_USER"),
        "password": os.environ.get("SQL_PASSWORD"),
    }


@pytest.fixture(params=["sqlite", "server"])  # noqa: ANN001 ANN201
def sql_mode(request: pytest.FixtureRequest) -> str:
    """
    Provides SQL backend mode. Skips server if env is not configured.

    Params:
        request: pytest request

    Returns:
        str: Either "sqlite" or "server" (server may be skipped)
    """
    cfg = _read_server_cfg()
    if request.param == "server" and not all(
        [cfg["dialect"], cfg["host"], cfg["database"], cfg["user"], cfg["password"]]
    ):
        pytest.skip("No server SQL configured (set SQL_* environment variables).")
    return request.param


@pytest.fixture(scope="session")  # noqa: ANN001 ANN201
def sqlite_dbfile() -> str:
    """
    Create a SQLite db file under tests/artifacts; remove on teardown.

    Returns:
        str: Absolute path to SQLite database file
    """
    tests_dir = Path(__file__).resolve().parents[1]
    artifacts = tests_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    db_path = artifacts / "sqlite_test.db"
    try:
        if db_path.exists():
            db_path.unlink()
        yield str(db_path)
    finally:
        try:
            if db_path.exists():
                db_path.unlink()
        except Exception:
            pass


@pytest.fixture  # noqa: ANN001 ANN201
def sql_database(sql_mode: str, sqlite_dbfile: str) -> SQLDatabase:
    """
    Provide a configured SQLDatabase for tests.

    Params:
        sql_mode: Either "sqlite" or "server"
        sqlite_dbfile: Path for SQLite database file

    Returns:
        SQLDatabase: Initialized database connection
    """
    if sql_mode == "sqlite":
        return SQLDatabase(SQLiteBackend(db_path=sqlite_dbfile))

    cfg = _read_server_cfg()
    backend = SQLServerBackend(
        user=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        database=cfg["database"],
        port=cfg["port"],
        dialect=cfg["dialect"],
        driver=cfg["driver"],
    )
    # Smoke check to validate connectivity; skip on failure to avoid CI flakes
    try:
        db = SQLDatabase(backend)
        _ = db.list_tables()
        return db
    except Exception as e:
        pytest.skip(f"Server SQL connection failed: {e}")


def test_add_table(sql_database: SQLDatabase) -> None:
    table = Table(df=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}), name="test_table")
    sql_database.add_table(table)
    assert set(sql_database.list_tables()) == {"test_table", "metadata"}
    assert sql_database.get_table("test_table")._df.equals(table._df)


def test_drop_table(sql_database: SQLDatabase) -> None:
    table = Table(df=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}), name="drop_me")
    sql_database.add_table(table)
    assert "drop_me" in sql_database.list_tables()
    sql_database.drop_table("drop_me")
    assert "drop_me" not in sql_database.list_tables()
    sql_database.drop_table("drop_me", must_exist=False)
    with pytest.raises(sqlalchemy.exc.OperationalError):
        sql_database.drop_table("drop_me", must_exist=True)


def test_get_table_schema(sql_database: SQLDatabase) -> None:
    table = Table(df=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}), name="test_table1")
    sql_database.add_table(table)
    schema = sql_database.get_table_schema("test_table1")

    expected_schema = [
        {"name": "column1", "type": BIGINT(), "nullable": True, "default": None, "primary_key": 0},
        {"name": "column2", "type": BIGINT(), "nullable": True, "default": None, "primary_key": 0},
    ]

    assert len(schema) == len(expected_schema)
    for actual_col, expected_col in zip(schema, expected_schema, strict=False):
        assert actual_col["name"] == expected_col["name"]
        assert str(actual_col["type"]) == str(expected_col["type"])
        assert actual_col["nullable"] == expected_col["nullable"]
        assert actual_col["default"] == expected_col["default"]
        assert actual_col["primary_key"] == expected_col["primary_key"]


def test_get_table_metadata(sql_database: SQLDatabase) -> None:
    table = Table(
        df=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}),
        name="test_table2",
        metadata={"test": "i am some metadata"},
    )
    sql_database.add_table(table)
    metadata = sql_database.get_table_metadata("test_table2")
    assert metadata == {"test": "i am some metadata"}


def test_query(sql_database: SQLDatabase) -> None:
    table = Table(df=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}), name="test_table3")
    sql_database.add_table(table)
    result = sql_database.query("SELECT * FROM test_table3")
    assert result.fetchall() == [(1, 4), (2, 5), (3, 6)]


def test_begin_with_persistent_connection(sql_database: SQLDatabase) -> None:
    """Test begin() with an existing persistent connection."""
    table = Table(df=pd.DataFrame({"id": [1, 2], "value": [10, 20]}), name="transaction_test")
    sql_database.add_table(table)

    sql_database.open_persistent_connection()
    try:
        with sql_database.begin() as conn:
            conn.execute(text("INSERT INTO transaction_test (id, value) VALUES (3, 30)"))
        result = sql_database.query("SELECT COUNT(*) FROM transaction_test").fetchone()
        assert result[0] == 3
    finally:
        sql_database.close()


def test_begin_with_rollback(sql_database: SQLDatabase) -> None:
    """Test begin() rolls back on exception."""
    table = Table(df=pd.DataFrame({"id": [1, 2], "value": [10, 20]}), name="rollback_test")
    sql_database.add_table(table)

    sql_database.open_persistent_connection()
    try:
        with pytest.raises(Exception):
            with sql_database.begin() as conn:
                conn.execute(text("INSERT INTO rollback_test (id, value) VALUES (4, 40)"))
                raise Exception("Test rollback")
        result = sql_database.query("SELECT COUNT(*) FROM rollback_test").fetchone()
        assert result[0] == 2
    finally:
        sql_database.close()


def test_begin_without_connection(sql_database: SQLDatabase) -> None:
    """Test begin() raises when no connection exists."""
    sql_database.close()
    with pytest.raises(RuntimeError, match="No active connection"):
        with sql_database.begin():
            pass


def test_connect_and_begin_success(sql_database: SQLDatabase) -> None:
    """Test connect_and_begin() commits on success."""
    table = Table(df=pd.DataFrame({"id": [1, 2], "value": [100, 200]}), name="connect_begin_test")
    sql_database.add_table(table)
    sql_database.close()
    with sql_database.connect_and_begin() as conn:
        conn.execute(text("INSERT INTO connect_begin_test (id, value) VALUES (3, 300)"))
    assert sql_database.connection is None
    result = sql_database.query("SELECT COUNT(*) FROM connect_begin_test").fetchone()
    assert result[0] == 3


def test_connect_and_begin_rollback(sql_database: SQLDatabase) -> None:
    """Test connect_and_begin() rolls back on exception."""
    table = Table(df=pd.DataFrame({"id": [1, 2], "value": [1000, 2000]}), name="connect_rollback_test")
    sql_database.add_table(table)
    sql_database.close()
    with pytest.raises(Exception):
        with sql_database.connect_and_begin() as conn:
            conn.execute(text("INSERT INTO connect_rollback_test (id, value) VALUES (4, 4000)"))
            raise Exception("Test rollback")
    assert sql_database.connection is None
    result = sql_database.query("SELECT COUNT(*) FROM connect_rollback_test").fetchone()
    assert result[0] == 2


def test_connect_and_begin_with_persistent_connection(sql_database: SQLDatabase) -> None:
    """Test connect_and_begin() reuses existing persistent connection."""
    table = Table(df=pd.DataFrame({"id": [1, 2], "value": [10000, 20000]}), name="persistent_begin_test")
    sql_database.add_table(table)
    sql_database.open_persistent_connection()
    original_connection = sql_database.connection
    try:
        with sql_database.connect_and_begin() as conn:
            assert conn is original_connection
            conn.execute(text("INSERT INTO persistent_begin_test (id, value) VALUES (3, 30000)"))
        assert sql_database.connection is original_connection
        result = sql_database.query("SELECT COUNT(*) FROM persistent_begin_test").fetchone()
        assert result[0] == 3
    finally:
        sql_database.close()


def test_should_commit_behavior(sql_database: SQLDatabase) -> None:
    """Test _should_commit() flag across contexts."""
    assert sql_database._should_commit() is True
    sql_database.open_persistent_connection()
    try:
        with sql_database.begin():
            assert sql_database._should_commit() is False
    finally:
        sql_database.close()
    assert sql_database._should_commit() is True
    with sql_database.connect_and_begin():
        assert sql_database._should_commit() is False
    assert sql_database._should_commit() is True


def test_data_operations_in_transaction(sql_database: SQLDatabase) -> None:
    """DML operations rollback; SQLite DDL does not rollback."""
    table = Table(df=pd.DataFrame({"id": [1, 2], "value": [10, 20]}), name="data_tx_test")
    sql_database.add_table(table)
    try:
        with sql_database.connect_and_begin() as conn:
            conn.execute(text("INSERT INTO data_tx_test (id, value) VALUES (3, 30)"))
            conn.execute(text("INSERT INTO data_tx_test (id, value) VALUES (4, 40)"))
            result = conn.execute(text("SELECT COUNT(*) FROM data_tx_test")).fetchone()
            assert result[0] == 4
            raise Exception("Force rollback")
    except Exception:
        pass
    result = sql_database.query("SELECT COUNT(*) FROM data_tx_test").fetchone()
    assert result[0] == 2
    sql_database.drop_table("data_tx_test")


def test_add_table_outside_transaction_commits(sql_database: SQLDatabase) -> None:
    """add_table commits immediately when not in a transaction context."""
    sql_database.close()
    table = Table(df=pd.DataFrame({"id": [1, 2], "value": [100, 200]}), name="commit_test_table")
    sql_database.add_table(table)
    tables = sql_database.list_tables()
    assert "commit_test_table" in tables
    sql_database.drop_table("commit_test_table")


def test_drop_table_outside_transaction_commits(sql_database: SQLDatabase) -> None:
    """drop_table commits immediately when not in a transaction context."""
    table = Table(df=pd.DataFrame({"id": [1, 2], "value": [300, 400]}), name="drop_commit_test")
    sql_database.add_table(table)
    assert "drop_commit_test" in sql_database.list_tables()
    sql_database.close()
    sql_database.drop_table("drop_commit_test")
    tables = sql_database.list_tables()
    assert "drop_commit_test" not in tables


def test_transaction_flag_during_operations(sql_database: SQLDatabase) -> None:
    """_should_commit() is consulted during operations within a transaction."""
    with sql_database.connect_and_begin():
        original_should_commit = sql_database._should_commit
        commit_calls: list[bool] = []

        def mock_should_commit() -> bool:
            result = original_should_commit()
            commit_calls.append(result)
            return result

        sql_database._should_commit = mock_should_commit  # type: ignore[method-assign]
        try:
            table = Table(df=pd.DataFrame({"id": [1, 2], "value": [10, 20]}), name="flag_test_table")
            sql_database.add_table(table)
            assert len(commit_calls) > 0
            assert all(call is False for call in commit_calls)
        finally:
            sql_database._should_commit = original_should_commit  # type: ignore[assignment]
    sql_database.drop_table("flag_test_table")
