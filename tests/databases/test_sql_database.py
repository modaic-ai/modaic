import os
import pathlib

import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.types import BIGINT

from modaic.context.table import Table
from modaic.databases.sql_database import SQLDatabase, SQLiteConfig

base_dir = pathlib.Path(__file__).parent


class TestSQLiteDatabase:
    @classmethod
    def setup_class(cls):
        os.remove(base_dir / "artifacts/test.db")
        cls.db = SQLDatabase(
            config=SQLiteConfig(
                db_path=base_dir / "artifacts/test.db",
            )
        )

    def test_add_table(self):
        table = Table(
            df=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}),
            name="test_table",
        )
        self.db.add_table(table)
        assert set(self.db.list_tables()) == {"test_table", "metadata"}
        assert self.db.get_table("test_table")._df.equals(table._df)

    def test_drop_table(self):
        table = Table(
            df=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}),
            name="drop_me",
        )
        self.db.add_table(table)
        assert "drop_me" in self.db.list_tables()
        self.db.drop_table("drop_me")
        assert "drop_me" not in self.db.list_tables()
        self.db.drop_table("drop_me", must_exist=False)
        with pytest.raises(sqlalchemy.exc.OperationalError):
            self.db.drop_table("drop_me", must_exist=True)

    def test_get_table_schema(self):
        table = Table(
            df=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}),
            name="test_table1",
        )
        self.db.add_table(table)
        schema = self.db.get_table_schema("test_table1")
        print(schema)

        expected_schema = [
            {
                "name": "column1",
                "type": BIGINT(),
                "nullable": True,
                "default": None,
                "primary_key": 0,
            },
            {
                "name": "column2",
                "type": BIGINT(),
                "nullable": True,
                "default": None,
                "primary_key": 0,
            },
        ]

        assert len(schema) == len(expected_schema)
        for i, (actual_col, expected_col) in enumerate(zip(schema, expected_schema)):
            assert actual_col["name"] == expected_col["name"]
            assert str(actual_col["type"]) == str(expected_col["type"])
            assert actual_col["nullable"] == expected_col["nullable"]
            assert actual_col["default"] == expected_col["default"]
            assert actual_col["primary_key"] == expected_col["primary_key"]

    def test_get_table_metadata(self):
        table = Table(
            df=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}),
            name="test_table2",
            metadata={"test": "i am some metadata"},
        )
        self.db.add_table(table)
        metadata = self.db.get_table_metadata("test_table2")
        assert metadata == {"test": "i am some metadata"}

    def test_query(self):
        table = Table(
            df=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}),
            name="test_table3",
        )
        self.db.add_table(table)
        result = self.db.query("SELECT * FROM test_table3")
        assert result.fetchall() == [(1, 4), (2, 5), (3, 6)]

    def test_begin_with_persistent_connection(self):
        """Test begin() context manager with an existing persistent connection."""
        # Setup table for testing
        table = Table(
            df=pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
            name="transaction_test",
        )
        self.db.add_table(table)

        # Open persistent connection
        self.db.open_persistent_connection()

        try:
            # Test successful transaction
            with self.db.begin() as conn:
                conn.execute(text("INSERT INTO transaction_test (id, value) VALUES (3, 30)"))

            # Verify the insert was committed
            result = self.db.query("SELECT COUNT(*) FROM transaction_test").fetchone()
            assert result[0] == 3

        finally:
            self.db.close()

    def test_begin_with_rollback(self):
        """Test begin() context manager with rollback on exception."""
        # Setup table for testing
        table = Table(
            df=pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
            name="rollback_test",
        )
        self.db.add_table(table)

        # Open persistent connection
        self.db.open_persistent_connection()

        try:
            # Test transaction rollback
            with pytest.raises(Exception):
                with self.db.begin() as conn:
                    conn.execute(text("INSERT INTO rollback_test (id, value) VALUES (4, 40)"))
                    # Force an exception to trigger rollback
                    raise Exception("Test rollback")

            # Verify the insert was rolled back
            result = self.db.query("SELECT COUNT(*) FROM rollback_test").fetchone()
            assert result[0] == 2  # Original count, insert was rolled back

        finally:
            self.db.close()

    def test_begin_without_connection(self):
        """Test begin() raises error when no connection exists."""
        # Ensure no connection exists
        self.db.close()

        with pytest.raises(RuntimeError, match="No active connection"):
            with self.db.begin():
                pass

    def test_connect_and_begin_success(self):
        """Test connect_and_begin() context manager - successful transaction."""
        # Setup table for testing
        table = Table(
            df=pd.DataFrame({"id": [1, 2], "value": [100, 200]}),
            name="connect_begin_test",
        )
        self.db.add_table(table)

        # Ensure no persistent connection
        self.db.close()

        # Test successful transaction with temporary connection
        with self.db.connect_and_begin() as conn:
            conn.execute(text("INSERT INTO connect_begin_test (id, value) VALUES (3, 300)"))

        # Verify the insert was committed and connection was closed
        assert self.db.connection is None
        result = self.db.query("SELECT COUNT(*) FROM connect_begin_test").fetchone()
        assert result[0] == 3

    def test_connect_and_begin_rollback(self):
        """Test connect_and_begin() context manager - rollback on exception."""
        # Setup table for testing
        table = Table(
            df=pd.DataFrame({"id": [1, 2], "value": [1000, 2000]}),
            name="connect_rollback_test",
        )
        self.db.add_table(table)

        # Ensure no persistent connection
        self.db.close()

        # Test transaction rollback with temporary connection
        with pytest.raises(Exception):
            with self.db.connect_and_begin() as conn:
                conn.execute(text("INSERT INTO connect_rollback_test (id, value) VALUES (4, 4000)"))
                # Force an exception to trigger rollback
                raise Exception("Test rollback")

        # Verify the insert was rolled back and connection was closed
        assert self.db.connection is None
        result = self.db.query("SELECT COUNT(*) FROM connect_rollback_test").fetchone()
        assert result[0] == 2  # Original count, insert was rolled back

    def test_connect_and_begin_with_persistent_connection(self):
        """Test connect_and_begin() reuses existing persistent connection."""
        # Setup table for testing
        table = Table(
            df=pd.DataFrame({"id": [1, 2], "value": [10000, 20000]}),
            name="persistent_begin_test",
        )
        self.db.add_table(table)

        # Open persistent connection
        self.db.open_persistent_connection()
        original_connection = self.db.connection

        try:
            # Test transaction reuses persistent connection
            with self.db.connect_and_begin() as conn:
                assert conn is original_connection
                conn.execute(text("INSERT INTO persistent_begin_test (id, value) VALUES (3, 30000)"))

            # Verify connection is still open and insert was committed
            assert self.db.connection is original_connection
            result = self.db.query("SELECT COUNT(*) FROM persistent_begin_test").fetchone()
            assert result[0] == 3

        finally:
            self.db.close()

    def test_should_commit_behavior(self):
        """Test _should_commit() returns correct values based on transaction state."""
        # Outside transaction context
        assert self.db._should_commit() == True

        # Within begin() context
        self.db.open_persistent_connection()
        try:
            with self.db.begin() as conn:
                assert self.db._should_commit() == False
        finally:
            self.db.close()

        # After transaction context
        assert self.db._should_commit() == True

        # Within connect_and_begin() context
        with self.db.connect_and_begin() as conn:
            assert self.db._should_commit() == False

        # After connect_and_begin() context
        assert self.db._should_commit() == True

    def test_data_operations_in_transaction(self):
        """Test that data operations respect transaction boundaries.
        Note: DDL operations (CREATE/DROP TABLE) cannot be rolled back in SQLite,
        but DML operations (INSERT/UPDATE/DELETE) can be.
        """
        # Setup a table for testing
        table = Table(
            df=pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
            name="data_tx_test",
        )
        self.db.add_table(table)

        # Test data rollback within transaction
        try:
            with self.db.connect_and_begin() as conn:
                # Insert data within transaction
                conn.execute(text("INSERT INTO data_tx_test (id, value) VALUES (3, 30)"))
                conn.execute(text("INSERT INTO data_tx_test (id, value) VALUES (4, 40)"))

                # Data should be visible within transaction
                result = conn.execute(text("SELECT COUNT(*) FROM data_tx_test")).fetchone()
                assert result[0] == 4

                # Force rollback
                raise Exception("Force rollback")

        except Exception:
            pass  # Expected

        # After rollback, original data should remain
        result = self.db.query("SELECT COUNT(*) FROM data_tx_test").fetchone()
        assert result[0] == 2  # Original data only

        # Clean up
        self.db.drop_table("data_tx_test")

    def test_add_table_outside_transaction_commits(self):
        """Test add_table commits immediately when not in transaction context."""
        # Ensure no persistent connection
        self.db.close()

        table = Table(
            df=pd.DataFrame({"id": [1, 2], "value": [100, 200]}),
            name="commit_test_table",
        )

        # Add table outside transaction context
        self.db.add_table(table)

        # Verify it's committed by checking in a new connection
        tables = self.db.list_tables()
        assert "commit_test_table" in tables

        # Clean up
        self.db.drop_table("commit_test_table")

    def test_drop_table_outside_transaction_commits(self):
        """Test drop_table commits immediately when not in transaction context."""
        # Setup - add a table first
        table = Table(
            df=pd.DataFrame({"id": [1, 2], "value": [300, 400]}),
            name="drop_commit_test",
        )
        self.db.add_table(table)
        assert "drop_commit_test" in self.db.list_tables()

        # Ensure no persistent connection
        self.db.close()

        # Drop table outside transaction context
        self.db.drop_table("drop_commit_test")

        # Verify it's committed by checking in a new connection
        tables = self.db.list_tables()
        assert "drop_commit_test" not in tables

    def test_transaction_flag_during_operations(self):
        """Test that _should_commit() works correctly during database operations."""
        # Verify flag behavior during add_table
        with self.db.connect_and_begin() as conn:
            # The flag should be set during the transaction
            original_should_commit = self.db._should_commit

            # Create a mock to verify _should_commit is called
            commit_calls = []

            def mock_should_commit():
                result = original_should_commit()
                commit_calls.append(result)
                return result

            self.db._should_commit = mock_should_commit

            try:
                # Add a table - should check _should_commit
                table = Table(
                    df=pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
                    name="flag_test_table",
                )
                self.db.add_table(table)

                # Verify _should_commit was called and returned False (in transaction)
                assert len(commit_calls) > 0
                assert all(call == False for call in commit_calls)

            finally:
                # Restore original method
                self.db._should_commit = original_should_commit

        # Clean up the table that was created (DDL operations persist even after rollback in SQLite)
        self.db.drop_table("flag_test_table")


class TestInMemoryDatabase:
    @classmethod
    def setup_class(cls):
        cls.db = SQLDatabase(
            config=SQLiteConfig(
                in_memory=True,
            )
        )


@pytest.mark.skip(reason="Not implemented")
class TestMysqlDatabase:
    @classmethod
    def setup_class(cls):
        pass


@pytest.mark.skip(reason="Not implemented")
class TestPostgresDatabase:
    @classmethod
    def setup_class(cls):
        pass
