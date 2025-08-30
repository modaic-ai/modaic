import pandas as pd
import warnings
import random
import re
import hashlib
from io import BytesIO
from .base import Source, SourceType, Molecular, ContextSchema
from ..storage.file_store import ContextStorage
from typing import Optional, Callable, ClassVar, Type
import duckdb
from contextlib import contextmanager
import os
from pathlib import Path

from .dtype_mapping import (
    INTEGER_DTYPE_MAPPING,
    FLOAT_DTYPE_MAPPING,
    OTHER_DTYPE_MAPPING,
    SPECIAL_INTEGER_DTYPE_MAPPING,
)


class TableSchema(ContextSchema):
    context_class: ClassVar[str] = "Table"
    name: str


class Table(Molecular):
    """
    A molecular context object that represents a table. Can be queried with SQL.
    """

    schema: ClassVar[Type[ContextSchema]] = TableSchema

    def __init__(
        self, df: pd.DataFrame, name: str, prepare_for_sql: bool = True, **kwargs
    ):
        """
        Initializes a Table context object.

        Args:
            df: The dataframe to represent as a table.
            name: The name of the table.
            prepare_for_sql: Whether to prepare the table for SQL queries.
            **kwargs: Additional keyword arguments to pass to the Molecular context object.
        """
        super().__init__(**kwargs)
        self._df = df
        self.name = name

        if prepare_for_sql:
            self.sanitize_columns()
            if not is_valid_table_name(name):
                self.name = Table.sanitize_name(name)
                warnings.warn(
                    f"Table name {name} is not a valid SQL table name and has been sanitized to {self.name}. To keep the original name, initialize with `prepare_for_sql=False`"
                )

    def get_sample_values(self, col: str):  # TODO: Rename and add docstring
        # TODO look up columnn
        series = self._df[col]

        valid_values = [
            x for x in series.dropna().unique() if pd.notnull(x) and len(str(x)) < 64
        ]
        sample_values = random.sample(valid_values, min(3, len(valid_values)))

        # Convert numpy types to Python native types for JSON serialization
        converted_values = []
        for val in sample_values:
            if hasattr(val, "item"):  # numpy types have .item() method
                converted_values.append(val.item())
            else:
                converted_values.append(val)

        return converted_values if converted_values else []

    def downcast_columns(self):  # TODO: Rename and add docstring
        self._df = self._df.apply(downcast_pd_series)

    def get_col(self, col_name: str) -> pd.Series:
        """
        Gets a single column from the table.

        Args:
            col_name: Name of the column to get

        Returns:
            The specified column as a pandas Series.
        """
        return self._df[col_name]

    def get_schema_with_samples(self):  # TODO; Rename and add docstring
        """
        Returns a dictionary of mapping column names to dictionaries containing the column type and sample values.

        Example:
            ```python
            >>> df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
            >>> table = Table(df, name="table")
            >>> table.get_schema_with_samples()
            {"Column1": {"type": "INT", "sample_values": [1, 2, 3]}, "Column2": {"type": "INT", "sample_values": [4, 5, 6]}, "Column3": {"type": "INT", "sample_values": [7, 8, 9]}}
            ```
        """
        column_dict = {}
        for col in self._df.columns:
            if isinstance(self._df[col], pd.DataFrame):
                print(f"Column {col} is a DataFrame, skipping...")
                raise ValueError(
                    f"Column {col} is a DataFrame, which is not supported."
                )
            column_dict[col] = {
                "type": pandas_to_mysql_dtype(self._df[col].dtype),
                "sample_values": self.get_sample_values(col),
            }

        return column_dict

    def schema_info(self):
        column_dict = self.get_schema_with_samples()

        schema_dict = {"table_name": self.name, "column_dict": column_dict}

        return schema_dict

    def sanitize_columns(self):
        columns = [sanitize_name(col) for col in self._df.columns]
        columns = [
            "No" if i == 0 and (not col or pd.isna(col)) else col
            for i, col in enumerate(columns)
        ]

        seen = {}
        new_columns = []
        for col in columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        self._df.columns = new_columns

    def query(self, query: str):  # TODO: add example
        """
        Queries the table. All queries run should refer to the table as `this` or `This`
        """
        return duckdb.query_df(self._df, "this", query).to_df()

    def markdown(self) -> str:  # TODO: add example
        """
        Converts the table to markdown format.
        Returns a markdown representation of the table with the table name as header.
        """
        content = ""
        content += f"Table name: {self.name}\n"

        # Add header row
        columns = [str(col) for col in self._df.columns]
        content += "| " + " | ".join(columns) + " |\n"

        # Add header separator
        content += "| " + " | ".join(["---"] * len(columns)) + " |\n"

        # Add data rows
        for _, row in self._df.iterrows():
            row_values = []
            for value in row:
                if pd.isna(value) or value is None:
                    row_values.append("")
                else:
                    row_values.append(str(value))
            content += "| " + " | ".join(row_values) + " |\n"

        return content

    def readme(self):
        """
        readme method for table. Returns a markdown representation of the table.

        Example:
            ```python
            >>> df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
            >>> table = Table(df, name="table")
            >>> table.readme()
            "Table name: table\n"
            " | Column1 | Column2 | Column3 | \n"
            " | --- | --- | --- | \n"
            " | 1 | 2 | 3 | \n"
            " | 4 | 5 | 6 | \n"
            " | 7 | 8 | 9 | \n"
            ```
        """
        return self.markdown()

    def embedme(self):
        """
        embedme method for table. Returns a markdown representation of the table.
        """
        return self.markdown()

    @staticmethod
    def sanitize_name(name: str) -> str:
        return sanitize_name(name)

    @classmethod
    def from_excel(
        cls,
        file: str | Path | BytesIO,
        name: Optional[str] = None,
        sheet_name: int | str = 0,
        metadata: dict = {},
        **kwargs,
    ):
        assert not isinstance(sheet_name, list) or sheet_name is not None, (
            "`Table` does not support multi-tabbed sheets, use `MultiTabbedTable` instead"
        )
        assert not isinstance(file, BytesIO) or name is not None, (
            "Name must be provided if reading from a BytesIO object instead of a file path"
        )
        df = pd.read_excel(file, sheet_name=sheet_name)

        if name is None and isinstance(file, BytesIO):
            xls = pd.ExcelFile(file)
            if len(xls.sheet_names) > 1:
                warnings.warn(
                    f"Multiple sheets found in {file}, using sheet {sheet_name}"
                )
            name = sanitize_name(xls.sheet_names[sheet_name])
        elif name is None:
            name = sanitize_name(os.path.basename(file))

        source = Source(
            file,
            SourceType.LOCAL_PATH if isinstance(file, str) else SourceType.LOCAL_PATH,
        )
        return cls(df, name=name, metadata=metadata, source=source, **kwargs)

    @classmethod
    def from_csv(
        cls,
        file: str | BytesIO,
        name: Optional[str] = None,
        metadata: dict = {},
        **kwargs,
    ):
        df = pd.read_csv(file)
        name = name or sanitize_name(file)
        source = Source(
            file,
            SourceType.LOCAL_PATH if isinstance(file, str) else SourceType.LOCAL_PATH,
        )
        return cls(df, name, metadata, source, **kwargs)


class MultiTabbedTableSchema(ContextSchema):
    context_class: ClassVar[str] = "MultiTabbedTable"
    tables: dict[str, TableSchema]


class MultiTabbedTable(Molecular):
    schema: ClassVar[Type[ContextSchema]] = MultiTabbedTableSchema

    def __init__(self, tables: dict[str, Table], **kwargs):
        super().__init__(**kwargs)
        self.tables = tables
        self.sql_db = None

    def __getitem__(self, key: str):
        return self.tables[key]

    def __setitem__(self, key: str, value: Table):
        self.tables[key] = value

    def __len__(self):
        return len(self.tables)

    def __iter__(self):
        return iter(self.tables)

    def __next__(self):
        return next(self.tables)

    def __contains__(self, key: str):
        return key in self.tables

    def init_sql(self):
        """
        Initilizes and in memory sql database for querying
        """
        self.sql_db = duckdb.connect(database=":memory:")
        for table_name, table in self.tables.items():
            self.sql_db.register(table_name, table.df)

    def close_sql(self):
        """
        Closes the in memory sql database
        """
        self.sql_db.close()
        self.sql_db = None

    @contextmanager
    def sql(self):
        self.init_sql()
        yield self.sql_db
        self.close_sql()

    def query(self, query: str):
        """
        Queries the in memory sql database
        """
        if self.sql_db is None:
            raise ValueError(
                "Attempted to run query on MultiTabbedTable without initializing the SQL database. Use with `with MultiTabbedTable.sql():` or `MultiTabbedTable.init_sql()`"
            )
        try:
            df = self.sql_db.execute(query).fetchdf()
            return Table(
                df=df, name="query_result", source=Source(self, SourceType.OBJECT)
            )
        except Exception as e:
            raise ValueError(f"Error querying SQL database: {e}")

    @classmethod
    def from_excel(
        cls,
        file: str | BytesIO,
        name: Optional[str] = None,
        metadata: dict = {},
        sheet_name: int | str | list[int | str] = None,
        **kwargs,
    ):
        df = pd.read_excel(file, sheet_name=sheet_name)
        name = name or sanitize_name(file)
        source = Source(
            file,
            SourceType.LOCAL_PATH if isinstance(file, str) else SourceType.LOCAL_PATH,
        )
        return cls(df, name, metadata, source, **kwargs)

    @classmethod
    def from_gsheet():
        raise NotImplementedError("Not implemented")

    @classmethod
    def from_sharepoint():
        raise NotImplementedError("Not implemented")

    @classmethod
    def from_s3():
        raise NotImplementedError("Not implemented")


def downcast_pd_series(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, downcast="integer")
    except ValueError:
        pass
    try:
        return pd.to_numeric(series, downcast="float")
    except ValueError:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(series)
    except ValueError:
        pass
    return series


def pandas_to_mysql_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        if str(dtype) in SPECIAL_INTEGER_DTYPE_MAPPING:
            return SPECIAL_INTEGER_DTYPE_MAPPING[str(dtype)]
        return INTEGER_DTYPE_MAPPING.get(dtype, "INT")

    elif pd.api.types.is_float_dtype(dtype):
        return FLOAT_DTYPE_MAPPING.get(dtype, "FLOAT")

    elif pd.api.types.is_bool_dtype(dtype):
        return OTHER_DTYPE_MAPPING["boolean"]

    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return OTHER_DTYPE_MAPPING["datetime"]

    elif pd.api.types.is_timedelta64_dtype(dtype):
        return OTHER_DTYPE_MAPPING["timedelta"]

    elif pd.api.types.is_string_dtype(dtype):
        return OTHER_DTYPE_MAPPING["string"]

    elif pd.api.types.is_categorical_dtype(dtype):
        return OTHER_DTYPE_MAPPING["category"]

    else:
        return OTHER_DTYPE_MAPPING["default"]


def sanitize_name(original_name: str) -> str:  # TODO: also sanitize SQL keywords
    """
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

    Args:
        original_name: The name to sanitize.

    Returns:
        The sanitized name.
    """
    # Remove file extension
    name = original_name.split(".")[0]

    # Replace illegal characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)

    # Remove leading/trailing underscores
    if len(name) > 2:
        name = name.strip("_")

    # Convert to lowercase
    name = name.lower()

    # Ensure name does not start with a number
    if name[0].isdigit():
        name = "t_" + name

    # If name is longer than 64 chars, truncate and add a hash suffix
    if len(name) > 64:
        prefix = name[:20].rstrip("_")
        hash_suffix = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        name = f"{prefix}_{hash_suffix}"

    return name


def is_valid_table_name(name: str) -> bool:
    """
    Checks if a name is a valid table name.

    Args:
        name: The name to validate.

    Returns:
        True if the name is valid, False otherwise.
    """
    valid = (
        name.islower()
        and not name.startswith("_")
        and not name.endswith("_")
        and not name[0].isdigit()
        and len(name) <= 64
    )
    return valid


if __name__ == "__main__":
    # table = Table.from_excel("/Users/tytodd/Desktop/Projects/DSTableRag/TableRAG/offline_data_ingestion_and_query_interface/dataset/hybridqa/dev_excel/Swiss_Super_League_0.xlsx")
    table = Table.from_csv("/Users/tytodd/Desktop/Projects/DSTableRag/test.csv")
    # print(table.df)
    # col = table.get_col("Weight", downcast=True)
    # print(col)
    # print(col.dtype)
    print(table.schema_info())
    # print(table.get_col("FC Basel"))
