import hashlib
import random
import re
import warnings
from abc import ABC
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Dict, List, Literal, Optional, Set

import duckdb
import numpy as np
import pandas as pd
from pydantic import PrivateAttr, ValidatorFunctionWrapHandler, field_validator, model_validator

from modaic.context.base import Context, HydratedAttr, requires_hydration
from modaic.types import Field

from ..storage.file_store import FileStore
from .dtype_mapping import (
    FLOAT_DTYPE_MAPPING,
    INTEGER_DTYPE_MAPPING,
    OTHER_DTYPE_MAPPING,
    SPECIAL_INTEGER_DTYPE_MAPPING,
)
from .text import Text


class BaseTable(Context, ABC):
    name: str
    _df: pd.DataFrame = PrivateAttr()

    @field_validator("name", mode="before")
    @classmethod
    def sanitize_name(cls, name: str) -> str:
        return sanitize_name(name)

    def column_samples(self, col: str) -> list[Any]:
        """
        Return up to 3 distinct sample values from the given column.

        Picks at most three unique, non-null, short (&lt;64 chars) values from
        the column, favoring speed by sampling after de-duplicating values.

        Args:
            col: Column name to sample from.

        Returns:
            A list with up to three JSON-serializable sample values.
        """
        # TODO look up columnn
        series = self._df[col]

        valid_values = [x for x in series.dropna().unique() if pd.notnull(x) and len(str(x)) < 64]
        sample_values = random.sample(valid_values, min(3, len(valid_values)))

        # Convert numpy types to Python native types for JSON serialization
        converted_values = []
        for val in sample_values:
            if hasattr(val, "item"):  # numpy types have .item() method
                converted_values.append(val.item())
            else:
                converted_values.append(val)

        return converted_values if converted_values else []

    def get_col(self, col_name: str) -> pd.Series:
        """
        Gets a single column from the table.

        Args:
            col_name: Name of the column to get

        Returns:
            The specified column as a pandas Series.
        """
        return self._df[col_name]

    def schema_info(self) -> dict[str, Any]:
        column_dict = {}
        for col in self._df.columns:
            if isinstance(self._df[col], pd.DataFrame):
                raise ValueError(f"Column {col} is a DataFrame, which is not supported.")
            column_dict[col] = {
                "type": _pandas_to_mysql_dtype(self._df[col].dtype),
                "sample_values": self.column_samples(col),
            }

        schema_dict = {"table_name": self.name, "column_dict": column_dict}

        return schema_dict

    def query(self, query: str) -> pd.DataFrame:
        """
        Queries the table using DuckDB SQL.

        Notes:
        - Refer to the in-memory table as `this` (alias `This`).

        Example:
        ```python
        # Select a few rows
        df = table.query("SELECT * FROM this LIMIT 5")

        # Aggregate over a column
        df = table.query("SELECT category, COUNT(*) AS n FROM this GROUP BY category")
        ```
        """
        return duckdb.query_df(self._df, self.name, query).to_df()

    def embedme(self) -> str:
        """
        embedme method for table. Returns a markdown representation of the table.
        """
        return self.markdown()

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

    def to_text(self) -> Text:
        """
        Converts the table to markdown and returns a Text context object.
        """
        return Text(self.markdown())


class Table(BaseTable):
    content: str = ""
    df: List[Dict[str, Any]] = Field(hidden=True)
    _df: pd.DataFrame = PrivateAttr()

    @model_validator(mode="wrap")
    @classmethod
    def truncate(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> "Table":
        df = data["df"]
        if isinstance(df, pd.DataFrame):
            serialized_df = df.to_dict(orient="records")
            pd_df = df
        else:
            serialized_df = df
            pd_df = pd.DataFrame(serialized_df)
        data["df"] = serialized_df
        self = handler(data)
        self._df = pd_df
        return self


class TableFile(BaseTable):
    """
    A Context object to represent table documents such as excel, csv and tsv files.
    """

    file_ref: str
    file_type: Literal["xls", "xlsx", "csv", "tsv"]
    sheet_name: Optional[str] = None
    _df: pd.DataFrame = HydratedAttr()

    @classmethod
    def from_file(
        cls,
        file_ref: str,
        file: Path | IO,
        file_type: Literal["xls", "xlsx", "csv", "tsv"] = "xls",
        name: Optional[str] = None,
        sheet_name: Optional[str] = None,
        **kwargs,
    ) -> "TableFile":
        # NOTE: Always set a sheet name for excel files
        if file_type in ["xls", "xlsx"] and sheet_name is None:
            xls = pd.ExcelFile(file)
            sheet_name = xls.sheet_names[0]
        # NOTE: ensure we always have a semantic name for the table
        if name is None and file_type in ["xls", "xlsx"]:
            name = sheet_name
        elif name is None:
            name = file_ref.split("/")[-1].split(".")[0]
        instance = cls(
            name=name,
            file_ref=file_ref,
            file_type=file_type,
            sheet_name=sheet_name,
            **kwargs,
        )
        instance._hydrate_from_file(file)
        return instance

    @classmethod
    def from_file_store(cls, file_ref: str, file_store: FileStore, **kwargs) -> "TableFile":
        file_result = file_store.get(file_ref)
        if "sheet_name" in file_result.metadata:
            sheet_name = file_result.metadata["sheet_name"]
        else:
            sheet_name = None
        return cls.from_file(
            file_ref,
            file_result.file,
            file_result.type,
            name=file_result.name,
            sheet_name=sheet_name,
            **kwargs,
        )

    def hydrate(self, file_store: FileStore):
        file = file_store.get(self.file_ref)
        self._hydrate_from_file(file)

    def _hydrate_from_file(self, file: Path | IO):
        if self.file_type in ["excel", "xlsx"]:
            if self.sheet_name is None:
                df = pd.read_excel(file)
            else:
                df = pd.read_excel(file, sheet_name=self.sheet_name)
        elif self.file_type == "csv":
            df = pd.read_csv(file)
        elif self.file_type == "tsv":
            df = pd.read_csv(file, sep="\t")
        else:
            raise ValueError(f"TableFile got unsupported file type: {self.file_type}")
        self._df = _process_df(df)

    @requires_hydration
    def column_samples(self, col: str) -> list[Any]:
        """
        Returns up to 3 distinct sample values from the given column.
        """
        return super().column_samples(col)

    @requires_hydration
    def get_col(self, col_name: str) -> pd.Series:
        """
        Gets a single column from the table.
        """
        return super().get_col(col_name)

    @requires_hydration
    def schema_info(self) -> dict[str, Any]:
        """
        Returns the schema information of the table.
        """
        return super().schema_info()

    @requires_hydration
    def query(self, query: str) -> pd.DataFrame:
        """
        Queries the table using DuckDB SQL.
        """
        return super().query(query)

    @requires_hydration
    def embedme(self) -> str:
        """
        Converts the table to markdown and returns a Text context object.
        """
        return super().embedme()

    @requires_hydration
    def markdown(self) -> str:
        """
        Converts the table to markdown format.
        """
        return super().markdown()

    @requires_hydration
    def to_text(self) -> Text:
        """
        Converts the table to markdown and returns a Text context object.
        """
        return super().to_text()


class BaseTabbedTable(Context):
    names: Set[str]
    _tables: Optional[Dict[str, pd.DataFrame]] = PrivateAttr()
    _sql_db: Optional[duckdb.DuckDBPyConnection] = PrivateAttr()

    def init_sql(self):
        """
        Initilizes and in memory sql database for querying
        """
        self._sql_db = duckdb.connect(database=":memory:")
        for table_name, table in self._tables.items():
            self._sql_db.register(table_name, table.df)

    def close_sql(self):
        """
        Closes the in memory sql database
        """
        self._sql_db.close()
        self._sql_db = None

    @contextmanager
    def sql(self):
        self.init_sql()
        yield self._sql_db
        self.close_sql()

    def query(self, query: str) -> Table:
        """
        Queries the in memory sql database
        """
        if self._sql_db is None:
            raise ValueError(
                "Attempted to run query on MultiTabbedTable without initializing the SQL database. Use with `with MultiTabbedTable.sql():` or `MultiTabbedTable.init_sql()`"
            )
        try:
            df = self._sql_db.execute(query).fetchdf()
            return Table(df=df, name="query_result")
        except Exception as e:
            raise ValueError("Error querying SQL database") from e


class TabbedTable(BaseTabbedTable):
    tables: Dict[str, List[Dict[str, Any]]]
    _tables: Dict[str, pd.DataFrame] = PrivateAttr()
    _sql_db: duckdb.DuckDBPyConnection = PrivateAttr()

    @field_validator("tables", mode="before")
    @classmethod
    def serialize_tables(cls, tables: Dict[str, pd.DataFrame] | Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        first_val = next(iter(tables.values()))
        if isinstance(first_val, pd.DataFrame):
            serialized_tables = {k: v.to_dict(orient="records") for k, v in tables.items()}
        else:
            serialized_tables = tables
        return serialized_tables

    @model_validator(mode="after")
    def set_tables(self) -> "TabbedTable":
        self._tables = {k: pd.DataFrame(v) for k, v in self.tables.items()}
        return self


class TabbedTableFile(BaseTabbedTable):
    file_ref: str
    file_type: Literal["excel"] = "excel"
    _tables: Dict[str, pd.DataFrame] = HydratedAttr()
    _sql_db: duckdb.DuckDBPyConnection = HydratedAttr()

    @classmethod
    def from_file(
        cls,
        file_ref: str,
        file: Path | IO,
        file_type: Literal["excel"] = "excel",
        names: Optional[List[str]] = None,
        **kwargs,
    ) -> "TabbedTableFile":
        if file_type == "excel":
            xls = pd.ExcelFile(file)
            if names is None:
                names = xls.sheet_names
            else:
                for name in names:
                    if name not in xls.sheet_names:
                        raise ValueError(f"Sheet name {name} not found in file")
        elif names is None:
            raise ValueError(f"names must be provided for file type: {file_type}")

        instance = cls(
            file_ref=file_ref,
            file_type=file_type,
            names=set(names),
            **kwargs,
        )
        instance._hydrate_from_file(file)
        return instance

    @classmethod
    def from_file_store(cls, file_ref: str, file_store: FileStore, **kwargs) -> "TabbedTableFile":
        file_result = file_store.get(file_ref)
        if "sheet_name" in file_result.metadata:
            sheet_name = file_result.metadata["sheet_name"]
        else:
            sheet_name = 0
        return cls.from_file(
            file_ref,
            file_result.file,
            file_result.type,
            name=file_result.name,
            sheet_name=sheet_name,
            **kwargs,
        )

    def hydrate(self, file_store: FileStore):
        file = file_store.get(self.file_ref)
        self._hydrate_from_file(file)

    def _hydrate_from_file(self, file: Path | IO):
        if isinstance(file, IO):
            file = file.read()
        else:
            file = file.read_text()

        if self.file_type == "excel":
            df_dict = pd.read_excel(file, sheet_name=self.names)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

        if isinstance(df_dict, dict):
            self._tables = {name: _process_df(df) for name, df in df_dict.items()}
        else:
            self._tables = {self.names[0]: _process_df(df_dict)}

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


def _pandas_to_mysql_dtype(dtype: np.dtype) -> str:
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


def downcast_column(col: pd.Series) -> pd.Series:
    """
    Downcasts a column to the smallest possible dtype.
    """
    return col.apply(downcast_pd_series)


def _sanitize_columns(df: pd.DataFrame) -> None:
    columns = [sanitize_name(col) for col in df.columns]
    columns = ["No" if i == 0 and (not col or pd.isna(col)) else col for i, col in enumerate(columns)]

    seen = {}
    new_columns = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    df.columns = new_columns


def _process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the dataframe to ensure it is in the correct format.
    """
    # Downcast columns
    df = df.apply(downcast_pd_series)
    _sanitize_columns(df)
    return df
