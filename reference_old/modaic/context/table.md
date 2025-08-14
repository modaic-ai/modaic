# Table of Contents

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

---
sidebar_label: table
title: modaic.context.table
---

## Table Objects

```python
class Table(Molecular)
```

A molecular context object that represents a table. Can be queried with SQL.

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

#### get\_col

```python
def get_col(col_name: str) -> pd.Series
```

Gets a single column from the table.

**Arguments**:

- `col_name` - Name of the column to get
  

**Returns**:

  The specified column as a pandas Series.

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

#### query

```python
def query(query: str)
```

Queries the table. All queries run should refer to the table as `this` or `This`

#### markdown

```python
def markdown() -> str
```

Converts the table to markdown format.
Returns a markdown representation of the table with the table name as header.

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

#### embedme

```python
def embedme()
```

embedme method for table. Returns a markdown representation of the table.

## MultiTabbedTable Objects

```python
class MultiTabbedTable(Molecular)
```

#### init\_sql

```python
def init_sql()
```

Initilizes and in memory sql database for querying

#### close\_sql

```python
def close_sql()
```

Closes the in memory sql database

#### query

```python
def query(query: str)
```

Queries the in memory sql database

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

#### is\_valid\_table\_name

```python
def is_valid_table_name(name: str) -> bool
```

Checks if a name is a valid table name.

**Arguments**:

- `name` - The name to validate.
  

**Returns**:

  True if the name is valid, False otherwise.

