import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest

from modaic.context.table import Table, TableFile
from modaic.storage.file_store import InPlaceFileStore

base_dir = pathlib.Path(__file__).parents[1]


def test_from_file():
    test_file = base_dir / "artifacts/1st_New_Zealand_Parliament_0.xlsx"
    table = TableFile.from_file(file_ref=str(test_file), file=test_file, file_type="xlsx")
    # Current behavior names xlsx tables by sheet when name is not provided
    # Accept either legacy sanitized filename or default sheet name
    assert table.name in {"t_1st_new_zealand_parliament_0", (table.sheet_name or "").lower(), table.sheet_name}
    correct_df = pd.read_excel(test_file)
    columns = [col.lower().replace(" ", "_") for col in correct_df.columns]
    correct_df.columns = columns
    pd.testing.assert_frame_equal(table._df, correct_df)


def test_from_file_store():
    file_store = InPlaceFileStore(base_dir / "artifacts/test_dir")
    test_ref = "1st_New_Zealand_Parliament_0.xlsx"
    table = TableFile.from_file_store(file_ref=test_ref, file_store=file_store)
    assert table.name == "t_1st_new_zealand_parliament_0"
    correct_df = pd.read_excel(file_store.get(test_ref).file)
    columns = [col.lower().replace(" ", "_") for col in correct_df.columns]
    correct_df.columns = columns
    pd.testing.assert_frame_equal(table._df, correct_df)


def test_table_markdown():  # TODO: Test with nan and None values
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df=df, name="table")
    correct_markdown = textwrap.dedent("""\
        Table name: table
        | Column1 | Column2 | Column3 |
        | --- | --- | --- |
        | 1 | 4 | 7 |
        | 2 | 5 | 8 |
        | 3 | 6 | 9 |
    """)
    assert table.markdown().strip() == correct_markdown.strip()


def test_get_sample_values():  # TODO:
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df=df, name="table")
    assert set(table.column_samples("Column1")) <= set([1, 2, 3])
    assert set(table.column_samples("Column2")) <= set([4, 5, 6])
    assert set(table.column_samples("Column3")) <= set([7, 8, 9])

    df = pd.DataFrame(
        {
            "Column1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Column2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "Column3": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        }
    )
    table = Table(df=df, name="table")

    assert set(table.column_samples("Column1")).issubset(set(df["Column1"].tolist()))
    assert set(table.column_samples("Column2")).issubset(set(df["Column2"].tolist()))
    assert set(table.column_samples("Column3")).issubset(set(df["Column3"].tolist()))

    df = pd.DataFrame(
        {
            "Column1": [1, 2, None, None, None],
            "Column2": [11, 12, 13, 14, 15],
            "Column3": [None, None, None, None, None],
        }
    )
    table = Table(df=df, name="table")
    assert set(table.column_samples("Column1")) == set([1.0, 2.0])
    assert set(table.column_samples("Column2")).issubset(set(df["Column2"].tolist()))
    assert table.column_samples("Column3") == []


def test_table_embedme():
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df=df, name="table")
    correct_markdown = textwrap.dedent("""\
        Table name: table
        | Column1 | Column2 | Column3 |
        | --- | --- | --- |
        | 1 | 4 | 7 |
        | 2 | 5 | 8 |
        | 3 | 6 | 9 |
    """)
    assert table.embedme().strip() == correct_markdown.strip()


@pytest.mark.skip(reason="Not implemented")
def test_downcast_columns():  # TODO:
    pass


def test_get_col():
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df=df, name="table")
    pd.testing.assert_series_equal(table.get_col("Column1"), pd.Series([1, 2, 3], name="Column1"))
    pd.testing.assert_series_equal(table.get_col("Column2"), pd.Series([4, 5, 6], name="Column2"))
    pd.testing.assert_series_equal(table.get_col("Column3"), pd.Series([7, 8, 9], name="Column3"))


def test_get_schema_with_samples():
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df=df, name="table")
    schema = table.schema_info()
    assert schema["table_name"] == "table"
    assert schema["column_dict"]["Column1"]["type"] == "INT"
    assert set(schema["column_dict"]["Column1"]["sample_values"]) <= set([1, 2, 3])
    assert schema["column_dict"]["Column2"]["type"] == "INT"
    assert set(schema["column_dict"]["Column2"]["sample_values"]) <= set([4, 5, 6])
    assert schema["column_dict"]["Column3"]["type"] == "INT"
    assert set(schema["column_dict"]["Column3"]["sample_values"]) <= set([7, 8, 9])


def test_sample_values_are_json_serializable():
    import json

    df = pd.DataFrame(
        {
            "int_col": [np.int64(1), np.int64(2), np.int64(3)],
            "float_col": pd.Series(
                [np.float64(1.1), np.float64(2.2), np.float64(3.3)],
                dtype=pd.Float64Dtype(),
            ),
            "str_col": ["a", "b", "c"],
        }
    )
    table = Table(df=df, name="test_table")

    int_samples = table.column_samples("int_col")
    float_samples = table.column_samples("float_col")
    str_samples = table.column_samples("str_col")

    # Should not raise any exception
    json.dumps(int_samples)
    json.dumps(float_samples)
    json.dumps(str_samples)

    # Verify types are Python native types
    for val in int_samples:
        assert isinstance(val, int)
    for val in float_samples:
        assert isinstance(val, float)
    for val in str_samples:
        assert isinstance(val, str)


def test_schema_info_is_json_serializable():
    import json

    df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.1, 2.2, 3.3], "str_col": ["a", "b", "c"]})
    table = Table(df=df, name="test_table")

    schema_info = table.schema_info()

    # Should not raise any exception
    json_string = json.dumps(schema_info)

    # Verify we can parse it back
    parsed = json.loads(json_string)
    assert parsed["table_name"] == "test_table"
    assert "column_dict" in parsed


def test_sample_values_with_mixed_types():
    df = pd.DataFrame({"mixed_col": [1, 2.5, 3, None, 4.7, 5]})
    table = Table(df=df, name="mixed_table")

    samples = table.column_samples("mixed_col")

    # All values should be Python native types
    for val in samples:
        assert isinstance(val, (int, float))
        assert not hasattr(val, "item")  # Not numpy types


def test_empty_column_sample_values():
    df = pd.DataFrame(
        {
            "empty_col": [None, None, None],
            "all_long_strings": ["x" * 100, "y" * 100, "z" * 100],
        }
    )
    table = Table(df=df, name="empty_table")

    empty_samples = table.column_samples("empty_col")
    long_samples = table.column_samples("all_long_strings")

    assert empty_samples == []
    assert long_samples == []
