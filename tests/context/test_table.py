import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest

from modaic.context.table import Table

base_dir = pathlib.Path(__file__).parent


def test_from_excel():
    test_file = base_dir / "artifacts/1st_New_Zealand_Parliament_0.xlsx"
    table = Table.from_excel(test_file)
    assert table.name == "t_1st_new_zealand_parliament_0"
    correct_df = pd.read_excel(test_file)
    columns = [col.lower().replace(" ", "_") for col in correct_df.columns]
    correct_df.columns = columns
    pd.testing.assert_frame_equal(table._df, correct_df)


def test_table_markdown():  # TODO: Test with nan and None values
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df, name="table")
    correct_markdown = textwrap.dedent("""\
        Table name: table
        | column1 | column2 | column3 |
        | --- | --- | --- |
        | 1 | 4 | 7 |
        | 2 | 5 | 8 |
        | 3 | 6 | 9 |
    """)
    assert table.markdown() == correct_markdown


def test_get_sample_values():  # TODO:
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df, name="table")
    assert set(table.get_sample_values("column1")) == set([1, 2, 3])
    assert set(table.get_sample_values("column2")) == set([4, 5, 6])
    assert set(table.get_sample_values("column3")) == set([7, 8, 9])

    df = pd.DataFrame(
        {
            "Column1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Column2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "Column3": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        }
    )
    table = Table(df, name="table")
    print("column1 sample values", table.get_sample_values("column1"))
    print("column1", df["column1"].tolist())

    assert set(table.get_sample_values("column1")).issubset(set(df["column1"].tolist()))
    assert set(table.get_sample_values("column2")).issubset(set(df["column2"].tolist()))
    assert set(table.get_sample_values("column3")).issubset(set(df["column3"].tolist()))

    df = pd.DataFrame(
        {
            "Column1": [1, 2, None, None, None],
            "Column2": [11, 12, 13, 14, 15],
            "Column3": [None, None, None, None, None],
        }
    )
    table = Table(df, name="table")
    assert set(table.get_sample_values("column1")) == set([1.0, 2.0])
    assert set(table.get_sample_values("column2")).issubset(set(df["column2"].tolist()))
    assert table.get_sample_values("column3") == []


def test_table_readme():
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df, name="table")
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df, name="table")
    correct_markdown = textwrap.dedent("""\
        Table name: table
        | column1 | column2 | column3 |
        | --- | --- | --- |
        | 1 | 4 | 7 |
        | 2 | 5 | 8 |
        | 3 | 6 | 9 |
    """)
    assert table.readme() == correct_markdown


def test_table_embedme():
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df, name="table")
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df, name="table")
    correct_markdown = textwrap.dedent("""\
        Table name: table
        | column1 | column2 | column3 |
        | --- | --- | --- |
        | 1 | 4 | 7 |
        | 2 | 5 | 8 |
        | 3 | 6 | 9 |
    """)
    assert table.embedme() == correct_markdown


@pytest.mark.skip(reason="Not implemented")
def test_downcast_columns():  # TODO:
    pass


def test_get_col():
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df, name="table")
    pd.testing.assert_series_equal(table.get_col("column1"), pd.Series([1, 2, 3], name="column1"))
    pd.testing.assert_series_equal(table.get_col("column2"), pd.Series([4, 5, 6], name="column2"))
    pd.testing.assert_series_equal(table.get_col("column3"), pd.Series([7, 8, 9], name="column3"))


def test_get_schema_with_samples():
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]})
    table = Table(df, name="table")
    print(table.get_schema_with_samples())
    schema = table.get_schema_with_samples()
    assert schema["column1"]["type"] == "INT"
    assert set(schema["column1"]["sample_values"]) == set([1, 2, 3])
    assert schema["column2"]["type"] == "INT"
    assert set(schema["column2"]["sample_values"]) == set([4, 5, 6])
    assert schema["column3"]["type"] == "INT"
    assert set(schema["column3"]["sample_values"]) == set([7, 8, 9])


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
    table = Table(df, name="test_table")

    int_samples = table.get_sample_values("int_col")
    float_samples = table.get_sample_values("float_col")
    str_samples = table.get_sample_values("str_col")

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
    table = Table(df, name="test_table")

    schema_info = table.schema_info()

    # Should not raise any exception
    json_string = json.dumps(schema_info)

    # Verify we can parse it back
    parsed = json.loads(json_string)
    assert parsed["table_name"] == "test_table"
    assert "column_dict" in parsed


def test_sample_values_with_mixed_types():
    df = pd.DataFrame({"mixed_col": [1, 2.5, 3, None, 4.7, 5]})
    table = Table(df, name="mixed_table")

    samples = table.get_sample_values("mixed_col")

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
    table = Table(df, name="empty_table")

    empty_samples = table.get_sample_values("empty_col")
    long_samples = table.get_sample_values("all_long_strings")

    assert empty_samples == []
    assert long_samples == []
