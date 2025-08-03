import pandas as pd
from modaic.context.table import Table
import pyperclip


if __name__ == "__main__":
    df = pd.DataFrame(
        {"Column1": [1, 2, 3], "Column2": [4, 5, 6], "Column3": [7, 8, 9]}
    )
    table = Table(df, name="table")
    print(table.markdown())
    pyperclip.copy(table.markdown())
