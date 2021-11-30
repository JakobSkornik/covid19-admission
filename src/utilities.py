import pandas as pd


# There are 231 columns in the original dataset
pd.set_option("display.max_columns", 500)


def load_xlsx(path: str) -> pd.DataFrame:
    """
    This method loads a xlsx file from path and returns a pandas DataFrame.
    """
    return pd.read_excel(path)
