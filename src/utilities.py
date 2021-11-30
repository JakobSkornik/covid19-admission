import pandas as pd


def load_xlsx(path: str) -> pd.DataFrame:
    """
    This method loads a xlsx file from path and returns a pandas DataFrame.
    """
    return pd.read_excel(path)
