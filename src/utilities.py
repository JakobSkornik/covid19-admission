import pandas as pd


# There are 231 columns in the original dataset
pd.set_option("display.max_columns", 500)


def load_xlsx(path: str) -> pd.DataFrame:
    """
    This method loads a xlsx file from path and returns a pandas DataFrame.
    """
    return pd.read_excel(path)


def get_target_variables(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    This method returns a DataFrame with two columns, the first is PATIENT_ID,
    the second is TARGET. TARGET being a binary indicator column.
    """

    # Create a dataframe, where each patient has true, if he was admitted to ICU during stay
    patient_grouped_bool = dataset.groupby("PATIENT_VISIT_IDENTIFIER").sum()["ICU"] > 0

    # Create new df and convert boolean to binary
    patient_grouped_binary = patient_grouped_bool.reset_index() * 1

    # Rename ICU to TARGET
    patient_grouped_binary.rename(columns={"ICU": "TARGET"}, inplace=True)

    return patient_grouped_binary


def append_target_variable(
    dataset: pd.DataFrame, target_df: pd.DataFrame
) -> pd.DataFrame:
    """
    This method appends the target column by joining two dataframes on column PATIENT_VISIT_IDENTIFIER.
    """
    col = "PATIENT_VISIT_IDENTIFIER"
    result = dataset.join(target_df.set_index(col), on=col)

    return result


def get_dummies(dataset: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    This method calls the pd.get_dummies method.
    """

    return pd.get_dummies(dataset, columns=cols, drop_first=True)
