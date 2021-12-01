import json
import pandas as pd


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


def get_datasets() -> dict:
    """
    This method creates a dictionary containing datasets.
    """

    # Import dataset and append target variable
    dataset = load_xlsx("data/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")
    patient_target_df = get_target_variables(dataset)
    dataset = append_target_variable(dataset, patient_target_df)

    # Drop useless rows and cols
    dataset = dataset[dataset.ICU != 1]
    dataset = dataset.drop(["PATIENT_VISIT_IDENTIFIER", "ICU"], axis=1)

    # Fill null values
    dataset_backward_fill = dataset.fillna(method="bfill")
    dataset_forward_fill = dataset.fillna(method="ffill")

    datasets = [dataset_backward_fill, dataset_forward_fill]

    # Set dummies
    for dataset in datasets:
        dataset.columns = dataset.columns.str.replace(" ", "_")
        dataset = get_dummies(dataset, cols=["AGE_PERCENTIL"])

    # Store datasets in a dict

    result = dict()

    result["ffill_datasets"] = dict()
    result["bfill_datasets"] = dict()

    dataset = dataset_forward_fill
    window_02_dataset = dataset[dataset.WINDOW == "0-2"]
    window_24_dataset = dataset[(dataset.WINDOW == "0-2") | (dataset.WINDOW == "2-4")]
    window_46_dataset = dataset[
        (dataset.WINDOW == "0-2")
        | (dataset.WINDOW == "2-4")
        | (dataset.WINDOW == "4-6")
    ]
    window_612_dataset = dataset[
        (dataset.WINDOW == "0-2")
        | (dataset.WINDOW == "2-4")
        | (dataset.WINDOW == "4-6")
        | (dataset.WINDOW == "6-12")
    ]

    result["ffill_datasets"] = {
        "window_0_2": window_02_dataset,
        "window_2_4": window_24_dataset,
        "window_4_6": window_46_dataset,
        "window_6_12": window_612_dataset,
        "window_all": dataset,
    }

    dataset = dataset_backward_fill
    window_02_dataset = dataset[dataset.WINDOW == "0-2"]
    window_24_dataset = dataset[(dataset.WINDOW == "0-2") | (dataset.WINDOW == "2-4")]
    window_46_dataset = dataset[
        (dataset.WINDOW == "0-2")
        | (dataset.WINDOW == "2-4")
        | (dataset.WINDOW == "4-6")
    ]
    window_612_dataset = dataset[
        (dataset.WINDOW == "0-2")
        | (dataset.WINDOW == "2-4")
        | (dataset.WINDOW == "4-6")
        | (dataset.WINDOW == "6-12")
    ]

    result["bfill_datasets"] = {
        "window_0_2": window_02_dataset,
        "window_2_4": window_24_dataset,
        "window_4_6": window_46_dataset,
        "window_6_12": window_612_dataset,
        "window_all": dataset,
    }

    return result
