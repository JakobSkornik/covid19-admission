import json
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


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


def get_datasets(method: str = "bfill") -> dict:
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
    dataset = dataset.fillna(method=method)

    # Rename cols
    dataset.columns = dataset.columns.str.replace(" ", "_")

    window_02_dataset = get_dummies(
        dataset[dataset.WINDOW == "0-2"].drop("WINDOW", axis=1), ["AGE_PERCENTIL"]
    )
    window_24_dataset = get_dummies(
        dataset[(dataset.WINDOW == "0-2") | (dataset.WINDOW == "2-4")].drop(
            "WINDOW", axis=1
        ),
        ["AGE_PERCENTIL"],
    )
    window_46_dataset = get_dummies(
        dataset[
            (dataset.WINDOW == "0-2")
            | (dataset.WINDOW == "2-4")
            | (dataset.WINDOW == "4-6")
        ].drop("WINDOW", axis=1),
        ["AGE_PERCENTIL"],
    )
    window_612_dataset = get_dummies(
        dataset[
            (dataset.WINDOW == "0-2")
            | (dataset.WINDOW == "2-4")
            | (dataset.WINDOW == "4-6")
            | (dataset.WINDOW == "6-12")
        ].drop("WINDOW", axis=1),
        ["AGE_PERCENTIL"],
    )

    result = {
        "window_0_2": window_02_dataset,
        "window_2_4": window_24_dataset,
        "window_4_6": window_46_dataset,
        "window_6_12": window_612_dataset,
        "window_all": get_dummies(dataset, ["WINDOW", "AGE_PERCENTIL"]),
    }

    return result


def random_forest(df: pd.DataFrame, target: str) -> tuple:
    """
    This method creates and executes a random forest model for a specified dataframe.
    """
    df_test = df.copy()
    y = df_test.pop(target)
    X = df_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    clf = RandomForestClassifier(max_depth=30, n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return (y_test, y_pred)


def evaluate(test: pd.Series, pred: np.ndarray) -> str:
    """
    Evaluates models accuracy.
    """
    acc = f"Accuracy: {metrics.accuracy_score(test, pred)}"
    prec = f"Precision: {metrics.precision_score(test, pred)}"
    rec = f"Recall: {metrics.recall_score(test, pred)}"
    return f"{acc}\n{prec}\n{rec}"


def visualize_confusion_matrix(test: pd.Series, pred: np.ndarray) -> None:
    """
    Shows a heatmap of confusion matrix.
    """
    cm = pd.DataFrame(metrics.confusion_matrix(test, pred))
    sb.heatmap(cm, annot=True, cmap="Blues")
