import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def load_xlsx(path: str) -> pd.DataFrame:
    """This method loads a xlsx file from path and returns a pandas DataFrame."""
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
    """This method appends the target column by joining two dataframes on column PATIENT_VISIT_IDENTIFIER."""

    col = "PATIENT_VISIT_IDENTIFIER"
    result = dataset.join(target_df.set_index(col), on=col)

    return result


def get_dummies(dataset: pd.DataFrame, cols: list) -> pd.DataFrame:
    """This method calls the pd.get_dummies method."""

    return pd.get_dummies(dataset, columns=cols, drop_first=True)


def get_datasets(method: str = "bfill") -> dict:
    """This method creates a dictionary containing datasets."""

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

    valid_json = dict(
        (key, val.to_json(orient="records")) for key, val in result.items()
    )

    with open("data/datasets.json", "w") as outfile:
        json.dump(valid_json, outfile)

    return json


def random_forest(df: pd.DataFrame, target: str) -> tuple:
    """This method creates and executes a random forest model for a specified dataframe."""

    df_test = df.copy()
    y = df_test.pop(target)
    X = df_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier(max_depth=30, n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return (y_test, y_pred)


def evaluate(test: pd.Series, pred: np.ndarray) -> str:
    """Evaluates models accuracy."""

    tn, fp, fn, tp = metrics.confusion_matrix(test, pred).ravel()
    acc = f"Accuracy: {metrics.accuracy_score(test, pred)}"
    sens = f"Sensitivity: {round(tp / (tp + fn), 2)}"
    spec = f"Specificity: {round(tn / (tn + fp), 2)}"
    return f"{acc}\n{sens}\n{spec}"


def visualize_confusion_matrix(test: pd.Series, pred: np.ndarray) -> None:
    """Shows a heatmap of confusion matrix."""

    cm = pd.DataFrame(metrics.confusion_matrix(test, pred))
    sb.heatmap(cm, annot=True, cmap="Blues")


def evaluate_custom(X, y, nn):
    correct = 0
    confusion_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    y = np.argmax(y, axis=1)

    for i in range(len(X)):
        truth = y[i]
        entry = X[i]
        predicted = nn.predict(entry)

        if predicted == truth:
            correct += 1

            if predicted == 1:
                confusion_matrix["TP"] += 1
            else:
                confusion_matrix["TN"] += 1

        else:
            if predicted == 1:
                confusion_matrix["FP"] += 1
            else:
                confusion_matrix["FN"] += 1

    print(
        f"""=======================
RESULTS:

    TP: {confusion_matrix["TP"]},
    TN: {confusion_matrix["TN"]},
    FP: {confusion_matrix["FP"]},
    FN: {confusion_matrix["FN"]}
    accuracy: {correct} / {len(X)} = {round((correct / len(X)) * 100, 2)}%
    sensitivity: {round(confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"]), 2)}
    specificity: {round(confusion_matrix["TN"] / (confusion_matrix["TN"] + confusion_matrix["FP"]), 2)}
    """
    )
    return confusion_matrix


def visualize_custom(cm_dict: dict) -> None:
    """Visualize confusion matrix."""

    cm = pd.DataFrame(
        np.array([[cm_dict["TN"], cm_dict["FP"]], [cm_dict["FN"], cm_dict["TP"]]])
    )
    sb.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Ground Truths")
    plt.ylabel("Predictions")


def split_at_x_percent(dataset: np.ndarray, x: int) -> tuple:
    """Returns two pd.DataFrames by splitting original one at x%."""

    rows = len(dataset)
    idx = int((rows * x) / 100)

    if len(dataset.shape) == 1:
        return dataset[:idx], dataset[idx + 1 :]

    return dataset[:idx, :], dataset[idx + 1 :, :]


def one_hot(a, num_classes):
    """Transforms vector to one-hot encoded vector."""

    if type(a) is list:
        a = np.array(a)

    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
