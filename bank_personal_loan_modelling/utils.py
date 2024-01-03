import pandas as pd
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.preprocessing import StandardScaler


def get_data(path_to_data):
    data = pd.read_csv(path_to_data)
    ssc = StandardScaler()
    cols_to_scale = ["Age", "Income", "CCAvg", "Mortgage"]
    data[cols_to_scale] = ssc.fit_transform(data[cols_to_scale])
    data.drop(["ID", "Experience", "ZIP Code"], axis=1, inplace=True)
    return data


def split_data(data):
    target_column = "Personal Loan"
    X = data.loc[:, data.columns != target_column]
    y = data.loc[:, target_column]
    return X, y


def get_metrics(y_test, y_predict):
    metrics = {}
    accuracy_score = accuracy(y_test, y_predict)
    precision_score = precision(y_test, y_predict)
    recall_score = recall(y_test, y_predict)
    f1_score = f1(y_test, y_predict)
    metrics = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1-score": f1_score,
    }
    return metrics
