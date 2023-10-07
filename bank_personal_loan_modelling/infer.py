import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DATA_PATH = os.path.dirname(os.getcwd()) + '/MLOps/dataset'
MODEL_SAVE_PATH = os.path.dirname(os.getcwd()) + '/MLOps/models/model.cbm'
TEST_PATH = DATA_PATH + '/test.csv'
PREDICT_PATH = os.path.dirname(os.getcwd()) + '/MLOps/dataset/predict.csv'

def get_data(path_to_data):
    data = pd.read_csv(path_to_data)
    ssc = StandardScaler()
    cols_to_scale = ['Age', 'Income', 'CCAvg', 'Mortgage']
    data[cols_to_scale] = ssc.fit_transform(data[cols_to_scale])
    data.drop(['ID', 'Experience', 'ZIP Code'], axis=1, inplace=True)
    return data

def split_data(data):
    target_column = 'Personal Loan'
    X = data.loc[:, data.columns != target_column]
    y = data.loc[:, target_column]
    return X, y

def predicting_data(X_test, y_test):
    model = CatBoostClassifier()
    model.load_model(MODEL_SAVE_PATH)
    y_predict = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_predict)}")
    print(f"Precision: {precision_score(y_test, y_predict)}")
    print(f"Recall: {recall_score(y_test, y_predict)}")
    print(f"F1-score: {f1_score(y_test, y_predict)}")
    pd.DataFrame(y_predict, columns=['Personal Loan']).to_csv(PREDICT_PATH)
    print(f"===== Predictions saved to {PREDICT_PATH}")

if __name__ == "__main__":
    test_data = get_data(TEST_PATH)
    X_test, y_test = split_data(test_data)
    predicting_data(X_test, y_test)
