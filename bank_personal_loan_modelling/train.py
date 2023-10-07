import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.dirname(os.getcwd()) + '/MLOps/dataset'
MODEL_SAVE_PATH = os.path.dirname(os.getcwd()) + '/MLOps/models/model.cbm'
TRAIN_PATH = DATA_PATH + '/train.csv'
TEST_PATH = DATA_PATH + '/test.csv'

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

def training_data(X_train, y_train):
    categorial_columns = ['Family', 'Education', 'Securities Account',
                          'CD Account', 'Online', 'CreditCard']
    model = CatBoostClassifier(iterations=1000, learning_rate=0.01, loss_function='CrossEntropy',
                              eval_metric='AUC', use_best_model=True, random_state=42, verbose=100) 
    print("===== Training ... =====")
    test_data = get_data(TEST_PATH)
    X_test, y_test = split_data(test_data)
    model.fit(X_train, y_train, cat_features=categorial_columns, eval_set=(X_test, y_test))
    model.save_model(MODEL_SAVE_PATH)
    print(f"===== Model saved to {MODEL_SAVE_PATH} =====")

if __name__ == "__main__":
    train_data = get_data(TRAIN_PATH)
    X_train, y_train = split_data(train_data)
    training_data(X_train, y_train)
