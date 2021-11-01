import numpy as np
import pandas as pd
from scipy.io import arff
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

trainfile = sys.argv[1] if len(sys.argv) > 1 else "NSL-KDD/KDDTrain+_20Percent.arff"
testfile = sys.argv[2] if len(sys.argv) > 2 else "NSL-KDD/KDDTest-21.arff"

def loadXy(train_file, test_file):
    train_data, meta = arff.loadarff(train_file)
    test_data, _ = arff.loadarff(test_file)

    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)
    df = pd.concat([df_train, df_test], ignore_index=True)

    categorical_ix = df.select_dtypes(include=['object', 'bool']).columns
    t = [('cat', OneHotEncoder(drop='if_binary'), categorical_ix)]
    tf = ColumnTransformer(transformers=t, remainder = "passthrough")

    data = tf.fit_transform(df)

    X, y = data[:,:-1], data[:,-1]

    n_train = len(df_train)
    return X[:n_train, ...], X[n_train:, ...], y[:n_train], y[n_train:]

X_train, X_test, y_train, y_test = loadXy(trainfile, testfile)

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)

y_pred_test = regressor.predict(X_test)

print(r2_score(y_test, y_pred_test))
