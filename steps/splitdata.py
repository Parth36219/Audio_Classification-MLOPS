import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
import numpy as np

@step
def split_data(t:tuple) -> tuple:
    X=t[0]
    y=t[1]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return (X_train,X_test,y_train,y_test)
