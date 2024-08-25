import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score
from zenml import step

@step
def evaluate_model(model: Sequential, z:tuple) -> tuple:
    """Evaluate the trained RNN model on the test data and return F1 score and accuracy."""
    X_test=z[1]
    y_test=z[3]
    
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    y_pred_prob = model.predict(X_test)
    
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')
    accuracy = accuracy_score(y_test_encoded, y_pred)
    return (f1,accuracy)
