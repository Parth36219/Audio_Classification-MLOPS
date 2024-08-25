import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from zenml import step
from tensorflow.keras import layers, models, callbacks
from zenml.client import Client
import mlflow


@step(experiment_tracker="mlflow_tracker")
def train_model(z:tuple) -> Sequential:
    X_train=z[0]
    y_train=z[2]
    mlflow.tensorflow.autolog()
    model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1:])),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(256, activation='relu'),
    layers.Dense(3, activation='softmax')
])


    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        # Print the model summary
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
    return model