from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.splitdata import split_data
from steps.train import train_model
from steps.evaluate import evaluate_model

@pipeline
def audio_emotion_classification_pipeline():
    """
    Audio Emotion Classification Pipeline using ZenML.

    Steps:
    1. Ingest audio data and preprocess it.
    2. Split the preprocessed data into training and evaluation sets.
    3. Train a CNN model
    4. Evaluate the trained model on the evaluation data and print a classification report.
    """
    # Ingest and preprocess the data

    t= ingest_data()
    
    # Split the data into training and evaluation sets
    z = split_data(t)

    # Train a model on the training data
    model = train_model(z)

    # Evaluate the model on the evaluation data
    score = evaluate_model(model, z)

    # Log or return the evaluation report
    print("Classification Report:")
    print(score)
