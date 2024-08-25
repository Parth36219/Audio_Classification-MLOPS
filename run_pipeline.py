import os
from zenml.client import Client
from pipelines.pipeline import audio_emotion_classification_pipeline
if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())#code to get mlflow uri
    audio_emotion_classification_pipeline()
    #run to get mlflow tracker
    ##mlflow ui --backend-store-uri "file:C:\Users\Parth\AppData\Roaming\zenml\local_stores\43e5d322-fda8-496c-8123-fbe59e053d1b\mlruns"