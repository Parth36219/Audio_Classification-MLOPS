import json
import os
import numpy as np
import pandas as pd
from steps.ingest_data import ingest_data
from steps.splitdata import split_data
from tensorflow.keras.models import Sequential
from steps.train import train_model
from steps.evaluate import evaluate_model
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
import pickle  # To save the trained model
from zenml.steps import BaseParameters, Output

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step
def save_and_deploy_model(model: Sequential) -> None:
    # Save the model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Launch Streamlit app for predictions
    os.system("streamlit run streamlit_app.py")

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.8,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    t = ingest_data()
    z = split_data(t)
    model = train_model(z)
    score = evaluate_model(model, z)
    # Step 5: Save the model and deploy it using Streamlit
    save_and_deploy_model(model=model)

# @pipeline(enable_cache=False, settings={"docker": docker_settings})
# def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
#     # Link all the steps artifacts together
#     batch_data = dynamic_importer()
#     model_deployment_service = prediction_service_loader(
#         pipeline_name=pipeline_name,
#         pipeline_step_name=pipeline_step_name,
#         running=False,
#     )
