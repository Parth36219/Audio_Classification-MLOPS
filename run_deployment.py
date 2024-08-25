from pipelines.deployment_pipeline import continuous_deployment_pipeline
import click
import mlflow
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from typing import cast
import streamlit as st
DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"
@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment pipeline to train and deploy a model (`deploy`), or to only run a prediction against the deployed model (`predict`). By default both will be run (`deploy_and_predict`).",
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model",
)
def run_deployment(config: str, min_accuracy: float):
    continuous_deployment_pipeline(min_accuracy=min_accuracy, workers=3, timeout=60)
    

if __name__ == "__main__":
    run_deployment()
