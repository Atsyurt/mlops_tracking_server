#!/usr/bin/env python3
"""
Model evaluation and validation script for Credit Card Fraud Detection MLOps Pipeline.

This script:
1. Loads a trained model from MLflow (champion alias)
2. Evaluates the model on test data
3. Calculates performance metrics
4. Validates against performance requirements
5. Sets up a simple API for model inference

Usage:
    python scripts\validate.py --model-version 1 --data-rev localv_1_processed  --start-api
"""
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
import mlflow.pyfunc
import subprocess
import uuid
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
# Import FastAPI for API setup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('model-validation')

# Constants
DATA_DIR = Path("data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
VALIDATION_DIR = MODELS_DIR / "validation"

# Performance requirements
PERFORMANCE_REQUIREMENTS = {
    "accuracy": 0.90,
    "precision": 0.85,
    "recall": 0.70,
    "f1": 0.75,
    "roc_auc": 0.85
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model validation script')
    parser.add_argument('--model-version', type=str, required=True,
                        help='MLflow model version to validate')
    parser.add_argument('--data-rev', type=str, required=True,
                        help='DVC revision/version of the test data to use')
    parser.add_argument('--start-api', action='store_true',
                        help='Start the prediction API after validation')
    # Add more arguments as needed
    return parser.parse_args()

def setup_directories():
    """Create necessary directories if they don't exist."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured validation directory exists: {VALIDATION_DIR}")
    return VALIDATION_DIR

def setup_mlflow():
    """Configure MLflow tracking."""
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'  # Adjust for remote
    os.environ['MLFLOW_TRACKING_USERNAME'] = "user"
    client = mlflow.tracking.MlflowClient()
    exp_name="Creditcard-Fraud-Detection"
    # Check if experiment already exists
    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = client.create_experiment(exp_name)
        
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)
    #set custom run name for this exp
    unique_id = uuid.uuid4()
    mlflow.set_tag('mlflow.runName', 'exp_validation_part_logs_'+str(unique_id))
    print(f"MLflow experiment set up: id {experiment_id}")

def load_test_data(data_rev):
    """Load preprocessed test data from specific DVC revision."""
    try:
        # Checkout the specific DVC version
        subprocess.run(["git", "checkout", data_rev], check=True)
        subprocess.run(["dvc", "checkout"], check=True)
        logger.info(f"Checked out data version: {data_rev}")
    except Exception as e:
        logger.error(f"Failed to checkout test data: {e}")
    try:
        #load test data
        test_path=PROCESSED_DATA_DIR / "test.csv"
        test_df = pd.read_csv(test_path)
        logger.info(f"Test data loaded successfully from {test_path}")
        return test_df
    
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        sys.exit(1)

def load_model(model_version):
    """Load model from MLflow using champion alias."""
    try:
        model = mlflow.pyfunc.load_model(f"models:/BestXGBModel/{model_version}")
        logger.info(f"Successfully loaded champion model version {model_version}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data and return metrics."""
    y_pred = model.predict(X_test)
    #y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }

    logger.info(f"Model evaluation metrics: {metrics}")
    return metrics

def validate_performance(metrics):
    """Validate model performance against predefined requirements."""
    validation_passed = all(
        metrics[key] >= PERFORMANCE_REQUIREMENTS[key] for key in PERFORMANCE_REQUIREMENTS
    )

    if validation_passed:
        logger.info("✅ Model meets all performance requirements.")
    else:
        logger.warning("❌ Model does not meet performance requirements.")

    return validation_passed


def create_visualizations(y_test, y_pred, y_pred_proba):
    """Create evaluation visualizations."""
import matplotlib.pyplot as plt
import mlflow

def create_visualizations(metrics):
    """Create a performance comparison line graph."""
    plt.figure(figsize=(8, 5))

    # Performance thresholds
    thresholds = list(PERFORMANCE_REQUIREMENTS.values())
    actual_values = list(metrics.values())
    labels = list(metrics.keys())

    # Plot the metrics vs. requirements
    plt.plot(labels, thresholds, marker='o', label="Requirements", linestyle="dashed", color="red")
    plt.plot(labels, actual_values, marker='o', label="Model Performance", linestyle="-", color="blue")

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Model Performance vs. Requirements")
    plt.legend()
    plt.grid()

    # Save the plot
    plot_path = "performance_plot.png"
    plt.savefig(plot_path)
    logger.info(f"Saved performance comparison plot to {plot_path}")

    # Log the plot to MLflow
    mlflow.log_artifact(plot_path)
    logger.info("Performance plot logged to MLflow.")

def log_to_mlflow(metrics, model_version, validation_passed):
    """Log evaluation results to MLflow."""
    mlflow.log_params({"model_version": model_version})
    mlflow.log_metrics(metrics)
    mlflow.set_tag("validation_passed", str(validation_passed))
    logger.info("Model evaluation results logged to MLflow.")

def setup_api(model):
    """Set up a FastAPI application for model inference."""
    app = FastAPI()

    class InputData(BaseModel):
        features: List[List[float]]

    @app.post("/predict")
    def predict(data: InputData):
        try:
            predictions = model.predict(np.array(data.features)).tolist()
            return {"predictions": predictions}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

def main():
    """Main function to orchestrate the model validation pipeline."""
    args = parse_args()
    logger.info(f"Starting model validation with model version {args.model_version} and data revision {args.data_rev}")
    #setup directories and mlflow setup
    setup_directories()
    setup_mlflow()
    test_df = load_test_data(args.data_rev)
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    model = load_model(args.model_version)

    metrics = evaluate_model(model, X_test, y_test)
    validation_passed = validate_performance(metrics)
    
    log_to_mlflow(metrics, args.model_version, validation_passed)
    create_visualizations(metrics)
    if args.start_api:
        app = setup_api(model)
        import uvicorn
        logger.info("Starting FastAPI model inference server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()











# def log_to_mlflow(metrics, model_version, validation_passed):
#     """Log evaluation results to MLflow."""
#     with mlflow.start_run():
#         mlflow.log_params({"model_version": model_version})
#         mlflow.log_metrics(metrics)
#         mlflow.set_tag("validation_passed", str(validation_passed))
#         logger.info("Model evaluation results logged to MLflow.")




