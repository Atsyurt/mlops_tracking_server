#!/usr/bin/env python3
"""
Model training script for Credit Card Fraud Detection MLOps Pipeline.

This script:
1. Loads preprocessed data from a specific DVC version
2. Trains a Gradient Boosting model (XGBoost, LightGBM, etc.)
3. Performs hyperparameter tuning
4. Tracks experiments with MLflow
5. Registers the best model

Usage:
    python train.py --data-rev <DVC_REVISION>
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import joblib
import xgboost as xgb
import uuid
# Import your chosen model library (e.g., xgboost, lightgbm)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import RandomizedSearchCV
import mlflow
# Import mlflow model tracking for your chosen model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('model-training')

# Constants
DATA_DIR = Path("data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument('--data-rev', type=str, required=True,
                        help='DVC revision/version of the processed data to use')
    # Add more arguments as needed
    return parser.parse_args()

def setup_directories():
    """Create necessary directories if they don't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured models directory exists: {MODELS_DIR}")

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
    mlflow.set_tag('mlflow.runName', 'exp_train_part_logs_'+str(unique_id))
    print(f"MLflow experiment set up: id {experiment_id}")

def load_data(data_rev):
    """Load preprocessed data from specific DVC revision."""
    try:
        # Checkout the specific DVC version
        subprocess.run(["git", "checkout", data_rev], check=True)
        subprocess.run(["dvc", "checkout"], check=True)
        logger.info(f"Checked out data version: {data_rev}")

        # Load the dataset
        train_path=PROCESSED_DATA_DIR / "train.csv"
        val_path=PROCESSED_DATA_DIR / "val.csv"
        test_path=PROCESSED_DATA_DIR / "test.csv"
        train_df = pd.read_csv(train_path)
        logger.info(f"Train data loaded successfully from {train_path}")
        val_df = pd.read_csv(val_path)
        logger.info(f"Validation data loaded successfully from {val_path}")
        test_df = pd.read_csv(test_path)
        logger.info(f"Test data loaded successfully from {test_path}")
        return train_df,val_df,test_df
    
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)



def train_model(X_train, y_train, X_val, y_val):
    """Train and tune the model, register top models with MLflow."""
    
    params = {
        "n_estimators": [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        "max_depth": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
    }
    
    model = xgb.XGBClassifier()
    search = RandomizedSearchCV(model, params, cv=3, scoring="roc_auc", n_jobs=-1, n_iter=10)
    search.fit(X_train, y_train)
    
    # Extract the top three models based on score
    sorted_indices = sorted(range(len(search.cv_results_["mean_test_score"])), 
                            key=lambda i: search.cv_results_["mean_test_score"][i], 
                            reverse=True)[:3]

    best_model = search.best_estimator_
    logger.info(f"Best model selected with parameters: {search.best_params_}")

    # Start MLflow run and log models
    mlflow.sklearn.log_model(best_model, "best_model")
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("roc_auc", search.best_score_)

    # Register the best model as "champion"
    best_model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
    mlflow.register_model(best_model_uri, "BestXGBModel")
    mlflow.set_tag("champion_model", "true")

    # Register top three models as "lowmodel"
    for i, idx in enumerate(sorted_indices):
            model_params = {key: params[key][idx % len(params[key])] for key in params}
            candidate_model = xgb.XGBClassifier(**model_params)
            candidate_model.fit(X_train, y_train)

            model_name = f"LowXGBModel_{i+1}"
            mlflow.sklearn.log_model(candidate_model, model_name)
            
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
            mlflow.register_model(model_uri, "LowXGBModel")
            mlflow.set_tag("lowmodel", "true")

    return best_model


def evaluate_model(model, X_val, y_val):
    """Evaluate the model on validation data."""
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred),
    }
    logger.info(f"Model evaluation metrics: {metrics}")
    return metrics


def log_to_mlflow(model, params, metrics, X_val, y_val):
    """Log the model, parameters, and metrics to MLflow."""

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.xgboost.log_model(model, "fraud_detection_model")
    logger.info("Model logged to MLflow")


def save_model(model):
    """Save the trained model to disk."""
    model_path = MODELS_DIR / "fraud_detection_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved successfully at: {model_path}")


def main():
    """Main function to orchestrate the model training pipeline."""
    args = parse_args()
    setup_directories()
    setup_mlflow()
    
    train_df, val_df, test_df = load_data(args.data_rev)
    X_train, y_train = train_df.drop("Class", axis=1), train_df["Class"]
    X_val, y_val = val_df.drop("Class", axis=1), val_df["Class"]

    model = train_model(X_train, y_train, X_val, y_val)
    metrics = evaluate_model(model, X_val, y_val)
    log_to_mlflow(model, model.get_params(), metrics, X_val, y_val)
    save_model(model)
    mlflow.end_run()
    
    logger.info("Model training completed successfully")

if __name__ == "__main__":
    main()
