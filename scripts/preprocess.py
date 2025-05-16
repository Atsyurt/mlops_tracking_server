#!/usr/bin/env python3
"""
Data preprocessing script for Credit Card Fraud Detection MLOps Pipeline.

This script:
1. Loads a specific version of raw data from DVC
2. Handles class imbalance
3. Normalizes features
4. Splits data into train/validation/test sets
5. Saves processed datasets back to DVC
6. Logs preprocessing steps to MLflow

Usage:
    python preprocess.py --data-rev <DVC_REVISION>
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import subprocess
from sklearn.utils import resample
import mlflow
# Import additional libraries as needed (e.g., imbalanced-learn for SMOTE)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('data-preprocessing')

# Constants
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "creditcard.csv"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data preprocessing script')
    parser.add_argument('--data-rev', type=str, required=True,
                        help='DVC revision/version of the raw data to use')
    # Add more arguments as needed
    return parser.parse_args()

def setup_directories():
    """Create necessary directories if they don't exist."""
    # TODO: Implement this function
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Created processed data directory.")

def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_experiment("creditcard_fraud_detection")
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
    # Set the MLflow tracking username who is created this work
    os.environ['MLFLOW_TRACKING_USERNAME'] = "user"
    logger.info("MLflow tracking set up.")


def load_data(data_rev):
    """Load raw data from specific DVC revision."""
    try:
        # Checkout the specific DVC version
        subprocess.run(["git", "checkout", data_rev], check=True)
        subprocess.run(["dvc", "checkout"], check=True)
        logger.info(f"Checked out data version: {data_rev}")

        # Load the dataset
        df = pd.read_csv(RAW_DATA_FILE)
        logger.info(f"Data loaded successfully from {RAW_DATA_FILE}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def analyze_data(df):
    """Perform exploratory data analysis and log results to MLflow."""
    mlflow.log_metric("Number of samples", len(df))
    mlflow.log_metric("Class distribution", df["Class"].value_counts().to_dict())
    logger.info("Logged dataset statistics to MLflow.")


def preprocess_data(df):
    """Preprocess the dataset."""
    # TODO: Implement this function to:
    # 1. Handle class imbalance (e.g., using SMOTE)
    # 2. Normalize features
    # 3. Split into train/validation/test sets
    # Check class distribution
    print("Original class distribution:")
    print(df["Class"].value_counts())

    # Separate majority and minority classes
    df_majority = df[df["Class"] == 0]  # Non-fraudulent transactions
    df_minority = df[df["Class"] == 1]  # Fraudulent transactions

    # Apply undersampling: Reduce majority class to match minority class size
    df_majority_undersampled = resample(df_majority, 
                                        replace=False,    # Sample without replacement
                                        n_samples=len(df_minority),  # Match minority class size
                                        random_state=42)  # Reproducibility

    # Combine minority class with undersampled majority class
    df_undersampled = pd.concat([df_majority_undersampled, df_minority])

    # Check new class distribution
    print("\nNew class distribution after undersampling:")
    print(df_undersampled["Class"].value_counts())
    X = df_undersampled.drop(columns=["Class"])
    y = df_undersampled["Class"]

    #Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
        
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled,y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
    # Convert to DataFrame for saving
    train_df = pd.DataFrame(X_train, columns=X.columns)
    train_df["Class"] = y_train
    val_df = pd.DataFrame(X_val, columns=X.columns)
    val_df["Class"] = y_val
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df["Class"] = y_test

    logger.info("Preprocessed data and split into train, validation, and test sets.")
    pass

def save_processed_data(train_df, val_df, test_df):
    """Save processed datasets and track with DVC."""
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Track processed data with DVC
    subprocess.run(["dvc", "add", str(train_path)], check=True)
    subprocess.run(["dvc", "add", str(val_path)], check=True)
    subprocess.run(["dvc", "add", str(test_path)], check=True)

    subprocess.run(["dvc", "push", "-r","minio"], check=True)

    #create version tag for the initial data
    print("Data is versioning...")
    subprocess.run(["git", "tag", "localv_1_processed"], check=True)
    logger.info("Data versioning is done")
    #push data to sotarage
    print("Datasets are pushing to storage...")
    subprocess.run(["dvc", "push", "-r","minio"], check=True)
    logger.info("Data pushing is done")

    logger.info("Processed datasets saved and tracked with DVC.")

def log_to_mlflow(stats, train_df, val_df, test_df):
    """Log preprocessing results and statistics to MLflow."""
    mlflow.log_params(stats)
    mlflow.log_metric("Train size", len(train_df))
    mlflow.log_metric("Validation size", len(val_df))
    mlflow.log_metric("Test size", len(test_df))
    logger.info("Logged preprocessing stats to MLflow.")



    
    pass

def main():
    """Main function to orchestrate the data preprocessing pipeline."""
    args = parse_args()
    logger.info(f"Starting data preprocessing pipeline with data revision: {args.data_rev}")
    
    # TODO: Implement the main workflow
    # 1. Setup directories and MLflow
    # 2. Load data from specific DVC revision
    # 3. Analyze data
    # 4. Preprocess data
    # 5. Save processed data
    # 6. Log results to MLflow
    setup_directories()
    setup_mlflow()
    df=load_data(args.data_rev)
    analyze_data(df)
    train_df, val_df, test_df = preprocess_data(df)
    save_processed_data(train_df, val_df, test_df)
    stats = {"DVC Revision": args.data_rev}
    log_to_mlflow(stats, train_df, val_df, test_df)



    logger.info("Data preprocessing completed successfully")

if __name__ == "__main__":
    main()
