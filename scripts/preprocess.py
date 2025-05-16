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
import uuid

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
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Created processed data directory.")

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
    # add custom name for run
    unique_id = uuid.uuid4()
    mlflow.set_tag('mlflow.runName', 'exp_preprocess_part_logs_'+str(unique_id))
    print(f"MLflow experiment set up: id {experiment_id}")


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


def analyze_data(df,train_df, val_df, test_df):
    """Perform exploratory data analysis and log results to MLflow."""
    mlflow.log_metric("OriginaL Data Number of samples", len(df))
    class_counts=df["Class"].value_counts()
    for class_label, count in class_counts.items():
        mlflow.log_metric(f"Original_Data_Class_{class_label}_count", int(count))

    mlflow.log_metric("Train data Number of samples", len(train_df))
    class_counts=train_df["Class"].value_counts()
    for class_label, count in class_counts.items():
        mlflow.log_metric(f"Train_Data_Class_{class_label}_count", int(count))

    mlflow.log_metric("Validation Number of samples", len(val_df))
    class_counts=val_df["Class"].value_counts()
    for class_label, count in class_counts.items():
        mlflow.log_metric(f"Validation_Data_Class_{class_label}_count", int(count))

    mlflow.log_metric("Test Number of samples", len(test_df))
    class_counts=test_df["Class"].value_counts()
    for class_label, count in class_counts.items():
        mlflow.log_metric(f"Test_Data_Class_{class_label}_count", int(count))

    logger.info("Logged dataset statistics to MLflow.")


def preprocess_data(df):
    """Preprocess the dataset."""
    # TODO: Implement this function to:
    # 1. Handle class imbalance (e.g., using SMOTE)
    # 2. Normalize features
    # 3. Split into train/validation/test sets
    # Check class distribution

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
    #print(df_undersampled["Class"].value_counts())
    y = df_undersampled["Class"]
    y=y.to_numpy()
    temp_df=df_undersampled
    temp_df=temp_df.drop(columns=["Class"])
    column_names=temp_df.columns
    X = df_undersampled.drop(columns=["Class"]).to_numpy()


    #Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    #X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled,y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)
        
    # Convert to DataFrame for saving
    train_df = pd.DataFrame(X_train, columns=column_names)

    train_df["Class"] = y_train.astype("int")
    val_df = pd.DataFrame(X_val, columns=column_names)
    val_df["Class"] = y_val.astype("int")
    test_df = pd.DataFrame(X_test, columns=column_names)
    test_df["Class"] = y_test.astype("int")

    logger.info("Preprocessed data and split into train, validation, and test sets.")
    return train_df,val_df,test_df


def save_processed_data(train_df, val_df, test_df):
    """Save processed datasets and track with DVC."""
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    ##-------- Track processed datasets with DVC

    try:
        # create new dataset version
        subprocess.run(["dvc", "add", str(train_path)], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("Error dvc adding skiping this part")
    try:
        new_dvc_file=str(train_path)+".dvc"
        print("git commits for  DVC setup started")
        # create dvc files and  Commit changes for train data commit (you should use Git and configure your own credentials)
        subprocess.run(["git", "add", ".dvc", new_dvc_file], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("Error git commiting the dvc skiping this part")

    
    try:
        # create new dataset version for val data
        subprocess.run(["dvc", "add", str(val_path)], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("Error dvc adding skiping this part")
    try:
        new_dvc_file=str(val_path)+".dvc"
        print("git commits for  DVC setup started")
        # create dvc files and  Commit changes for val. data commit (you should use Git and configure your own credentials)
        subprocess.run(["git", "add", ".dvc", new_dvc_file], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("Error git commiting the dvc skiping this part")
    

    try:
        # create new dataset version for test data
        subprocess.run(["dvc", "add", str(test_path)], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("Error dvc adding skiping this part")
    try:
        new_dvc_file=str(test_path)+".dvc"
        print("git commits for  DVC setup started")
        # create dvc files and  Commit changes for test data commit (you should use Git and configure your own credentials)
        subprocess.run(["git", "add", ".dvc", new_dvc_file], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("Error git commiting the dvc skiping this part")
    

    ##--------  Commit dvcs,Versioning and push datasets
    try:
    #newly data dvcs commit
        commit_message=str("newly created datasets(train,val,test) loaded to dvc")
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        logger.info(" git commits is done.")
    except subprocess.CalledProcessError as e:
        logger.warning("Error git commiting the dvc skiping this part")


    try:
        #create version tag for the newly created data after commiting them
        print("Data is versioning...")
        subprocess.run(["git", "tag", "localv_1_processed"], check=True)
        logger.info("Data versioning is done")
    except subprocess.CalledProcessError as e:
        logger.warning("Error git versioning skiping this part")
    


    try:
    #push data to storage with dvc to s3 minio bucket
        print("Datasets are pushing to storage...")
        subprocess.run(["dvc", "push", "-r","minio"], check=True)
        logger.info("Data pushing is done")
    except subprocess.CalledProcessError as e:
        logger.warning("Error dvc pushing skiping this part")
    #create version tag for the initial data

    logger.info("Processed datasets saved and tracked with DVC.")

def log_to_mlflow(stats, train_df, val_df, test_df):
    """Log preprocessing results and statistics to MLflow."""
    mlflow.log_params(stats)
    mlflow.log_metric("Train size", len(train_df))
    mlflow.log_metric("Validation size", len(val_df))
    mlflow.log_metric("Test size", len(test_df))
    logger.info("Logged preprocessing stats to MLflow.")

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
    train_df, val_df, test_df = preprocess_data(df)

    analyze_data(df,train_df, val_df, test_df)
    save_processed_data(train_df, val_df, test_df)
    stats = {"DVC Revision": args.data_rev}
    log_to_mlflow(stats, train_df, val_df, test_df)
    mlflow.end_run()



    logger.info("Data preprocessing completed successfully")

if __name__ == "__main__":
    main()
