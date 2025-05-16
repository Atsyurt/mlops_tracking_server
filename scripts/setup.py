#!/usr/bin/env python3
"""
Data acquisition script for Credit Card Fraud Detection MLOps Pipeline.

This script:
1. Downloads the Credit Card Fraud Detection dataset
2. Initializes DVC
3. Adds the raw data to DVC tracking
4. Pushes to the DVC remote
"""

import os
import sys
import logging
import requests
import hashlib
import pandas as pd
from pathlib import Path
import subprocess

import boto3




# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('data-acquisition')

# Constants
DATA_URL = "https://nextcloud.scopicsoftware.com/s/bo5PTKgpngWymGE/download/creditcard-data.csv"
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "creditcard-data.csv"
# Expected SHA256 checksum of the file (you should calculate this for your specific dataset)
EXPECTED_SHA256 = None

def delete_allfiles_from_s3():
    endpoint_url = "http://localhost:9000"  # Adjust if running remotely
    aws_access_key_id = "user"
    aws_secret_access_key = "WW4e69Wwcv0w"
    bucket_name = "dataset-bucket"

    # Create S3 client in order delete old files if any
    s3_client = boto3.client(
    "s3",
    endpoint_url=endpoint_url,  # MinIO URL
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
    )
    # List all objects in the bucket and delete them
    print("Old files are deleting...")
    try:
        objects = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if "Contents" in objects:
            for obj in objects["Contents"]:
                s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
                print(f"Deleted: {obj['Key']}")
            
            logger.info(f"All files in {bucket_name} have been deleted.")
        else:
            print(f"No files found in {bucket_name}.")
    except Exception as e:
        print(f"Error deleting files: {e}")




import os
import requests
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_URL = "https://example.com/dataset.csv"  # Replace with actual dataset URL






def setup_directories():
    """Create necessary directories if they don't exist."""
    # TODO: Implement this function
    print("Directory setup process...")
    dataset_dir = Path("./data")
    dataset_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    dataset_path = dataset_dir / "creditcard-data.csv"
    logger.info("Directory setup process is done.")
    return dataset_dir, dataset_path
    pass

def download_data(dataset_path):
    """Download the dataset from the source URL if not already available."""
    
    # Check if the dataset already exists
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}. Skipping download.")
        logger.info("Dataset already exists, skipping download.")
        return
    
    print("Data is downloading...")
    
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()  # Raise an error for unsuccessful requests
        
        with open(dataset_path, "wb") as file:
            file.write(response.content)
        
        print(f"Download successful! File saved at: {dataset_path}")
        logger.info("Data download process is done.")
    
    except requests.RequestException as e:
        logger.error(f"Failed to download dataset: {e}")

def validate_data(dataset_path):
    """Validate the downloaded data file integrity."""

    print("Data validaiton check test process ...")
    if not os.path.exists(dataset_path):
        logger.error(f"File '{dataset_path}' does not exist!")
        raise FileNotFoundError(f"File '{dataset_path}' does not exist.")
    
    if os.path.getsize(dataset_path) == 0:
        logger.error(f"dataset folder is empty")
        raise ValueError(f"File '{dataset_path}' is empty.")
    
    if not str(dataset_path).lower().endswith(".csv"):
        logger.error("The file is NOT a CSV!")
        raise ValueError("The file is NOT a CSV!")

    print(f"Validation passed: '{dataset_path}' exists and is not empty.")
        # Load file
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

    # Check for missing values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"Dataset contains {null_count} missing values!")
    else:
        logger.info("Validation passed: Data structure and integrity are valid.")
    # Check for expected columns
    # Expected columns
    expected_columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        logger.error(f"Missing expected columns: {missing_columns}")
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # Check for null values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"Dataset contains {null_count} missing values!")
    else:
        logger.info("Validation passed: Data structure and integrity are valid.")
        logger.info("Data validaiton check test process is done.")


import os
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_old_file_tracking(old_file_path):
    """Remove old file tracking from Git and DVC, handling errors gracefully."""
    if os.path.exists(old_file_path):
        try:
            # Remove from Git cache
            subprocess.run(["git", "rm", "--cached", str(old_file_path)], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error removing {old_file_path} from Git cache: {e}")

        try:
            # Remove from DVC tracking
            subprocess.run(["dvc", "remove", str(old_file_path)], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error removing {old_file_path} from DVC tracking: {e}")

        logger.info("Successfully removed old files from tracking")
    else:
        logger.warning(f"File '{old_file_path}' does not exist, skipping removal.")



def initialize_dvc(data_folder,data_path):
    """Initialize DVC and add data to tracking."""
    # TODO: Implement this function

    print("DVC Initialize process ...")
    subprocess.run(["dvc", "init","-f"], check=True)

    if os.path.exists(data_folder):
        #file+".dvc"
        old_file_path=str(data_path)+".dvc"
        try:
            # Remove from Git cache
            subprocess.run(["git", "rm", "--cached", str(old_file_path)], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error removing {old_file_path} from Git cache: {e}")

        try:
            # Remove from DVC tracking
            subprocess.run(["dvc", "remove", str(old_file_path)], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error removing {old_file_path} from DVC tracking: {e}")

        logger.info("Successfully removed old files from tracking")
    else:
        logger.warning(f"File '{old_file_path}' does not exist, skipping removal.")


    print("DVC setup begins initially for original data from scratch...")
    if os.path.exists(data_path):
        subprocess.run(["dvc", "add", data_path], check=True)
        subprocess.run(["dvc", "remote", "add", "-d", "minio", "s3://dataset-bucket", "-f"])
        subprocess.run(["dvc", "remote", "modify", "minio", "access_key_id", "user"])
        subprocess.run(["dvc", "remote", "modify", "minio", "secret_access_key", "WW4e69Wwcv0w"])
        subprocess.run(["dvc", "remote", "modify", "minio", "endpointurl", "http://localhost:9000"])
        logger.info("DVC setup is done.")
    else:
        print(f"Data path '{data_path}' does not exist. Please check your setup.")
    
    new_dvc_file=str(data_path)+".dvc"
    print("git commits for  DVC setup started")
    # create dvc files and  Commit changes for initial commit (you should use Git and configure your own credentials)
    subprocess.run(["git", "add", ".dvc", new_dvc_file], check=True)
    subprocess.run(["git", "commit", "-m", "Initialize DVC and add data"], check=True)
    logger.info(" git commits is done.")
    #create version tag for the initial data
    print("Data is versioning...")
    subprocess.run(["git", "tag", "localv_1"], check=True)
    logger.info("Data versioning is done")
    #push data to sotarage
    print("Data is pushing to storage")
    subprocess.run(["dvc", "push", "-r","minio"], check=True)
    logger.info("Data pushing is done")


    pass

def main():
    """Main function to orchestrate the data acquisition process."""
    logger.info("Starting data acquisition process")

    #Set up MinIO credentials and endpoint
    endpoint_url = "http://localhost:9000"  # Adjust if running remotely
    aws_access_key_id = "user"
    aws_secret_access_key = "WW4e69Wwcv0w"
    bucket_name = "models-bucket"

    
    # TODO: Implement the main workflow
    # 1. Create directories
    # 2. Download data
    # 3. Validate data
    # 4. Initialize DVC and add data
    delete_allfiles_from_s3()
    data_folder,dataset_path=setup_directories()
    download_data(dataset_path)
    validate_data(dataset_path)
    initialize_dvc(data_folder,dataset_path)
    logger.info("Data acquisition completed successfully")

if __name__ == "__main__":
    main()
