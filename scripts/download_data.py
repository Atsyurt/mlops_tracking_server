import requests
import pandas as pd
from pathlib import Path


import requests
import pandas as pd
from pathlib import Path

# URL of the dataset
url = "https://nextcloud.scopicsoftware.com/s/bo5PTKgpngWymGE/download/creditcard-data.csv"

# Define a cross-platform path for saving the dataset
dataset_dir = Path("./dataset")
dataset_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

dataset_path = dataset_dir / "creditcard-data.csv"

# Download the dataset
response = requests.get(url)
if response.status_code == 200:
    with open(dataset_path, "wb") as file:
        file.write(response.content)
    print(f"Download successful! File saved at: {dataset_path}")
else:
    print("Failed to download dataset.")

# Load the dataset into a pandas DataFrame
df = pd.read_csv(dataset_path)

# Display the first few rows
print(df.head())