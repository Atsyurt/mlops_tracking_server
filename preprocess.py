import pandas as pd
from sklearn.utils import resample

# Load the dataset (Ensure you have 'creditcard.csv' in your working directory)
df = pd.read_csv("creditcard.csv")

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

# Save the new dataset
df_undersampled.to_csv("creditcard_undersampled.csv", index=False)
print("\nUndersampled dataset saved as 'creditcard_undersampled.csv'.")