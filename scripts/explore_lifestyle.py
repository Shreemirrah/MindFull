import pandas as pd

# Load the dataset
df = pd.read_csv("../data/alzheimers_disease_data.csv")
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 Rows:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDiagnosis Distribution:\n", df['Diagnosis'].value_counts())
# Copilot suggested this exploration structure