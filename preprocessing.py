import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Check if the file exists
file_path = "/Users/shubhanshu/Projects/RE1/data/data.csv"
if os.path.exists(file_path):
    print(f"File found at {file_path}")
    df = load_data(file_path)
else:
    print(f"File not found at {file_path}")

# Clean data (drop missing values)
def clean_data(df):
    # You could add more specific cleaning steps based on your data
    df = df.dropna()  # Remove rows with missing values
    return df

# Feature engineering (standardize all features)
def feature_engineering(df):
    scaler = StandardScaler()
    # Standardize all columns (except the target 'MEDV')
    feature_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

# Split data into training and testing sets
def split_data(df, target_column):
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target
    return train_test_split(X, y, test_size=0.2, random_state=42)
