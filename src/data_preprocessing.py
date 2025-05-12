# Data Preprocessing
# Importing Libraries
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(path):
    SEED = 0
    
    # test the path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not find at {path}")
    
    # importing the dataset
    df = pd.read_csv(path)
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Splitting the dataset into features and target variable
    # Features: all columns except 'status' and 'name'
    X = df.drop(columns=["status", "name"])
    y = df["status"]
    
    # Normalizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=SEED)
    
    return X_train, X_test, y_train, y_test, scaler
