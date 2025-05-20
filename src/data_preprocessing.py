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


    # Split test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=SEED)

    # Split validation
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=SEED)
        
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
