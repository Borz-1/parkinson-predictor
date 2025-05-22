# Preprocessing des données
# Importer les librairies
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def load_and_preprocess_data(path, pca=False):
    SEED = 0
    
    # test du chemin
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not find at {path}")
    
    # importer le dataset
    df = pd.read_csv(path)
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Split en X et y sans garder status et name
    X = df.drop(columns=["status", "name"])
    y = df["status"]

    # Split test (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)

    # Split validation
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=SEED)
    
    # Normaliser les données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    if pca:        
        pca = PCA(n_components=3)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_val_scaled = pca.transform(X_val_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
    
        
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
