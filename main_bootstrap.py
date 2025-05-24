from src.train import bootstrap_validate_model
import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement des données
df = pd.read_csv("data/parkinsons/parkinsons.data")
X = df.drop(columns=["name", "status"]).values
y = df["status"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Lancement bootstrap
bootstrap_validate_model(X_train, y_train, n_iterations=100, use_pca=True, params={})