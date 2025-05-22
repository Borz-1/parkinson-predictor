from src.train import cross_validate_model
import pandas as pd
import numpy as np

# Chargement des données
df = pd.read_csv("data/parkinsons/parkinsons.data")
X = df.drop(columns=["name", "status"]).values
y = df["status"].values

# Validation croisée avec PCA
accuracies, recalls, aucs = cross_validate_model(X, y, n_splits=5, use_pca=True)

print(f"moyenne accuracy : {np.mean(accuracies)}")
print(f"moyenne rappelle : {np.mean(recalls)}")
print(f"moyenne auc : {np.mean(aucs)}")

