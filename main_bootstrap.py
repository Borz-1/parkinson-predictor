from src.train import bootstrap_validate_model
from src.model import build_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Chargement des données
df = pd.read_csv("data/parkinsons/parkinsons.data")
X = df.drop(columns=["name", "status"]).values
y = df["status"].values

# Meilleurs hyperparamètres (issus de la recherche précédente)
meilleures_params = {
    'hidden_layers': [64, 32, 16],
    'dropout_rate': 0.2,
    'use_batchnorm': False,
    'activation': 'relu',
    'optimizer': 'adam',
    'batch_size': 32
}

# Séparation train/test pour évaluation finale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=0
)

# Évaluation Bootstrap
print("Évaluation bootstrap sur le train :")
bootstrap_validate_model(
    X_train, y_train,
    n_iterations=100,
    use_pca=True,
    params=meilleures_params
)

# Préparation des paramètres
build_params = {
    "hidden_layers": meilleures_params["hidden_layers"],
    "dropout_rate": meilleures_params["dropout_rate"],
    "use_batchnorm": meilleures_params["use_batchnorm"],
    "activation": meilleures_params["activation"],
    "optimizer": meilleures_params["optimizer"]
}

# Prétraitement Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Application de PCA
pca = PCA(n_components=3)
X_train_scaled = pca.fit_transform(X_train_scaled)
X_test_scaled = pca.transform(X_test_scaled)

# Entraînement final du modèle sur le train complet
model = build_model(X_train_scaled.shape[1], **build_params)

model.compile(
    loss='binary_crossentropy',
    optimizer=meilleures_params['optimizer'],
    metrics=['accuracy', 'Recall']
)

model.fit(X_train_scaled, y_train, epochs=50,
          batch_size=meilleures_params['batch_size'],
          verbose=0)

# Évaluation finale sur le jeu de test
loss, acc, recall = model.evaluate(X_test_scaled, y_test, verbose=0)
print("\nÉvaluation finale sur le jeu de test :")
print(f"Accuracy : {acc:.4f} | Recall : {recall:.4f}")
