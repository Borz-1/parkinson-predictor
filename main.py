from src.data_preprocessing import load_and_preprocess_data
from src.train import train_simple_model, cross_validate_model
from src.model import build_model
from src.visualization import plot_history, plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import classification_report
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import os
import gc


# Noyau pour que le modèle soit plus prévisible
def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(0)

# éviter que les paramètres de modèles éxecuter à la suite se mélangent.
def reset_session(seed=0):
    K.clear_session()
    gc.collect()
    set_seed(seed)
    
    

# Division en train, val et test
path = "data/parkinsons/parkinsons.data"
X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data(path)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")



def run_model(X_train, X_val, X_test, y_train, y_val, y_test, input_shape, label='', params=None):
    print(f"\n-- Modèle : {label} ---")
    # Entraînement du modèle
    model, history = train_simple_model(X_train, X_val, y_train, y_val, input_shape, params)
    
    # Affichage de l'évolution des métriques à travers les epochs
    plot_history(history)

    # Affichage de la matrice de confusion
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    plot_confusion_matrix(y_test, y_pred, labels=["sain", "parkinson"])
    print(classification_report(y_test, y_pred))

    # Probabilités entre 0 et 1
    y_proba = model.predict(X_test).ravel()
    # Affichage ROC
    plot_roc_curve(y_test, y_proba, model_name="Parkinson Predictor")
    
    return model, history
    

# top 2 et top 3 modèles
"""
Top 2
Params : {'hidden_layers': [128], 'dropout_rate': 0.2, 'use_batchnorm': True, 'activation': 'tanh', 'optimizer': 'rmsprop', 'batch_size': 16}
Accuracy : 0.8103
Recall :   0.9103
AUC :      0.7843

Top 3
Params : {'hidden_layers': [128], 'dropout_rate': 0.4, 'use_batchnorm': True, 'activation': 'relu', 'optimizer': 'adam', 'batch_size': 16}
Accuracy : 0.8205
Recall :   0.8976
AUC :      0.8098
"""

params = {
    'hidden_layers': [128],
    'dropout_rate': 0.2,
    'use_batchnorm': True,
    'activation': 'tanh',
    'optimizer': 'rmsprop',
    'batch_size': 16
}


# Données PCA
X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, _ = load_and_preprocess_data(path, pca=True)
model_pca, history_pca = run_model(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, X_train_pca.shape[1], label='Avec PCA', params=params)


print("\nRésumé des résultats :")
print(f"{'Modèle':<12} | {'Accuracy':<9} | {'Recall':<7}")
print("-" * 35)
print(f"{'Avec PCA':<12} | {model_pca.evaluate(X_test_pca, y_test)[1]:.4f}    | {model_pca.evaluate(X_test_pca, y_test)[2]:.4f}")


model_pca.save("models/meilleur_modele_parkinson.h5")
