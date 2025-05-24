from src.train import cross_validate_model
import pandas as pd
import numpy as np
import itertools

df = pd.read_csv("data/parkinsons/parkinsons.data")
X = df.drop(columns=["name", "status"]).values
y = df["status"].values

# On va faire un GridSearchCV à la main
# Hyperparamètre à tester
hidden_layer_options = [[128], [128, 64], [64, 32, 16]]
dropout_rates = [0.2, 0.4]
batchnorm_options = [True, False]
activations = ['relu', 'tanh']
optimizers = ['adam', 'rmsprop']
batch_sizes = [32, 64]


results = []


# itertools.product() va créer toutes les combinaisons possibles
all_combinations = list(itertools.product(
    hidden_layer_options,
    dropout_rates,
    batchnorm_options,
    activations,
    optimizers,
    batch_sizes
))

results = []

# test de toutes les combinaisons
for i, (hl, dr, bn, act, opt, bs) in enumerate(all_combinations):
    print(f"\n🔍 Test {i+1}/{len(all_combinations)}:")
    params = {
        'hidden_layers': hl,
        'dropout_rate': dr,
        'use_batchnorm': bn,
        'activation': act,
        'optimizer': opt,
        'batch_size': bs
    }
    
    print("Params:", params)
    accuracies, recalls, aucs = cross_validate_model(X, y, n_splits=5, use_pca=True, params=params)
    
    results.append({
        "params": params,
        "accuracy": np.mean(accuracies),
        "recall": np.mean(recalls),
        "auc": np.mean(aucs)
    })


# Trier selon l'AUC ou une autre métrique
results_sorted = sorted(results, key=lambda x: x["recall"], reverse=True)

# Afficher les 3 meilleurs modèles
for i, res in enumerate(results_sorted[:3], 1):
    print(f"\nTop {i}")
    print(f"Params : {res['params']}")
    print(f"Accuracy : {res['accuracy']:.4f}")
    print(f"Recall :   {res['recall']:.4f}")
    print(f"AUC :      {res['auc']:.4f}")
    
    
    
df = pd.DataFrame(results)
df.to_csv("results/search_results.csv", index=False)
    
    
"""
Top 1
Params : {'hidden_layers': [64, 32, 16], 'dropout_rate': 0.2, 'use_batchnorm': False, 'activation': 'relu', 'optimizer': 'adam', 'batch_size': 32}
Accuracy : 0.8359
Recall :   0.8652
AUC :      0.8974

Top 2
Params : {'hidden_layers': [128, 64], 'dropout_rate': 0.4, 'use_batchnorm': True, 'activation': 'relu', 'optimizer': 'rmsprop', 'batch_size': 64}
Accuracy : 0.8410
Recall :   0.8548
AUC :      0.8929

Top 3
Params : {'hidden_layers': [128, 64], 'dropout_rate': 0.2, 'use_batchnorm': True, 'activation': 'tanh', 'optimizer': 'adam', 'batch_size': 32}
Accuracy : 0.8308
Recall :   0.8354
AUC :      0.9070
    
"""