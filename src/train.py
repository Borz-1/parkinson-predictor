from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from src.model import build_model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.utils import resample
from tensorflow.keras.optimizers import Adam, RMSprop, SGD




def train_model(X_train, X_val, X_test, y_train, y_val, y_test, input_shape):
    model = build_model(input_shape, use_batchnorm=True)
    
    early_stop = EarlyStopping(monitor='val_recall', mode='max', patience=5, restore_best_weights=True)
    
    # connaître les poids pour le déséquilibre
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    print("class_weight : ", class_weight)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs = 50,
        batch_size=16,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=1,
        shuffle=False
    )
    
    # Evalutation
    loss, acc, recall = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {acc:.4f}, recall: {recall:.4f}")
    
    # Prédictions et rapport
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))
    
    print(confusion_matrix(y_test, y_pred))
    
    return model, history



def train_simple_model(X_train, X_val, y_train, y_val, input_shape, params=None):
    
    if params is None:
        params = {}
        
    # Séparer les paramètres
    model_params = {k: params[k] for k in ['hidden_layers', 'dropout_rate', 'use_batchnorm', 'activation']}
    
    
    model = build_model(input_shape, **model_params)
    
    optimizers = {
        'adam': Adam(),
        'rmsprop': RMSprop(),
        'sgd': SGD()
        }
    
    optimizer = optimizers.get(params.get('optimizer', 'adam'), Adam())

    
    model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'Recall']
    )
    
    # connaître les poids pour le déséquilibre
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    early_stop = EarlyStopping(monitor='val_recall', mode='max', patience=5, restore_best_weights=True)

    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs = 50,
        batch_size=params.get('batch_size', 32),
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=0,
        shuffle=False
    )
    
    return model, history


def cross_validate_model(X, y, n_splits, use_pca=True, params={}):
    accuracies = []
    recalls = []
    aucs = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    i = 0
    for train_idx, val_idx in kf.split(X):

        print(f"Fold {i+1}")
        i += 1
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        if use_pca:
            pca = PCA(n_components=3)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_val_scaled = pca.transform(X_val_scaled)
            
        model, history = train_simple_model(X_train_scaled, X_val_scaled, y_train, y_val, X_train_scaled.shape[1], params)
        
        y_pred = model.predict(X_val_scaled)
    
        
        accuracies.append(accuracy_score(y_val, y_pred > 0.5))
        recalls.append(recall_score(y_val, y_pred > 0.5))
        aucs.append(roc_auc_score(y_val, y_pred))

    print("\nRésumé cross-validation :")
    print(f"Accuracy : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Recall    : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"AUC       : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")


    return accuracies, recalls, aucs


# Bootstrap, variante de la crossvalidation idéale pour peu de données. 
# Tirage avec remises pour l'entraînement, les données restante pour le test 
def bootstrap_validate_model(X, y, n_iterations=100, use_pca=True, params={}):
    scores = []
    
    for i in range(n_iterations):
        
        # tirage avec remise
        X_train, y_train = resample(X, y, replace=True, random_state=0)
        
        # Données non tirés
        test_mask = ~np.isin(np.arange(len(X)), X_train)
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        # Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if use_pca:
            pca = PCA(n_components=3)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)

        model, _ = train_simple_model(X_train_scaled, X_test_scaled, y_train, y_test, X_train_scaled.shape[1], params)
        y_pred = model.predict(X_test_scaled)

        score = recall_score(y_test, y_pred > 0.5)
        scores.append(score)

    print(f"Recall moyen bootstrap : {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores
    
    