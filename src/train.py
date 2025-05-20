from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from src.model import build_model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def train_model(X_train, X_val, X_test, y_train, y_val, y_test, input_shape):
    model = build_model(input_shape, use_batchnorm=True)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
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