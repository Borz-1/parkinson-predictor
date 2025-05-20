from src.data_preprocessing import load_and_preprocess_data
from src.train import train_model
from src.model import build_model
import random
import numpy as np
import tensorflow as tf
import os

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)


path = "data/parkinsons/parkinsons.data"
X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data(path)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


model, history = train_model(X_train, X_val, X_test, y_train, y_val, y_test, X_train.shape[1])

X_val.shape
y_val.shape






