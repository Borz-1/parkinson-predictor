from src.data_preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("data/parkinsons/parkinsons.data")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


