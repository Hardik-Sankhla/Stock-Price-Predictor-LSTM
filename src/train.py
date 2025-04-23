# src/train.py
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib
from src.data_loader import download_data
from src.preprocessing import preprocess_data, create_sequences
from src.model import build_model

# Load and preprocess
download_data()
data = preprocess_data()
X, y = create_sequences(data)
X = X.reshape(X.shape[0], X.shape[1], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train
model = build_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)
model.save("models/lstm_model.h5")