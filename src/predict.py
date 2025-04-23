# src/predict.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.preprocessing import create_sequences
import pandas as pd

def predict_stock():
    model = load_model("models/lstm_model.h5")
    scaler = joblib.load("data/scaler.save")
    df = pd.read_csv("data/raw_stock.csv")
    close_prices = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)
    X, y = create_sequences(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    predictions = model.predict(X)
    predicted_prices = scaler.inverse_transform(predictions)
    return predicted_prices[-1], scaler.inverse_transform(y.reshape(-1, 1))[-1]
