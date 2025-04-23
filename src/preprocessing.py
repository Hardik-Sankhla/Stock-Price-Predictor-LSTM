# src/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data(filename="data/raw_stock.csv"):
    df = pd.read_csv(filename)
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    joblib.dump(scaler, "data/scaler.save")
    return scaled_data

def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
