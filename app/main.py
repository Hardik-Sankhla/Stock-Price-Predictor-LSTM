# app/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import joblib
import yfinance as yf
from tensorflow.keras.models import load_model
from src.preprocessing import create_sequences
import numpy as np

app = FastAPI()

# Load model and scaler
model = load_model("models/lstm_model.h5")
scaler = joblib.load("data/scaler.save")

class PredictionResponse(BaseModel):
    predicted_price: float
    last_actual_price: float

@app.get("/")
def read_root():
    return {"message": "Stock Price Predictor API is running."}

@app.get("/predict", response_model=PredictionResponse)
def predict(ticker: str = Query("AAPL")):
    # Download latest stock data
    df = yf.download(ticker, start="2015-01-01", end=None)
    close_prices = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)
    X, y = create_sequences(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Predict
    prediction = model.predict(X)
    predicted_price = scaler.inverse_transform(prediction)[-1][0]
    last_actual = scaler.inverse_transform(y.reshape(-1, 1))[-1][0]

    return PredictionResponse(predicted_price=predicted_price, last_actual_price=last_actual)
