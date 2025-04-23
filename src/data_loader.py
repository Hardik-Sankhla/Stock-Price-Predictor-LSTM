# src/data_loader.py
import yfinance as yf
import pandas as pd
import os

def download_data(ticker="AAPL", start="2015-01-01", end="2021-01-01"):
    # Check if the 'data' directory exists, if not, create it
    if not os.path.exists('data'):
        os.makedirs('data')

    # Download stock data using yfinance
    df = yf.download(ticker, start=start, end=end)

    # Save the data as CSV in the 'data' folder
    df.to_csv(f"data/raw_stock_{ticker}.csv")

    return df
