# src/data_loader.py
import yfinance as yf
import pandas as pd

def download_data(ticker="AAPL", start="2015-01-01", end="2021-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df.to_csv("data/raw_stock.csv")
    return df