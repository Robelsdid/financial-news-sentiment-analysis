import pandas as pd
import os

def load_stock_data(tickers, data_dir="data"):
    """
    Loads historical stock data for a list of tickers from CSV files in the specified directory.
    Returns a dictionary of DataFrames keyed by ticker.
    """
    stock_data = {}
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker}_historical_data.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            stock_data[ticker] = df
        else:
            print(f"File not found for {ticker} (expected: {file_path})")
    return stock_data