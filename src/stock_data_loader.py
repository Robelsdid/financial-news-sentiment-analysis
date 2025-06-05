import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def load_stock_data(tickers, data_dir="../data", start_date=None, end_date=None):
    """
    Load historical stock data for multiple tickers from CSV files.
    
    Args:
        tickers (list): List of stock ticker symbols
        data_dir (str): Directory containing the CSV files
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        
    Returns:
        dict: Dictionary of DataFrames containing stock data for each ticker
    """
    stock_data = {}
    
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker}_historical_data.csv")
        if not os.path.exists(file_path):
            print(f"Warning: No data file found for {ticker}")
            continue
            
        try:
            # Read CSV with explicit date parsing
            df = pd.read_csv(file_path, parse_dates=['Date'])
            
            # Drop any duplicate dates, keeping the first occurrence
            df = df.drop_duplicates(subset=['Date'], keep='first')
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Set date as index
            df.set_index('Date', inplace=True)
            
            # Filter by date range if specified
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            # Store in dictionary
            stock_data[ticker] = df
            
            print(f"Loaded {ticker} data: {len(df)} rows from {df.index.min()} to {df.index.max()}")
            
        except Exception as e:
            print(f"Error loading data for {ticker}: {str(e)}")
            continue
    
    return stock_data

def update_stock_data(tickers, data_dir="../data", days_back=1):
    """
    Update stock data for multiple tickers by downloading the most recent data.
    
    Args:
        tickers (list): List of stock ticker symbols
        data_dir (str): Directory containing the stock data CSV files
        days_back (int): Number of days of historical data to download
    
    Returns:
        dict: Dictionary with ticker symbols as keys and DataFrames as values
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    return load_stock_data(tickers, data_dir, start_date, end_date)

def get_stock_info(ticker):
    """
    Get additional information about a stock using Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Dictionary containing stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant information
        stock_info = {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
            '52_week_low': info.get('fiftyTwoWeekLow', 0),
            'avg_volume': info.get('averageVolume', 0)
        }
        
        return stock_info
    except Exception as e:
        print(f"Error getting info for {ticker}: {e}")
        return None 
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