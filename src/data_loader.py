import pandas as pd
import os

def load_data(filename="raw_analyst_ratings.csv", data_dir="data"):
    """
    Loads a CSV file from the specified data directory.

    Args:
        filename (str): Name of the CSV file.
        data_dir (str): Directory where the data file is stored.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    file_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ {file_path} does not exist.")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded: {file_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        raise
