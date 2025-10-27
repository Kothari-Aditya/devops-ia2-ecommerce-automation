"""
Data loading module - loads raw CSV data.
Extracted from Jupyter notebook.
"""

import pandas as pd
import os
from config.config import RAW_DATA_PATH


def load_data(file_path=None):
    """
    Load the supply chain dataset.
    Uses exact loading logic from notebook.
    
    Args:
        file_path: Path to CSV file. If None, uses config default.
    
    Returns:
        pandas DataFrame with raw data
    """
    if file_path is None:
        file_path = RAW_DATA_PATH
    
    # Exact loading from notebook (Cell 2)
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    print(f"Data loaded successfully!")
    print(f"Shape: {data.shape}")
    print(f"Columns: {len(data.columns)}")
    
    return data


if __name__ == "__main__":
    # Test the loader
    df = load_data()
    print("\nFirst few rows:")
    print(df.head())