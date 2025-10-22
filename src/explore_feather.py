#!/usr/bin/env python3
"""
Script to explore feather file structure and keys.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def explore_feather_file(file_path):
    """
    Explore a feather file and print its structure and keys.
    
    Args:
        file_path: Path to the feather file
    """
    try:
        # Read the feather file
        print(f"Reading feather file: {file_path}")
        df = pd.read_feather(file_path)
        
        print(f"\n=== Feather File Information ===")
        print(f"File path: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        print(f"\n=== Column Details ===")
        for col in df.columns:
            print(f"\nColumn: {col}")
            print(f"  Type: {df[col].dtype}")
            print(f"  Non-null count: {df[col].count()}")
            print(f"  Null count: {df[col].isnull().sum()}")
            
            if df[col].dtype in ['object', 'string']:
                print(f"  Unique values: {df[col].nunique()}")
                if df[col].nunique() <= 10:
                    print(f"  Values: {df[col].unique()}")
            elif np.issubdtype(df[col].dtype, np.number):
                print(f"  Min: {df[col].min()}")
                print(f"  Max: {df[col].max()}")
                print(f"  Mean: {df[col].mean():.4f}")
                print(f"  Std: {df[col].std():.4f}")
        
        print(f"\n=== Sample Data (first 5 rows) ===")
        print(df.head())
        
        print(f"\n=== Memory Usage ===")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return df
        
    except Exception as e:
        print(f"Error reading feather file: {e}")
        return None

def main():
    # File path
    file_path = "/workspace/av2data/val/0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2/annotations.feather"
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"File does not exist: {file_path}")
        print("Available directories in /workspace/av2data/train/:")
        try:
            train_dir = Path("/workspace/av2data/train/")
            if train_dir.exists():
                for item in train_dir.iterdir():
                    print(f"  {item.name}")
            else:
                print("  /workspace/av2data/train/ does not exist")
        except Exception as e:
            print(f"Error listing directories: {e}")
        return
    
    # Explore the feather file
    df = explore_feather_file(file_path)
    
    if df is not None:
        print(f"\n=== Summary ===")
        print(f"Successfully loaded feather file with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {', '.join(df.columns)}")

if __name__ == "__main__":
    main()
