"""
data.py - Data loading and feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split

from .config import TARGET_COLUMN, RANDOM_STATE, TRAIN_SIZE, EXTRACT_PATTERNS, MIN_YEAR


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw CSV with optimized dtypes."""
    print(f"\n[Data] Loading {filepath}...")
    dtype_map = {
        'Model Series': 'object',
        'Couple System': 'object',
        'Grouser Tracks': 'object',
        'Hydraulics Flow': 'object'
    }
    df = pd.read_csv(filepath, low_memory=False, dtype=dtype_map)
    print(f"[Data] Loaded {len(df):,} rows")
    return df


def extract_numeric_from_text(series: pd.Series, pattern: str) -> pd.Series:
    """Extract numeric values from text using regex (vectorized)."""
    return series.str.extract(pattern, expand=False).astype(float)


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering."""
    print("\n[Data] Feature engineering...")
    df = df.copy()
    
    # Fix unrealistic years
    df['Is_Year_Placeholder'] = (df['Year Made'] == 1000).astype(int)
    mask = df['Year Made'] < MIN_YEAR
    df.loc[mask, 'Year Made'] = np.nan
    
    # Text mining
    for name, pattern in EXTRACT_PATTERNS.items():
        df[name] = extract_numeric_from_text(df['Product Class Description'], pattern)
    
    # Temporal features
    sale_date = pd.to_datetime(df['Sales date'], errors='coerce')
    df['Sale_Date'] = sale_date
    df['Sale_Year'] = sale_date.dt.year
    df['Sale_Month'] = sale_date.dt.month
    df['Sale_Quarter'] = sale_date.dt.quarter
    
    # Derived features
    df['Machine_Age'] = df['Sale_Year'] - df['Year Made']
    df['Has_Hours'] = df['MachineHours CurrentMeter'].notna().astype(int)
    
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test (70/15/15)."""
    print("[Data] Splitting (70/15/15)...")
    train_df, temp = train_test_split(df, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=RANDOM_STATE)
    print(f"[Data] Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    return train_df, val_df, test_df


def load_and_split(filepath: Path, return_separated: bool = False) -> Dict:
    """Main entry point: load → engineer → split."""
    df = load_raw_data(filepath)
    df = engineer_all_features(df)
    train_df, val_df, test_df = split_data(df)
    
    if return_separated:
        return {
            'X_train': train_df.drop(columns=[TARGET_COLUMN]),
            'y_train': train_df[TARGET_COLUMN],
            'X_val': val_df.drop(columns=[TARGET_COLUMN]),
            'y_val': val_df[TARGET_COLUMN],
            'X_test': test_df.drop(columns=[TARGET_COLUMN]),
            'y_test': test_df[TARGET_COLUMN],
        }
    return {'train': train_df, 'val': val_df, 'test': test_df}