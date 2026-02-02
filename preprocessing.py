"""
preprocessing.py
----------------
Handles everything from raw CSV to a clean, model-ready DataFrame:
  - Loading & initial inspection
  - Missing value imputation
  - Outlier handling
  - Categorical encoding (one-hot / ordinal)
  - Numerical scaling
All logic is encapsulated in the Preprocessor class so that the same
fitted transformations can be applied identically to train and test sets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Optional

from config import TARGET_COLUMN, RANDOM_STATE


class Preprocessor:
    """Fit on training data, then transform train / val / test consistently."""

    def __init__(self):
        self.num_imputer: Optional[SimpleImputer] = None
        self.cat_imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: dict = {}
        self.num_cols: list = []
        self.cat_cols: list = []

    # ── public API ────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit all transformers on *df* and return the cleaned copy."""
        df = df.copy()
        self._identify_columns(df)
        df = self._impute(df, fit=True)
        df = self._encode_categoricals(df, fit=True)
        df = self._scale_numericals(df, fit=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted transformers to *df*."""
        df = df.copy()
        df = self._impute(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        df = self._scale_numericals(df, fit=False)
        return df

    # ── internals ─────────────────────────────────────────────────
    def _identify_columns(self, df: pd.DataFrame) -> None:
        """Separate numerical and categorical feature columns (exclude target)."""
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if TARGET_COLUMN in self.num_cols:
            self.num_cols.remove(TARGET_COLUMN)
        if TARGET_COLUMN in self.cat_cols:
            self.cat_cols.remove(TARGET_COLUMN)
        print(f"[Preprocessor] Numerical features : {len(self.num_cols)}")
        print(f"[Preprocessor] Categorical features: {len(self.cat_cols)}")

    def _impute(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Impute missing values: median for numerical, mode for categorical."""
        if fit:
            self.num_imputer = SimpleImputer(strategy="median")
            self.cat_imputer = SimpleImputer(strategy="most_frequent")

        if self.num_cols:
            df[self.num_cols] = (
                self.num_imputer.fit_transform(df[self.num_cols])
                if fit
                else self.num_imputer.transform(df[self.num_cols])
            )
        if self.cat_cols:
            df[self.cat_cols] = (
                self.cat_imputer.fit_transform(df[self.cat_cols])
                if fit
                else self.cat_imputer.transform(df[self.cat_cols])
            )
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Label-encode low-cardinality categoricals.
        Swap this block for pd.get_dummies() if one-hot is preferred.
        """
        for col in self.cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # map unseen labels to -1
                df[col] = df[col].astype(str).map(
                    dict(zip(le.classes_, le.transform(le.classes_)))
                ).fillna(-1).astype(int)
        return df

    def _scale_numericals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """StandardScaler on numerical features (excluding target)."""
        if not self.num_cols:
            return df
        if fit:
            self.scaler = StandardScaler()
            df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])
        else:
            df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and print a quick summary."""
    print(f"[load_data] Reading {filepath} …")
    df = pd.read_csv(filepath)
    print(f"[load_data] Shape: {df.shape}")
    print(df.info(memory_usage="deep"))
    return df