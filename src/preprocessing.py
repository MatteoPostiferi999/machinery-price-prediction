"""
preprocessing.py
================
Handles data loading, feature engineering, and preprocessing pipeline.
All transformations are modular and follow scikit-learn conventions.

Pipeline flow:
    1. Feature engineering (temporal, derived, text extraction)
    2. Train/Val/Test split (70/15/15)
    3. Fit preprocessor on training set only
    4. Transform all splits using fitted transformers (no leakage)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Set, Union
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from .config import (
    TARGET_COLUMN,
    TARGET_LOG,
    RANDOM_STATE,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    DROP_FEATURES,
    EXTRACT_PATTERNS,
    GROUP_IMPUTE_FEATURES,
    PRICE_LOWER_BOUND,
    PRICE_UPPER_BOUND,
    MIN_YEAR,
    TARGET_ENCODE_FEATURES,
    ONEHOT_FEATURES,
    RARE_LABEL_THRESHOLD
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize column names for LightGBM compatibility.
    Replaces special JSON characters with underscores and resolves duplicates.
    """
    # Vectorized replacement map using regex
    pattern = r'[\[\]<>"\'{},.:]'
    
    new_columns = df.columns.astype(str).str.replace(pattern, '_', regex=True)
    
    # Resolve duplicates efficiently
    seen: Dict[str, int] = {}
    final_columns = []
    
    for col in new_columns:
        if col in seen:
            seen[col] += 1
            final_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_columns.append(col)

    df.columns = final_columns
    return df


def load_data(filepath: Path) -> pd.DataFrame:
    """Load raw CSV data with optimized types where possible."""
    print(f"[load_data] Loading data from {filepath}...")
    
    # Specific dtypes to prevent mixed-type warnings
    dtype_map = {
        'Model Series': 'object',
        'Couple System': 'object',
        'Grouser Tracks': 'object',
        'Hydraulics Flow': 'object'
    }

    df = pd.read_csv(filepath, low_memory=False, dtype=dtype_map)
    print(f"[load_data] Loaded {len(df):,} rows x {len(df.columns)} columns")
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_numeric_from_text(series: pd.Series, pattern: str) -> pd.Series:
    """Extract numeric values from text using a regex pattern (Vectorized)."""
    return series.str.extract(pattern, expand=False).astype(float)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    df = df.copy()
    print("\n[engineer_features] Starting feature engineering...")

    # 1. Data Quality: Fix unrealistic years
    print(f"[engineer_features] Fixing Year Made < {MIN_YEAR}...")
    df['Is_Year_Placeholder'] = (df['Year Made'] == 1000).astype(int)
    
    mask_year = df['Year Made'] < MIN_YEAR
    print(f"[engineer_features]   -> Set {mask_year.sum():,} unrealistic years to NaN")
    df.loc[mask_year, 'Year Made'] = np.nan

    # 2. Text Mining: Extract technical specs (Vectorized)
    print("[engineer_features] Extracting numeric features from Product Class Description...")
    for feature_name, pattern in EXTRACT_PATTERNS.items():
        df[feature_name] = extract_numeric_from_text(df['Product Class Description'], pattern)
        pct = (df[feature_name].notna().mean()) * 100
        print(f"[engineer_features]   -> {feature_name}: {pct:.1f}% coverage")

    # 3. Temporal Features
    print("[engineer_features] Creating temporal features...")
    sale_date = pd.to_datetime(df['Sales date'], errors='coerce')
    df['Sale_Date'] = sale_date
    df['Sale_Year'] = sale_date.dt.year
    df['Sale_Month'] = sale_date.dt.month
    df['Sale_Quarter'] = sale_date.dt.quarter

    # 4. Derived Features
    print("[engineer_features] Creating derived features...")
    df['Machine_Age'] = df['Sale_Year'] - df['Year Made']
    df['Has_Hours'] = df['MachineHours CurrentMeter'].notna().astype(int)

    print("[engineer_features] Feature engineering complete!")
    return df


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Handles all preprocessing transformations with proper train/test separation.
    Inherits from sklearn BaseEstimator/TransformerMixin for pipeline compatibility.
    """

    def __init__(self):
        self.group_medians: Dict[str, pd.Series] = {}
        self.global_medians: Dict[str, float] = {}
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        
        # Encoding state
        self.target_encode_maps: Dict[str, pd.Series] = {}
        self.global_target_mean: float = 0.0
        self.known_categories: Dict[str, Set] = {}
        self.rare_categories_map: Dict[str, Set] = {}
        self.onehot_columns: List[str] = []
        
        # Consistency state (Fix for "Feature names must match" error)
        self.final_columns: List[str] = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, y=None) -> 'Preprocessor':
        """
        Fit preprocessing transformers on training data.
        """
        print("\n[Preprocessor.fit] Fitting preprocessing transformers...")
        df = df.copy()

        # Identify feature types
        self._identify_feature_types(df)

        # 1. Group-aware imputation (Vectorized GroupBy)
        cols_to_impute = [f for f in GROUP_IMPUTE_FEATURES if f in df.columns]
        if cols_to_impute and 'Product Group' in df.columns:
            print("[Preprocessor.fit] Calculating group-aware medians...")
            medians_df = df.groupby('Product Group')[cols_to_impute].median()
            for col in cols_to_impute:
                self.group_medians[col] = medians_df[col]

        # 2. Global Imputation Statistics (Tier 2 Fallback)
        if self.numeric_features:
            self.global_medians = df[self.numeric_features].median().to_dict()

        # 3. Target encoding maps
        if TARGET_COLUMN in df.columns:
            self.global_target_mean = df[TARGET_COLUMN].mean()
            te_features = [f for f in TARGET_ENCODE_FEATURES if f in df.columns]
            
            print(f"[Preprocessor.fit] Target encoding {len(te_features)} features...")
            for feature in te_features:
                self.target_encode_maps[feature] = df.groupby(feature)[TARGET_COLUMN].mean()

        # 4. Rare label detection
        ohe_features = [f for f in ONEHOT_FEATURES if f in df.columns]
        if ohe_features:
            print(f"[Preprocessor.fit] Detecting rare categories (Threshold={RARE_LABEL_THRESHOLD})...")
            df_sim = df[ohe_features].copy()
            for feature in ohe_features:
                self.known_categories[feature] = set(df[feature].dropna().unique())
                counts = df[feature].value_counts()
                rare_cats = set(counts[counts < RARE_LABEL_THRESHOLD].index)
                self.rare_categories_map[feature] = rare_cats
                if rare_cats:
                    mask_rare = df_sim[feature].isin(rare_cats)
                    df_sim.loc[mask_rare, feature] = 'Other'

            df_sim = df_sim.fillna('Unknown')
            dummies = pd.get_dummies(df_sim, columns=ohe_features, drop_first=True)
            self.onehot_columns = dummies.columns.tolist()
            print(f"[Preprocessor.fit]   -> Will create {len(self.onehot_columns)} OHE features")

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Apply preprocessing using fitted state.
        Ensures exact column order match between train and test.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        print(f"\n[Preprocessor.transform] Transforming data (is_train={is_train})...")
        df = df.copy()

        # 1. Target transformation (Training only)
        if is_train and TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(PRICE_LOWER_BOUND, PRICE_UPPER_BOUND)
            df[TARGET_LOG] = np.log1p(df[TARGET_COLUMN])

        # 2. Imputation
        self._apply_imputation(df)

        # 3. Drop Columns
        self._apply_column_drops(df)

        # 4. Categorical Encoding
        df = self._apply_encoding(df)

        # 5. Cleanup & Column Alignment (CRITICAL FIX)
        # Drop residual non-numerics (excluding target)
        numerics = df.select_dtypes(include=[np.number]).columns
        
        # Keep targets only if present
        targets = [c for c in [TARGET_COLUMN, TARGET_LOG] if c in df.columns]
        
        # If training, define the final feature structure
        if is_train:
            # Save the exact list of feature columns (excluding targets for consistency checks)
            self.final_columns = [c for c in numerics if c not in [TARGET_COLUMN, TARGET_LOG]]
            
            # Combine features + targets for the output dataframe
            final_selection = self.final_columns + targets
            df = df[final_selection]
            
        else:
            # If validation/test, FORCE the dataframe to match training columns exactly
            if self.final_columns is None:
                raise RuntimeError("Transform called on test data before train data defined final columns.")
            
            # Reindex ensures columns are in the exact same order as fit()
            # Missing columns are filled with 0, extra columns are dropped.
            features_df = df.reindex(columns=self.final_columns, fill_value=0)
            
            # Re-attach targets if they exist (for validation evaluation)
            if targets:
                df = pd.concat([features_df, df[targets]], axis=1)
            else:
                df = features_df

        # Final safety fill
        if df.isna().any().any():
            df = df.fillna(0)

        print(f"[Preprocessor.transform] Complete. Shape: {df.shape}")
        return df

    def fit_transform(self, df: pd.DataFrame, y=None, is_train: bool = True, **fit_params) -> pd.DataFrame:
        """Override to ensure is_train defaults to True during pipeline training."""
        return self.fit(df, y).transform(df, is_train=is_train)

    # ── Internal Methods (unchanged logic) ────────────────────────────────

    def _identify_feature_types(self, df: pd.DataFrame):
        exclude = {TARGET_COLUMN, TARGET_LOG, 'Sale_Date'} | set(DROP_FEATURES)
        self.numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        self.categorical_features = [c for c in df.select_dtypes(include=['object', 'string']).columns if c not in exclude]

    def _apply_imputation(self, df: pd.DataFrame):
        for col, medians in self.group_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(df['Product Group'].map(medians))
        
        for col in self.numeric_features:
            if col in df.columns and col in self.global_medians:
                df[col] = df[col].fillna(self.global_medians[col])

        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')

    def _apply_column_drops(self, df: pd.DataFrame):
        drop_list = set(DROP_FEATURES)
        drop_list.update(df.select_dtypes(include=['datetime64']).columns)
        protected = set(TARGET_ENCODE_FEATURES + ONEHOT_FEATURES)
        objects = df.select_dtypes(include=['object', 'string']).columns
        drop_list.update([c for c in objects if c not in protected and c not in {TARGET_COLUMN, TARGET_LOG}])
        valid_drops = [c for c in drop_list if c in df.columns]
        if valid_drops:
            df.drop(columns=valid_drops, inplace=True)

    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, mapping in self.target_encode_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(self.global_target_mean)

        ohe_cols = [c for c in ONEHOT_FEATURES if c in df.columns]
        if ohe_cols and self.onehot_columns:
            df_ohe = df[ohe_cols].copy()
            for col in ohe_cols:
                rare = self.rare_categories_map.get(col, set())
                known = self.known_categories.get(col, set())
                if rare or known:
                    mask_rare = df_ohe[col].isin(rare)
                    mask_unknown = ~df_ohe[col].isin(known)
                    df_ohe.loc[mask_rare | mask_unknown, col] = 'Other'
            
            df_ohe = df_ohe.fillna('Unknown')
            dummies = pd.get_dummies(df_ohe, columns=ohe_cols, drop_first=True, dtype=np.int8)
            dummies = dummies.reindex(columns=self.onehot_columns, fill_value=0)
            df = df.drop(columns=ohe_cols)
            df = pd.concat([df, dummies], axis=1)

        return sanitize_column_names(df)
    
# ============================================================================
# PIPELINE ENTRY POINT
# ============================================================================

def prepare_data(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> Dict[str, any]:
    """Execute full preparation pipeline: Engineer -> Split -> Preprocess."""
    print("\n" + "=" * 70)
    print(" DATA PREPARATION PIPELINE")
    print("=" * 70)

    # 1. Feature Engineering
    df = engineer_features(df)

    # 2. Splits (70% Train / 15% Val / 15% Test)
    train_df, temp_df = train_test_split(df, train_size=TRAIN_SIZE, random_state=random_state, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, shuffle=True)
    
    print(f"Split sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # 3. Preprocessing
    preprocessor = Preprocessor()
    
    # Fit & Transform Train
    train_processed = preprocessor.fit_transform(train_df, is_train=True)
    
    # Transform Val/Test (is_train=False ensures no target leakage/transformation)
    val_processed = preprocessor.transform(val_df, is_train=False)
    test_processed = preprocessor.transform(test_df, is_train=False)

    # 4. Feature/Target Separation
    def split_xy(data_df):
        X = data_df.drop(columns=[TARGET_COLUMN, TARGET_LOG], errors='ignore')
        y_log = data_df[TARGET_LOG] if TARGET_LOG in data_df.columns else np.log1p(data_df[TARGET_COLUMN])
        return X, y_log

    X_train, y_train = split_xy(train_processed)
    X_val, y_val = split_xy(val_processed)
    X_test, y_test = split_xy(test_processed)

    print("\n[prepare_data] Data preparation complete.")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'y_train_orig': train_df[TARGET_COLUMN],
        'y_val_orig': val_df[TARGET_COLUMN],
        'y_test_orig': test_df[TARGET_COLUMN],
        'preprocessor': preprocessor
    }