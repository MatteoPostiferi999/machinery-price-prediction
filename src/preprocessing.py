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

Encoding strategy:
    - High-cardinality categoricals → Target Encoding (mapped to mean target per category)
    - Low-cardinality categoricals  → Rare-label grouping + One-Hot Encoding
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split

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
    DROP_MISSING_THRESHOLD,
    TARGET_ENCODE_FEATURES,
    ONEHOT_FEATURES,
    RARE_LABEL_THRESHOLD
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize column names to be compatible with LightGBM.

    LightGBM does not support special JSON characters: [ ] < > : " , { } .
    This function replaces them with safe alternatives and resolves
    any resulting duplicate column names by appending a numeric suffix.

    Args:
        df: DataFrame with potentially problematic column names

    Returns:
        DataFrame with sanitized column names
    """
    replacements = {
        '[': '(', ']': ')', '<': 'lt', '>': 'gt',
        ':': '_', '"': '', ',': '_', '{': '(', '}': ')', '.': '_'
    }

    new_columns = []
    for col in df.columns:
        new_col = col
        for char, replacement in replacements.items():
            new_col = new_col.replace(char, replacement)
        new_columns.append(new_col)

    # Resolve duplicates by appending _1, _2, ...
    final_columns = []
    seen = {}
    for col in new_columns:
        if col in seen:
            seen[col] += 1
            final_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_columns.append(col)

    df.columns = final_columns
    return df


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load raw CSV data with proper dtype handling.

    Args:
        filepath: Path to CSV file

    Returns:
        Raw DataFrame
    """
    print(f"[load_data] Loading data from {filepath}...")

    df = pd.read_csv(
        filepath,
        low_memory=False,
        dtype={
            'Model Series': str,
            'Couple System': str,
            'Grouser Tracks': str,
            'Hydraulics Flow': str
        }
    )

    print(f"[load_data] Loaded {len(df):,} rows x {len(df.columns)} columns")
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_numeric_from_text(series: pd.Series, pattern: str) -> pd.Series:
    """
    Extract numeric values from text using a regex pattern.

    Args:
        series: Text column to extract from
        pattern: Regex pattern with one capture group

    Returns:
        Series with extracted numeric values (NaN where pattern not found)
    """
    def _extract(text):
        if pd.isna(text):
            return np.nan
        match = re.search(pattern, str(text))
        return float(match.group(1)) if match else np.nan

    return series.apply(_extract)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Steps:
        1. Fix unrealistic Year Made values (data quality)
        2. Extract technical specs from Product Class Description (text mining)
        3. Create temporal features from Sales date
        4. Create derived features (Machine_Age, Has_Hours)

    Args:
        df: Raw DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    print("\n[engineer_features] Starting feature engineering...")

    # ── 1. Data Quality: Fix unrealistic years ───────────────────
    print(f"[engineer_features] Fixing Year Made < {MIN_YEAR}...")
    df['Is_Year_Placeholder'] = (df['Year Made'] == 1000).astype(int)

    before = (df['Year Made'] < MIN_YEAR).sum()
    df.loc[df['Year Made'] < MIN_YEAR, 'Year Made'] = np.nan
    print(f"[engineer_features]   -> Set {before:,} unrealistic years to NaN")

    # ── 2. Text Mining: Extract technical specs ──────────────────
    print("[engineer_features] Extracting numeric features from Product Class Description...")
    for feature_name, pattern in EXTRACT_PATTERNS.items():
        df[feature_name] = extract_numeric_from_text(
            df['Product Class Description'],
            pattern
        )
        non_null = df[feature_name].notna().sum()
        pct = (non_null / len(df)) * 100
        print(f"[engineer_features]   -> {feature_name}: {non_null:,} values ({pct:.1f}%)")

    # ── 3. Temporal Features ──────────────────────────────────────
    print("[engineer_features] Creating temporal features...")
    df['Sale_Date'] = pd.to_datetime(df['Sales date'], errors='coerce')
    df['Sale_Year'] = df['Sale_Date'].dt.year
    df['Sale_Month'] = df['Sale_Date'].dt.month
    df['Sale_Quarter'] = df['Sale_Date'].dt.quarter

    # ── 4. Derived Features ───────────────────────────────────────
    print("[engineer_features] Creating derived features...")

    # Machine Age: key predictor of depreciation
    df['Machine_Age'] = df['Sale_Year'] - df['Year Made']

    # Has_Hours: binary indicator — machines with hour meter data
    # tend to sell at a ~12.6% price premium (signal from EDA)
    df['Has_Hours'] = df['MachineHours CurrentMeter'].notna().astype(int)
    has_hours_pct = (df['Has_Hours'] == 1).sum() / len(df) * 100
    print(f"[engineer_features]   -> Has_Hours: {has_hours_pct:.1f}% have hour meter data")

    print("[engineer_features] Feature engineering complete!")
    return df


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

class Preprocessor:
    """
    Handles all preprocessing transformations with proper train/test separation.
    Follows scikit-learn fit/transform pattern to prevent data leakage.

    State fitted on training data:
        - group_medians: per-group median values for imputation
        - target_encode_maps: category -> mean target price mappings
        - rare_categories_map: categories below frequency threshold
        - known_categories: set of categories seen during training
        - onehot_columns: expected one-hot columns after get_dummies
    """

    def __init__(self):
        self.group_medians: Dict[str, pd.Series] = {}
        self.numeric_features: list = []
        self.categorical_features: list = []
        self._is_fitted = False

        # Target encoding state
        self.target_encode_maps: Dict[str, pd.Series] = {}
        self.global_target_mean: float = 0.0

        # One-hot encoding state
        self.known_categories: Dict[str, set] = {}
        self.rare_categories_map: Dict[str, set] = {}
        self.onehot_columns: list = []

    def fit(self, df: pd.DataFrame) -> 'Preprocessor':
        """
        Fit preprocessing transformers on training data only.

        Computes and stores:
            - Feature type identification (numeric vs categorical)
            - Group-aware medians for imputation
            - Target encoding maps (category -> mean price)
            - Rare category sets and expected one-hot columns

        Args:
            df: Training DataFrame (with engineered features)

        Returns:
            self (for method chaining)
        """
        print("\n[Preprocessor.fit] Fitting preprocessing transformers...")
        df = df.copy()

        # ── Identify feature types ────────────────────────────────
        self._identify_feature_types(df)

        # ── Group-aware imputation statistics ─────────────────────
        print("[Preprocessor.fit] Calculating group-aware medians...")
        for feature in GROUP_IMPUTE_FEATURES:
            if feature in df.columns and 'Product Group' in df.columns:
                self.group_medians[feature] = df.groupby('Product Group')[feature].median()
                print(f"[Preprocessor.fit]   -> {feature}: medians per Product Group calculated")

        # ── Target encoding maps ──────────────────────────────────
        if TARGET_COLUMN in df.columns:
            self.global_target_mean = df[TARGET_COLUMN].mean()
            print(f"[Preprocessor.fit] Global target mean: ${self.global_target_mean:,.0f}")

            te_features_present = [f for f in TARGET_ENCODE_FEATURES if f in df.columns]
            if te_features_present:
                print(f"[Preprocessor.fit] Calculating target encoding maps for {len(te_features_present)} features...")
                for feature in te_features_present:
                    self.target_encode_maps[feature] = df.groupby(feature)[TARGET_COLUMN].mean()
                    print(f"[Preprocessor.fit]   -> {feature}: {len(self.target_encode_maps[feature])} categories mapped")

        # ── Rare label detection + one-hot column alignment ───────
        ohe_features_present = [f for f in ONEHOT_FEATURES if f in df.columns]
        if ohe_features_present:
            # Store known categories per feature (for unseen category handling at transform time)
            for feature in ohe_features_present:
                self.known_categories[feature] = set(df[feature].dropna().unique())

            print(f"[Preprocessor.fit] Calculating rare categories for {len(ohe_features_present)} OHE features (threshold={RARE_LABEL_THRESHOLD})...")
            for feature in ohe_features_present:
                value_counts = df[feature].value_counts()
                rare_cats = set(value_counts[value_counts < RARE_LABEL_THRESHOLD].index)
                self.rare_categories_map[feature] = rare_cats
                if rare_cats:
                    print(f"[Preprocessor.fit]   -> {feature}: {len(rare_cats)} rare categories will become 'Other'")

            # Simulate get_dummies on training data to capture expected columns
            df_ohe = df[ohe_features_present].copy()
            for feature in ohe_features_present:
                rare_cats = self.rare_categories_map.get(feature, set())
                if rare_cats:
                    df_ohe[feature] = df_ohe[feature].apply(lambda x: 'Other' if x in rare_cats else x)
            df_ohe = df_ohe.fillna('Unknown')
            dummies = pd.get_dummies(df_ohe, columns=ohe_features_present, drop_first=True)
            self.onehot_columns = dummies.columns.tolist()
            print(f"[Preprocessor.fit]   -> Will create {len(self.onehot_columns)} one-hot encoded features")

        self._is_fitted = True
        print("[Preprocessor.fit] Preprocessing transformers fitted successfully!")
        return self

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Apply preprocessing transformations using fitted state.

        Args:
            df: DataFrame to transform (must have engineered features)
            is_train: If True, applies target capping and log transformation

        Returns:
            Preprocessed DataFrame ready for modeling
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call .fit() first.")

        print(f"\n[Preprocessor.transform] Transforming data (is_train={is_train})...")
        df = df.copy()

        # ── 1. Target transformation (training set only) ─────────
        if is_train and TARGET_COLUMN in df.columns:
            print(f"[Preprocessor.transform] Capping {TARGET_COLUMN} outliers...")
            df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(PRICE_LOWER_BOUND, PRICE_UPPER_BOUND)

            print(f"[Preprocessor.transform] Applying log transformation to {TARGET_COLUMN}...")
            df[TARGET_LOG] = np.log1p(df[TARGET_COLUMN])
            print(f"[Preprocessor.transform]   -> Skewness: {df[TARGET_COLUMN].skew():.2f} (raw) -> {df[TARGET_LOG].skew():.2f} (log)")

        # ── 2. Imputation ─────────────────────────────────────────
        df = self._impute_missing(df)

        # ── 3. Drop unwanted columns ──────────────────────────────
        df = self._drop_columns(df)

        # ── 4. Categorical encoding ───────────────────────────────
        df = self._encode_categorical(df)

        # ── 5. Final safety checks ────────────────────────────────
        # Drop any residual non-numeric columns (except target)
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        cols_to_drop = [c for c in non_numeric if c not in [TARGET_COLUMN, TARGET_LOG]]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"[Preprocessor.transform] Dropped {len(cols_to_drop)} residual non-numeric columns")

        # Fill any remaining NaN with 0 (safety net after encoding)
        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            print(f"[Preprocessor.transform] Filling remaining NaN in {len(nan_cols)} columns with 0")
            df = df.fillna(0)

        print(f"[Preprocessor.transform] Transformation complete. Final shape: {df.shape}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step (convenience method for training data)."""
        return self.fit(df).transform(df, is_train=True)

    # ── Internal Methods ──────────────────────────────────────────────────

    def _identify_feature_types(self, df: pd.DataFrame) -> None:
        """Identify numeric and categorical features, excluding target and drop list."""
        exclude = [TARGET_COLUMN, TARGET_LOG, 'Sale_Date'] + DROP_FEATURES

        self.numeric_features = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude
        ]
        self.categorical_features = [
            col for col in df.select_dtypes(include=['object', 'string']).columns
            if col not in exclude
        ]

        print(f"[Preprocessor] Identified {len(self.numeric_features)} numeric features")
        print(f"[Preprocessor] Identified {len(self.categorical_features)} categorical features")

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply two-tier imputation strategy:
            1. Group-aware: numeric features imputed with per-Product-Group medians
            2. Standard:    remaining numeric NaN -> column median, categorical NaN -> 'Unknown'
        """
        print("[Preprocessor] Applying imputation...")

        # Tier 1: Group-aware imputation (fitted medians from training)
        for feature, group_medians in self.group_medians.items():
            if feature in df.columns:
                before = df[feature].isna().sum()
                df[feature] = df[feature].fillna(df['Product Group'].map(group_medians))
                filled = before - df[feature].isna().sum()
                if filled > 0:
                    print(f"[Preprocessor]   -> {feature}: filled {filled:,} values using Product Group medians")

        # Tier 2: Standard imputation for remaining nulls
        for col in self.numeric_features:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    df[col] = df[col].fillna(df[col].median())
                    print(f"[Preprocessor]   -> {col}: filled {missing:,} NaN with median")

        for col in self.categorical_features:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    df[col] = df[col].fillna('Unknown')
                    print(f"[Preprocessor]   -> {col}: filled {missing:,} with 'Unknown'")

        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop unwanted columns in three passes:
            1. Explicit drop list from config
            2. Datetime columns (already extracted to temporal features)
            3. Remaining object columns not scheduled for encoding
        """
        cols_to_drop = []

        # Pass 1: Config drop list
        cols_to_drop.extend([col for col in DROP_FEATURES if col in df.columns])

        # Pass 2: Datetime columns
        cols_to_drop.extend(df.select_dtypes(include=['datetime64']).columns.tolist())

        # Pass 3: Object columns not protected by encoding lists
        protected_cols = set(TARGET_ENCODE_FEATURES + ONEHOT_FEATURES)
        object_cols = [
            c for c in df.select_dtypes(include=['object', 'string']).columns
            if c not in [TARGET_COLUMN, TARGET_LOG]
            and c not in protected_cols
        ]
        cols_to_drop.extend(object_cols)

        cols_to_drop = list(set(cols_to_drop))
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors='ignore')
            print(f"[Preprocessor] Dropped {len(cols_to_drop)} unwanted columns")

        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Two-stage categorical encoding:
            1. Target Encoding — high-cardinality features mapped to mean target price.
               Unknown categories (unseen at fit time) fall back to global mean.
            2. One-Hot Encoding — low-cardinality features with rare-label grouping.
               Categories below RARE_LABEL_THRESHOLD or unseen at fit time are
               grouped into 'Other'. Output columns are aligned to training schema.
        """
        print("[Preprocessor] Applying categorical encoding...")

        # ── Stage 1: Target Encoding ──────────────────────────────
        te_features_present = [f for f in TARGET_ENCODE_FEATURES if f in df.columns]
        if te_features_present and self.target_encode_maps:
            print(f"[Preprocessor]   Target encoding {len(te_features_present)} features...")
            for feature in te_features_present:
                if feature in self.target_encode_maps:
                    df[feature] = df[feature].map(self.target_encode_maps[feature]).fillna(self.global_target_mean)
                    print(f"[Preprocessor]     -> {feature}: target encoded")

        # ── Stage 2: Rare Label Grouping + One-Hot Encoding ───────
        ohe_features_present = [f for f in ONEHOT_FEATURES if f in df.columns]
        if ohe_features_present and self.onehot_columns:
            print(f"[Preprocessor]   One-hot encoding {len(ohe_features_present)} features with rare label grouping...")

            df_ohe = df[ohe_features_present].copy()

            # Group rare and unseen categories into 'Other'
            for feature in ohe_features_present:
                rare_cats = self.rare_categories_map.get(feature, set())
                known_cats = self.known_categories.get(feature, set())
                df_ohe[feature] = df_ohe[feature].apply(
                    lambda x: 'Other' if (x in rare_cats or x not in known_cats) else x
                )
            df_ohe = df_ohe.fillna('Unknown')

            # One-hot encode and align to training column schema
            dummies = pd.get_dummies(df_ohe, columns=ohe_features_present, drop_first=True, dtype=int)

            for col in self.onehot_columns:
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[self.onehot_columns]
            dummies.index = df.index

            # Replace original categorical columns with dummies
            df = df.drop(columns=ohe_features_present)
            df = pd.concat([df, dummies], axis=1)
            print(f"[Preprocessor]     -> Created {len(self.onehot_columns)} one-hot features")

        # ── Sanitize column names for LightGBM compatibility ─────
        df = sanitize_column_names(df)
        return df


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def prepare_data(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> Dict[str, any]:
    """
    Complete data preparation pipeline: engineer -> split -> preprocess.

    Split strategy (no leakage):
        - 70% train  -> fit preprocessor here
        - 15% val    -> model selection
        - 15% test   -> final evaluation only

    Args:
        df: Raw DataFrame (loaded from CSV)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with keys:
            X_train, X_val, X_test          — preprocessed feature matrices
            y_train, y_val, y_test          — log-transformed target
            y_train_orig, y_val_orig, y_test_orig — original-scale target
            preprocessor                    — fitted Preprocessor instance
    """
    print("\n" + "=" * 70)
    print(" DATA PREPARATION PIPELINE")
    print("=" * 70)

    # ── Step 1: Feature Engineering (full dataset, no target info used) ──
    df = engineer_features(df)

    # ── Step 2: Train / Temp split (70 / 30) ─────────────────────────────
    print(f"\n[prepare_data] Splitting data: {TRAIN_SIZE*100:.0f}% train / {(VAL_SIZE+TEST_SIZE)*100:.0f}% temp...")
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_SIZE, random_state=random_state, shuffle=True
    )
    print(f"[prepare_data]   -> Train: {len(train_df):,} rows")
    print(f"[prepare_data]   -> Temp:  {len(temp_df):,} rows")

    # ── Step 3: Val / Test split (50 / 50 of temp = 15 / 15 of total) ───
    print(f"\n[prepare_data] Splitting temp into val / test (50/50)...")
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state, shuffle=True
    )
    print(f"[prepare_data]   -> Val:  {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"[prepare_data]   -> Test: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")

    # ── Step 4: Preprocess (fit on train only, transform all) ───────────
    preprocessor = Preprocessor()
    train_processed = preprocessor.fit_transform(train_df)
    val_processed   = preprocessor.transform(val_df,  is_train=False)
    test_processed  = preprocessor.transform(test_df, is_train=False)

    # ── Step 5: Separate features and target ─────────────────────────────
    print("\n[prepare_data] Separating features and target...")

    X_train = train_processed.drop(columns=[TARGET_COLUMN, TARGET_LOG], errors='ignore')
    X_val   = val_processed.drop(columns=[TARGET_COLUMN, TARGET_LOG], errors='ignore')
    X_test  = test_processed.drop(columns=[TARGET_COLUMN, TARGET_LOG], errors='ignore')

    # Log-transformed target (used for training)
    y_train = train_processed[TARGET_LOG]
    y_val   = val_processed[TARGET_LOG]   if TARGET_LOG in val_processed.columns   else np.log1p(val_df[TARGET_COLUMN])
    y_test  = test_processed[TARGET_LOG]  if TARGET_LOG in test_processed.columns  else np.log1p(test_df[TARGET_COLUMN])

    # Original-scale target (used for evaluation in real dollars)
    y_train_orig = train_df[TARGET_COLUMN]
    y_val_orig   = val_df[TARGET_COLUMN]
    y_test_orig  = test_df[TARGET_COLUMN]

    # ── Step 6: Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"  X_train: {X_train.shape}    X_val: {X_val.shape}    X_test: {X_test.shape}")
    print(f"  y_train mean: {y_train.mean():.3f}  |  y_val mean: {y_val.mean():.3f}  |  y_test mean: {y_test.mean():.3f}")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'y_train_orig': y_train_orig, 'y_val_orig': y_val_orig, 'y_test_orig': y_test_orig,
        'preprocessor': preprocessor
    }