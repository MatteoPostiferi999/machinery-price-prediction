"""
preprocessing.py
================
Handles data loading, feature engineering, and preprocessing pipeline.
All transformations are modular and follow scikit-learn conventions.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split

# Import configuration
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
    Sanitize column names to be compatible with LightGBM and other libraries.
    
    LightGBM does not support special JSON characters: [ ] < > : " ,
    This function replaces them with safe alternatives.
    
    Args:
        df: DataFrame with potentially problematic column names
        
    Returns:
        DataFrame with sanitized column names
    """
    # Map of problematic characters to replacements
    replacements = {
        '[': '(',
        ']': ')',
        '<': 'lt',
        '>': 'gt',
        ':': '_',
        '"': '',
        ',': '_',
        '{': '(',
        '}': ')', 
        '.': '_'
    }
    
    # Create new column names
    new_columns = []
    for col in df.columns:
        new_col = col
        for char, replacement in replacements.items():
            new_col = new_col.replace(char, replacement)
        new_columns.append(new_col)
    

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
    
    print(f"[load_data] Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_numeric_from_text(series: pd.Series, pattern: str) -> pd.Series:
    """
    Extract numeric values from text using regex pattern.
    
    Args:
        series: Text column to extract from
        pattern: Regex pattern with one capture group
        
    Returns:
        Series with extracted numeric values (NaN where not found)
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
    Implements insights from EDA:
    - Extract numeric features from text
    - Create temporal features
    - Create derived features
    - Handle data quality issues
    
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
    print(f"[engineer_features] Set {before:,} unrealistic years to NaN")
    
    # ── 2. Text Mining: Extract technical specs ──────────────────
    print("[engineer_features] Extracting numeric features from Product Class Description...")
    
    for feature_name, pattern in EXTRACT_PATTERNS.items():
        df[feature_name] = extract_numeric_from_text(
            df['Product Class Description'], 
            pattern
        )
        non_null = df[feature_name].notna().sum()
        pct = (non_null / len(df)) * 100
        print(f"[engineer_features]   → {feature_name}: {non_null:,} values ({pct:.1f}%)")
    
    # ── 3. Temporal Features ──────────────────────────────────────
    print("[engineer_features] Creating temporal features...")
    df['Sale_Date'] = pd.to_datetime(df['Sales date'], errors='coerce')
    df['Sale_Year'] = df['Sale_Date'].dt.year
    df['Sale_Month'] = df['Sale_Date'].dt.month
    df['Sale_Quarter'] = df['Sale_Date'].dt.quarter
    #df['Is_Steel_TEX'] = ((df['Product Group'] == 'TEX') & (df['Tupper Type'] == 'Steel')).astype(int)
    #df['Is_Rubber_TEX'] = ((df['Product Group'] == 'TEX') & (df['Tupper Type'] == 'Rubber')).astype(int)
    
    # ── 4. Derived Features ───────────────────────────────────────
    print("[engineer_features] Creating derived features...")
    
    # Machine Age (critical feature!)
    df['Machine_Age'] = df['Sale_Year'] - df['Year Made']
    
    # Has_Hours indicator (12.6% price bias!)
    df['Has_Hours'] = df['MachineHours CurrentMeter'].notna().astype(int)
    has_hours_pct = (df['Has_Hours'] == 1).sum() / len(df) * 100
    print(f"[engineer_features]   → Has_Hours: {has_hours_pct:.1f}% have hour meter data")
    
    print("[engineer_features] Feature engineering complete!")
    print(f"[engineer_features] Created features: {list(EXTRACT_PATTERNS.keys()) + ['Machine_Age', 'Has_Hours', 'Sale_Year', 'Sale_Month']}")
    
    return df


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

class Preprocessor:
    """
    Handles all preprocessing transformations with proper train/test separation.
    Follows scikit-learn fit/transform pattern to prevent data leakage.
    """
    
    def __init__(self):
        """Initialize preprocessor with empty state."""
        self.group_medians: Dict[str, pd.Series] = {}
        self.numeric_features: list = []
        self.categorical_features: list = []
        self._is_fitted = False
        self.known_categories: Dict[str, set] = {}   # col -> set of categories seen in training


        # Target encoding state
        self.target_encode_maps: Dict[str, pd.Series] = {}
        self.global_target_mean: float = 0.0

        # One-hot encoding state (rare label + dummies)
        self.rare_categories_map: Dict[str, set] = {}  # col -> set of rare categories
        self.onehot_columns: list = []  # columns after get_dummies (for alignment)
    
    def fit(self, df: pd.DataFrame) -> 'Preprocessor':
        """
        Fit preprocessing transformers on training data.
        
        Args:
            df: Training DataFrame (with engineered features)
            
        Returns:
            self (for method chaining)
        """
        print("\n[Preprocessor.fit] Fitting preprocessing transformers...")
        
        df = df.copy()

        print(f"[DEBUG fit] TARGET_COLUMN ('{TARGET_COLUMN}') in df.columns? {TARGET_COLUMN in df.columns}")
        print(f"[DEBUG fit] df.columns = {list(df.columns)}")
        
        # ── Identify feature types ────────────────────────────────
        self._identify_feature_types(df)
        
        # ── Calculate group-aware statistics ──────────────────────
        print("[Preprocessor.fit] Calculating group-aware medians...")
        for feature in GROUP_IMPUTE_FEATURES:
            if feature in df.columns and 'Product Group' in df.columns:
                self.group_medians[feature] = df.groupby('Product Group')[feature].median()
                print(f"[Preprocessor.fit]   → {feature}: medians per Product Group calculated")

        # ── Fit target encoding maps ──────────────────────────────
        if TARGET_COLUMN in df.columns:
            self.global_target_mean = df[TARGET_COLUMN].mean()
            print(f"[Preprocessor.fit] Global target mean: ${self.global_target_mean:,.0f}")

            te_features_present = [f for f in TARGET_ENCODE_FEATURES if f in df.columns]
            if te_features_present:
                print(f"[Preprocessor.fit] Calculating target encoding maps for {len(te_features_present)} features...")
                for feature in te_features_present:
                    self.target_encode_maps[feature] = df.groupby(feature)[TARGET_COLUMN].mean()
                    print(f"[Preprocessor.fit]   → {feature}: {len(self.target_encode_maps[feature])} categories mapped")

        # ── Fit rare categories map for one-hot encoding ──────────
        ohe_features_present = [f for f in ONEHOT_FEATURES if f in df.columns]
        if ohe_features_present:
            for feature in ohe_features_present:
                self.known_categories[feature] = set(df[feature].dropna().unique())  

            print(f"[Preprocessor.fit] Calculating rare categories for {len(ohe_features_present)} OHE features (threshold={RARE_LABEL_THRESHOLD})...")
            for feature in ohe_features_present:
                value_counts = df[feature].value_counts()
                rare_cats = set(value_counts[value_counts < RARE_LABEL_THRESHOLD].index)
                self.rare_categories_map[feature] = rare_cats
                if rare_cats:
                    print(f"[Preprocessor.fit]   → {feature}: {len(rare_cats)} rare categories will become 'Other'")

            # Apply rare label logic and compute dummy columns on train
            df_ohe = df[ohe_features_present].copy()
            for feature in ohe_features_present:
                rare_cats = self.rare_categories_map.get(feature, set())
                if rare_cats:
                    df_ohe[feature] = df_ohe[feature].apply(lambda x: 'Other' if x in rare_cats else x)
            df_ohe = df_ohe.fillna('Unknown')
            dummies = pd.get_dummies(df_ohe, columns=ohe_features_present, drop_first=True)
            self.onehot_columns = dummies.columns.tolist()
            print(f"[Preprocessor.fit]   → Will create {len(self.onehot_columns)} one-hot encoded features")
        
        self._is_fitted = True
        print("[Preprocessor.fit] Preprocessing transformers fitted successfully!")
        
        return self
    
    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Apply preprocessing transformations.
        
        Args:
            df: DataFrame to transform (must have engineered features)
            is_train: If True, also applies target transformation
            
        Returns:
            Preprocessed DataFrame ready for modeling
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform! Call .fit() first.")
        
        print(f"\n[Preprocessor.transform] Transforming data (is_train={is_train})...")
        df = df.copy()
        
        # ── 1. Target transformation (train only) ─────────────────
        if is_train and TARGET_COLUMN in df.columns:
            print(f"[Preprocessor.transform] Capping {TARGET_COLUMN} outliers...")
            before = df[TARGET_COLUMN].describe()
            df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(PRICE_LOWER_BOUND, PRICE_UPPER_BOUND)
            after = df[TARGET_COLUMN].describe()
            print(f"[Preprocessor.transform]   → Before: min=${before['min']:,.0f}, max=${before['max']:,.0f}")
            print(f"[Preprocessor.transform]   → After:  min=${after['min']:,.0f}, max=${after['max']:,.0f}")
            
            print(f"[Preprocessor.transform] Applying log transformation to {TARGET_COLUMN}...")
            df[TARGET_LOG] = np.log1p(df[TARGET_COLUMN])
            skew_before = df[TARGET_COLUMN].skew()
            skew_after = df[TARGET_LOG].skew()
            print(f"[Preprocessor.transform]   → Skewness before: {skew_before:.2f}")
            print(f"[Preprocessor.transform]   → Skewness after:  {skew_after:.2f}")
        
        # ── 2. Imputation ─────────────────────────────────────────
        df = self._impute_missing(df)

        # ── 3. Drop unwanted columns (BEFORE sanitize_column_names mutates names)
        df = self._drop_columns(df)

        # ── 4. Encoding ───────────────────────────────────────────
        df = self._encode_categorical(df)

        # ── 5. FINAL VALIDATION ───────────────────────────────────
        # Ensure all columns are numeric (critical for sklearn)
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"\n[Preprocessor] WARNING: Found {len(non_numeric)} non-numeric columns:")
            for col in non_numeric:
                print(f"  → {col}: {df[col].dtype}")
            
            # Drop non-numeric columns (except target)
            cols_to_drop = [c for c in non_numeric if c not in [TARGET_COLUMN, TARGET_LOG]]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                print(f"[Preprocessor] Dropped {len(cols_to_drop)} non-numeric columns")
        
        # Ensure no NaN
        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            print(f"\n[Preprocessor] WARNING: Found NaN in {len(nan_cols)} columns after preprocessing")
            print(f"[Preprocessor] Filling NaN with 0...")
            df = df.fillna(0)
        
        print(f"[Preprocessor.transform] Transformation complete! Final shape: {df.shape}")
        print(f"[Preprocessor.transform] All numeric: {len(df.select_dtypes(include=[np.number]).columns) == len(df.columns)}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step (convenience method for training data).
        
        Args:
            df: Training DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        return self.fit(df).transform(df, is_train=True)
    
    # ── Internal Methods ──────────────────────────────────────────────────
    
    def _identify_feature_types(self, df: pd.DataFrame) -> None:
        """Identify numeric and categorical features."""
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
        """Apply group-aware and standard imputation."""
        print("[Preprocessor] Applying imputation...")
        
        # Group-aware imputation
        for feature, group_medians in self.group_medians.items():
            if feature in df.columns:
                before = df[feature].isna().sum()
                df[feature] = df[feature].fillna(
                    df['Product Group'].map(group_medians)
                )
                after = df[feature].isna().sum()
                filled = before - after
                print(f"[Preprocessor]   → {feature}: filled {filled:,} values using Product Group medians")
        
        # Standard imputation (remaining nulls)
        for col in self.numeric_features:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"[Preprocessor]   → {col}: filled {missing:,} remaining NaN with median")
        
        for col in self.categorical_features:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    df[col] = df[col].fillna('Unknown')
                    print(f"[Preprocessor]   → {col}: filled {missing:,} with 'Unknown'")
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply categorical encoding in two steps:
        1. Target encoding for high-cardinality features (TARGET_ENCODE_FEATURES)
        2. Rare label grouping + one-hot encoding for low-cardinality features (ONEHOT_FEATURES)
        """
        print("[Preprocessor] Applying categorical encoding...")

    # ── DEBUG: stampa lo stato interno ──
        print(f"[DEBUG] self.target_encode_maps keys : {list(self.target_encode_maps.keys()) if self.target_encode_maps else 'VUOTO/None'}")
        print(f"[DEBUG] self.onehot_columns count    : {len(self.onehot_columns) if self.onehot_columns else 'VUOTO/None'}")
        print(f"[DEBUG] self.rare_categories_map keys: {list(self.rare_categories_map.keys()) if hasattr(self, 'rare_categories_map') and self.rare_categories_map else 'VUOTO/None'}")
        # ── fine debug ──


        # ── PART 1: Target Encoding ──────────────────────────────────
        te_features_present = [f for f in TARGET_ENCODE_FEATURES if f in df.columns]
        if te_features_present and self.target_encode_maps:
            print(f"[Preprocessor]   Target encoding {len(te_features_present)} features...")
            for feature in te_features_present:
                if feature in self.target_encode_maps:
                    encoding_map = self.target_encode_maps[feature]
                    # Map known categories, fill unknown with global mean
                    df[feature] = df[feature].map(encoding_map).fillna(self.global_target_mean)
                    print(f"[Preprocessor]     → {feature}: target encoded")

        # ── PART 2: Rare Label + One-Hot Encoding ────────────────────
        ohe_features_present = [f for f in ONEHOT_FEATURES if f in df.columns]
        if ohe_features_present and self.onehot_columns:
            print(f"[Preprocessor]   One-hot encoding {len(ohe_features_present)} features with rare label grouping...")

            # Apply rare label logic
            df_ohe = df[ohe_features_present].copy()
            for feature in ohe_features_present:
                rare_cats = self.rare_categories_map.get(feature, set())
                known_cats = self.known_categories.get(feature, set())   

                df_ohe[feature] = df_ohe[feature].apply(
                    lambda x: 'Other' if (x in rare_cats or x not in known_cats) else x  # ← modifica
                )
            df_ohe = df_ohe.fillna('Unknown')


            if rare_cats:
                    df_ohe[feature] = df_ohe[feature].apply(lambda x: 'Other' if x in rare_cats else x)
            df_ohe = df_ohe.fillna('Unknown')

            # Apply get_dummies
            dummies = pd.get_dummies(df_ohe, columns=ohe_features_present, drop_first=True, dtype=int)

            # Align columns with training set (add missing, drop extra)
            for col in self.onehot_columns:
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[self.onehot_columns]
            dummies.index = df.index

            # Drop original OHE columns and add dummies
            df = df.drop(columns=ohe_features_present)
            df = pd.concat([df, dummies], axis=1)
            print(f"[Preprocessor]     → Created {len(self.onehot_columns)} one-hot features")

        # ── Sanitize column names for LightGBM ───────────────────────
        df = sanitize_column_names(df)

        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unwanted columns including datetime and object types."""
        cols_to_drop = []
        
        # 1. Drop specified columns from config
        cols_to_drop.extend([col for col in DROP_FEATURES if col in df.columns])
        
        # 2. Drop datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        cols_to_drop.extend(datetime_cols)
        
        # 3. Drop remaining object/string columns (safety net)
        #    MA escludi quelle che devono ancora essere encodate
        protected_cols = set(TARGET_ENCODE_FEATURES + ONEHOT_FEATURES)   # ← aggiunta
        object_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        object_cols = [
            c for c in object_cols
            if c not in [TARGET_COLUMN, TARGET_LOG]
            and c not in protected_cols                                   # ← aggiunta
        ]
        cols_to_drop.extend(object_cols)
        
        # Remove duplicates
        cols_to_drop = list(set(cols_to_drop))
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors='ignore')
            print(f"[Preprocessor] Dropped {len(cols_to_drop)} unwanted columns")
        
        return df

# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def prepare_data(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE
) -> Dict[str, any]:
    """
    Complete data preparation pipeline with proper train/val/test split.
    Implements 70/15/15 split with no data leakage.
    
    Args:
        df: Raw DataFrame
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - X_train, X_val, X_test: Preprocessed features
            - y_train, y_val, y_test: Target values (log-transformed)
            - y_train_orig, y_val_orig, y_test_orig: Original target values
            - preprocessor: Fitted Preprocessor instance
    """
    print("\n" + "="*80)
    print("STARTING DATA PREPARATION PIPELINE")
    print("="*80)
    
    # ── STEP 1: Feature Engineering (on full dataset) ─────────────
    df = engineer_features(df)
    
    # ── STEP 2: Train/Temp Split (70% / 30%) ─────────────────────
    print(f"\n[prepare_data] Splitting data: {TRAIN_SIZE*100:.0f}% train / {(VAL_SIZE+TEST_SIZE)*100:.0f}% temp...")
    
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"[prepare_data]   → Train: {len(train_df):,} rows")
    print(f"[prepare_data]   → Temp:  {len(temp_df):,} rows")
    
    # ── STEP 3: Val/Test Split (15% / 15%) ───────────────────────
    print(f"\n[prepare_data] Splitting temp into val/test: 50% each...")
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # 50% of 30% = 15% of total
        random_state=random_state,
        shuffle=True
    )
    
    print(f"[prepare_data]   → Val:  {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"[prepare_data]   → Test: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # ── STEP 4: Preprocessing (fit on train, transform all) ──────
    preprocessor = Preprocessor()
    
    # Fit on training data ONLY
    train_processed = preprocessor.fit_transform(train_df)
    
    # Transform validation and test using fitted transformers
    val_processed = preprocessor.transform(val_df, is_train=False)
    test_processed = preprocessor.transform(test_df, is_train=False)
    
    # ── STEP 5: Separate features and target ─────────────────────
    print("\n[prepare_data] Separating features and target...")
    
    # Features (X)
    X_train = train_processed.drop(columns=[TARGET_COLUMN, TARGET_LOG], errors='ignore')
    X_val = val_processed.drop(columns=[TARGET_COLUMN, TARGET_LOG], errors='ignore')
    X_test = test_processed.drop(columns=[TARGET_COLUMN, TARGET_LOG], errors='ignore')
    
    # Target - Log transformed (for training)
    y_train = train_processed[TARGET_LOG]
    y_val = val_processed[TARGET_LOG] if TARGET_LOG in val_processed.columns else np.log1p(val_df[TARGET_COLUMN])
    y_test = test_processed[TARGET_LOG] if TARGET_LOG in test_processed.columns else np.log1p(test_df[TARGET_COLUMN])
    
    # Target - Original scale (for evaluation)
    y_train_orig = train_df[TARGET_COLUMN]
    y_val_orig = val_df[TARGET_COLUMN]
    y_test_orig = test_df[TARGET_COLUMN]
    
    # ── STEP 6: Final Validation ──────────────────────────────────
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    print(f"\n📊 FINAL SHAPES:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val:   {X_val.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"\n🎯 TARGET STATISTICS (log-transformed):")
    print(f"   y_train: mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    print(f"   y_val:   mean={y_val.mean():.3f}, std={y_val.std():.3f}")
    print(f"   y_test:  mean={y_test.mean():.3f}, std={y_test.std():.3f}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'y_train_orig': y_train_orig,
        'y_val_orig': y_val_orig,
        'y_test_orig': y_test_orig,
        'preprocessor': preprocessor
    }