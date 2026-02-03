"""
preprocessing.py - Sklearn-compatible preprocessing pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set
from sklearn.base import BaseEstimator, TransformerMixin

from .config import (
    TARGET_COLUMN, TARGET_LOG, DROP_FEATURES, GROUP_IMPUTE_FEATURES,
    PRICE_LOWER_BOUND, PRICE_UPPER_BOUND, TARGET_ENCODE_FEATURES,
    ONEHOT_FEATURES, RARE_LABEL_THRESHOLD
)


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace special chars in column names for LightGBM compatibility."""
    pattern = r'[\[\]<>"\'{},.:]'
    new_cols = df.columns.astype(str).str.replace(pattern, '_', regex=True)
    
    # Handle duplicates
    seen = {}
    final = []
    for col in new_cols:
        if col in seen:
            seen[col] += 1
            final.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final.append(col)
    
    df.columns = final
    return df


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible preprocessor with no data leakage.
    
    Pipeline:
        1. Group-aware imputation (median by Product Group)
        2. Global imputation (fallback medians)
        3. Target encoding (high-cardinality categoricals)
        4. One-hot encoding (low-cardinality categoricals)
        5. Column alignment (ensures train/test match)
    
    Usage:
        >>> prep = Preprocessor()
        >>> train_prep = prep.fit_transform(train_df, is_train=True)
        >>> val_prep = prep.transform(val_df, is_train=False)
    """
    
    def __init__(self):
        # Imputation state
        self.group_medians: Dict[str, pd.Series] = {}
        self.global_medians: Dict[str, float] = {}
        
        # Feature type lists
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        
        # Encoding state
        self.target_encode_maps: Dict[str, pd.Series] = {}
        self.global_target_mean: float = 0.0
        self.known_categories: Dict[str, Set] = {}
        self.rare_categories_map: Dict[str, Set] = {}
        self.onehot_columns: List[str] = []
        
        # Column alignment
        self.final_columns: List[str] = None
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame, y=None) -> 'Preprocessor':
        """Fit transformers on training data only."""
        print("\n[Preprocessor] Fitting...")
        df = df.copy()
        
        self._identify_features(df)
        self._fit_imputation(df)
        self._fit_encoding(df)
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Transform data using fitted state."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")
        
        print(f"[Preprocessor] Transforming (train={is_train})...")
        df = df.copy()
        
        # Target transformation (train only)
        if is_train and TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(PRICE_LOWER_BOUND, PRICE_UPPER_BOUND)
            df[TARGET_LOG] = np.log1p(df[TARGET_COLUMN])
        
        # Apply transformations
        self._apply_imputation(df)
        self._apply_column_drops(df)
        df = self._apply_encoding(df)
        df = self._align_columns(df, is_train)
        
        # Final safety fillna
        df = df.fillna(0)
        
        print(f"[Preprocessor] Output shape: {df.shape}")
        return df
    
    def fit_transform(self, df: pd.DataFrame, y=None, is_train: bool = True, **kwargs) -> pd.DataFrame:
        """Fit and transform in one call."""
        return self.fit(df, y).transform(df, is_train=is_train)
    
    # ── Internal Methods ────────────────────────────────────────────────
    
    def _identify_features(self, df: pd.DataFrame):
        """Classify columns into numeric/categorical."""
        exclude = {TARGET_COLUMN, TARGET_LOG, 'Sale_Date'} | set(DROP_FEATURES)
        self.numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns 
                                if c not in exclude]
        self.categorical_features = [c for c in df.select_dtypes(include=['object', 'string']).columns 
                                    if c not in exclude]
    
    def _fit_imputation(self, df: pd.DataFrame):
        """Learn group medians and global medians."""
        # Group-aware medians
        cols = [f for f in GROUP_IMPUTE_FEATURES if f in df.columns]
        if cols and 'Product Group' in df.columns:
            medians_df = df.groupby('Product Group')[cols].median()
            for col in cols:
                self.group_medians[col] = medians_df[col]
        
        # Global fallback
        if self.numeric_features:
            self.global_medians = df[self.numeric_features].median().to_dict()
    
    def _fit_encoding(self, df: pd.DataFrame):
        """Learn target encoding maps and rare categories."""
        # Target encoding
        if TARGET_COLUMN in df.columns:
            self.global_target_mean = df[TARGET_COLUMN].mean()
            te_feats = [f for f in TARGET_ENCODE_FEATURES if f in df.columns]
            for feat in te_feats:
                self.target_encode_maps[feat] = df.groupby(feat)[TARGET_COLUMN].mean()
        
        # Rare label detection for OHE
        ohe_feats = [f for f in ONEHOT_FEATURES if f in df.columns]
        if ohe_feats:
            df_sim = df[ohe_feats].copy()
            for feat in ohe_feats:
                self.known_categories[feat] = set(df[feat].dropna().unique())
                counts = df[feat].value_counts()
                rare = set(counts[counts < RARE_LABEL_THRESHOLD].index)
                self.rare_categories_map[feat] = rare
                
                # Simulate rare→Other
                if rare:
                    df_sim.loc[df_sim[feat].isin(rare), feat] = 'Other'
            
            df_sim = df_sim.fillna('Unknown')
            dummies = pd.get_dummies(df_sim, columns=ohe_feats, drop_first=True)
            self.onehot_columns = dummies.columns.tolist()
    
    def _apply_imputation(self, df: pd.DataFrame):
        """Impute using fitted medians."""
        # Group-aware
        for col, medians in self.group_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(df['Product Group'].map(medians))
        
        # Global
        for col in self.numeric_features:
            if col in df.columns and col in self.global_medians:
                df[col] = df[col].fillna(self.global_medians[col])
        
        # Categorical
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
    
    def _apply_column_drops(self, df: pd.DataFrame):
        """Drop specified columns and datetime types."""
        drop_list = set(DROP_FEATURES)
        drop_list.update(df.select_dtypes(include=['datetime64']).columns)
        
        # Drop unencoded categoricals (except those in encoding lists)
        protected = set(TARGET_ENCODE_FEATURES + ONEHOT_FEATURES)
        objects = df.select_dtypes(include=['object', 'string']).columns
        drop_list.update([c for c in objects if c not in protected 
                         and c not in {TARGET_COLUMN, TARGET_LOG}])
        
        valid = [c for c in drop_list if c in df.columns]
        if valid:
            df.drop(columns=valid, inplace=True)
    
    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target encoding and one-hot encoding."""
        # Target encoding
        for col, mapping in self.target_encode_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(self.global_target_mean)
        
        # One-hot encoding
        ohe_cols = [c for c in ONEHOT_FEATURES if c in df.columns]
        if ohe_cols and self.onehot_columns:
            df_ohe = df[ohe_cols].copy()
            
            # Handle rare/unknown categories
            for col in ohe_cols:
                rare = self.rare_categories_map.get(col, set())
                known = self.known_categories.get(col, set())
                
                mask_rare = df_ohe[col].isin(rare)
                mask_unknown = ~df_ohe[col].isin(known)
                df_ohe.loc[mask_rare | mask_unknown, col] = 'Other'
            
            df_ohe = df_ohe.fillna('Unknown')
            dummies = pd.get_dummies(df_ohe, columns=ohe_cols, drop_first=True, dtype=np.int8)
            dummies = dummies.reindex(columns=self.onehot_columns, fill_value=0)
            
            df = df.drop(columns=ohe_cols)
            df = pd.concat([df, dummies], axis=1)
        
        return sanitize_column_names(df)
    
    def _align_columns(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Ensure exact column match between train and test."""
        numerics = df.select_dtypes(include=[np.number]).columns
        targets = [c for c in [TARGET_COLUMN, TARGET_LOG] if c in df.columns]
        
        if is_train:
            # Define feature columns (exclude targets)
            self.final_columns = [c for c in numerics if c not in [TARGET_COLUMN, TARGET_LOG]]
            return df[self.final_columns + targets]
        else:
            # Force exact column match
            if self.final_columns is None:
                raise RuntimeError("Transform on test before train")
            
            features_df = df.reindex(columns=self.final_columns, fill_value=0)
            
            if targets:
                return pd.concat([features_df, df[targets]], axis=1)
            return features_df


def build_preprocessor() -> Preprocessor:
    """Factory function to create preprocessor instance."""
    return Preprocessor()