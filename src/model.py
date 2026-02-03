"""
model.py
========
Modular model training system for machinery price prediction.
Implements Ridge Regression and LightGBM with proper evaluation.

Design principles:
- Each model is a separate, reusable function
- All models follow the same interface
- Easy to add new models
- Comprehensive logging and error handling
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib

# Model imports
from sklearn.linear_model import Ridge
import lightgbm as lgb

# Import config
from .config import (
    MODELS,
    MODELS_DIR,
    RANDOM_STATE,
    PRIMARY_METRICS,
    BEST_METRIC
)


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None
) -> Ridge:
    """Train Ridge Regression model."""
    if params is None:
        params = MODELS['ridge']['params']
    
    print(f"\n[RIDGE] Training with alpha={params['alpha']:.2f}...")
    
    print(f"[DEBUG] X_train shape: {X_train.shape}")
    print(f"[DEBUG] X_train dtypes unique: {X_train.dtypes.unique()}")
    print(f"[DEBUG] Checking for non-numeric columns...")
    
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"[ERROR] Found {len(non_numeric)} non-numeric columns:")
        for col in non_numeric[:10]:  # Show first 10
            print(f"  - {col}: {X_train[col].dtype}")
        print(f"\n[ERROR] Sample values:")
        print(X_train[non_numeric].head())
        raise ValueError("Non-numeric columns found in training data!")
    
    # Check for NaN
    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols:
        print(f"[ERROR] Found {len(nan_cols)} columns with NaN:")
        for col in nan_cols[:10]:
            nan_count = X_train[col].isna().sum()
            print(f"  - {col}: {nan_count} NaN values")
        raise ValueError("NaN values found in training data!")
    
    print(f"[DEBUG] All checks passed! Training...")
    # ═══ END DEBUG ═══
    
    model = Ridge(**params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"[RIDGE] ✅ Training complete in {train_time:.2f}s")
    
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None
) -> lgb.LGBMRegressor:
    """
    Train LightGBM model with early stopping on validation set.
    
    Args:
        X_train: Training features
        y_train: Training target (log-transformed)
        X_val: Validation features
        y_val: Validation target (log-transformed)
        params: Model hyperparameters (uses config defaults if None)
        
    Returns:
        Trained LightGBM model
    """
    if params is None:
        params = MODELS['lightgbm']['params']
    
    print(f"\n[LIGHTGBM] Training with {params['n_estimators']} estimators...")
    print(f"[LIGHTGBM] Learning rate: {params['learning_rate']}, Max depth: {params['max_depth']}")
    
    model = lgb.LGBMRegressor(**params)
    
    start_time = time.time()
    
    # Train with early stopping
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)  # Silent training
        ]
    )
    
    train_time = time.time() - start_time
    
    print(f"[LIGHTGBM] ✅ Training complete in {train_time:.2f}s")
    print(f"[LIGHTGBM] Best iteration: {model.best_iteration_}")
    
    return model


# ============================================================================
# MODEL MANAGER CLASS
# ============================================================================

class ModelManager:
    """
    Manages training, evaluation, and comparison of multiple models.
    Provides a clean interface for the entire modeling workflow.
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.models = {}
        self.results = {}
        self.train_times = {}
        
        # Import evaluation functions here to avoid circular imports
        from .evaluation import evaluate_model
        self.evaluate_model = evaluate_model
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        y_val_orig: pd.Series
    ) -> Dict[str, Dict]:
        """
        Train all configured models and evaluate on validation set.
        
        Args:
            X_train: Training features
            y_train: Training target (log-transformed)
            X_val: Validation features
            y_val: Validation target (log-transformed)
            y_val_orig: Validation target (original scale)
            
        Returns:
            Dictionary with results for each model
        """
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)
        print(f"Training set: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
        print(f"Validation set: {X_val.shape[0]:,} samples")
        
        # Train Ridge
        self._train_single_model(
            'ridge',
            X_train, y_train, X_val, y_val, y_val_orig
        )
        
        # Train LightGBM
        self._train_single_model(
            'lightgbm',
            X_train, y_train, X_val, y_val, y_val_orig
        )
        
        return self.results
    
    def _train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        y_val_orig: pd.Series
    ) -> None:
        """
        Train a single model and evaluate it.
        
        Args:
            model_name: Name of model ('ridge' or 'lightgbm')
            X_train: Training features
            y_train: Training target (log-transformed)
            X_val: Validation features
            y_val: Validation target (log-transformed)
            y_val_orig: Validation target (original scale)
        """
        start_time = time.time()
        
        # Train model based on type
        if model_name == 'ridge':
            model = train_ridge(X_train, y_train)
        elif model_name == 'lightgbm':
            model = train_lightgbm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        train_time = time.time() - start_time
        
        # Make predictions
        y_val_pred_log = model.predict(X_val)
        y_val_pred = np.expm1(y_val_pred_log)  # Convert back to original scale
        
        # Evaluate
        metrics = self.evaluate_model(y_val_orig, y_val_pred)
        
        # Store results
        self.models[model_name] = model
        self.results[model_name] = metrics
        self.train_times[model_name] = train_time
        
        # Print results
        self._print_model_results(model_name, metrics, train_time)
    
    def _print_model_results(
        self,
        model_name: str,
        metrics: Dict[str, float],
        train_time: float
    ) -> None:
        """Print model evaluation results."""
        print(f"\n[{model_name.upper()}] Validation Results:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   MAE:  ${metrics['MAE']:>10,.2f}")
        print(f"   RMSE: ${metrics['RMSE']:>10,.2f}")
        print(f"   R²:   {metrics['R2']:>11.4f}")
    
    def get_best_model(self) -> Tuple[str, any]:
        """
        Get the best model based on configured metric.
        
        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.results:
            raise ValueError("No models have been trained yet!")
        
        if BEST_METRIC == 'R2':
            # Maximize R²
            best_name = max(
                self.results.keys(),
                key=lambda k: self.results[k]['R2']
            )
        else:
            # Minimize MAE or RMSE
            best_name = min(
                self.results.keys(),
                key=lambda k: self.results[k][BEST_METRIC]
            )
        
        return best_name, self.models[best_name]
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create a comparison table of all models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No models have been trained yet!")
        
        comparison = []
        
        for name, metrics in self.results.items():
            comparison.append({
                'Model': MODELS[name]['name'],
                'MAE ($)': f"${metrics['MAE']:,.0f}",
                'RMSE ($)': f"${metrics['RMSE']:,.0f}",
                'R²': f"{metrics['R2']:.4f}",
                'Train Time (s)': f"{self.train_times[name]:.2f}"
            })
        
        df = pd.DataFrame(comparison)
        
        # Sort by R² (best first)
        df['R²_numeric'] = df['R²'].astype(float)
        df = df.sort_values('R²_numeric', ascending=False)
        df = df.drop('R²_numeric', axis=1)
        
        return df
    
    def save_best_model(self, filename: str = "best_model.pkl") -> Path:
        """
        Save the best model to disk.
        
        Args:
            filename: Name of file to save model
            
        Returns:
            Path to saved model
        """
        best_name, best_model = self.get_best_model()
        
        filepath = MODELS_DIR / filename
        joblib.dump(best_model, filepath)
        
        print(f"\n💾 Best model ({MODELS[best_name]['name']}) saved to: {filepath}")
        
        # Also save metadata
        metadata = {
            'model_name': best_name,
            'model_type': MODELS[best_name]['name'],
            'metrics': self.results[best_name],
            'train_time': self.train_times[best_name],
            'features': list(best_model.feature_names_in_) if hasattr(best_model, 'feature_names_in_') else None
        }
        
        metadata_path = MODELS_DIR / filename.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"💾 Model metadata saved to: {metadata_path}")
        
        return filepath
    
    def save_all_models(self) -> None:
        """Save all trained models to disk."""
        print(f"\n💾 Saving all models to {MODELS_DIR}/")
        
        for name, model in self.models.items():
            filename = f"{name}_model.pkl"
            filepath = MODELS_DIR / filename
            joblib.dump(model, filepath)
            print(f"   ✅ Saved {MODELS[name]['name']}: {filename}")
    
    def get_feature_importance(self, model_name: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance or None if not available
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found!")
        
        model = self.models[model_name]
        
        # LightGBM has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else range(len(importances))
            
            df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(top_n)
            
            return df
        
        # Ridge has coefficients (not really "importance" but useful)
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_)  # Absolute value for ranking
            features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else range(len(coefs))
            
            df = pd.DataFrame({
                'Feature': features,
                'Coefficient (abs)': coefs
            }).sort_values('Coefficient (abs)', ascending=False).head(top_n)
            
            return df
        
        return None


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def train_models(data: Dict) -> ModelManager:
    """
    Convenience function to train all models.
    
    Args:
        data: Dictionary from prepare_data() with train/val/test splits
        
    Returns:
        Trained ModelManager instance
    """
    manager = ModelManager()
    
    manager.train_all_models(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        y_val_orig=data['y_val_orig']
    )
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(manager.compare_models().to_string(index=False))
    
    # Identify best model
    best_name, _ = manager.get_best_model()
    best_metrics = manager.results[best_name]
    print(f"\n🏆 Best Model: {MODELS[best_name]['name']}")
    print(f"   R²: {best_metrics['R2']:.4f}")
    print(f"   MAE: ${best_metrics['MAE']:,.0f}")
    
    # Save best model
    manager.save_best_model()
    
    return manager