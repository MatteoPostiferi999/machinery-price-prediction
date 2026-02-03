"""
model.py - Model training and management with sklearn compatibility
"""

import time
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from .config import MODELS, MODELS_DIR, BEST_METRIC
from .evaluation import evaluate_model


# ============================================================================
# TRAINING IMPLEMENTATIONS
# ============================================================================

def train_ridge(X_train: pd.DataFrame, y_train: pd.Series, params: Dict) -> Ridge:
    """Train Ridge regression (fast baseline)."""
    print(f"\n[RIDGE] Training with alpha={params.get('alpha', 1.0):.2f}...")
    model = Ridge(**params)
    start = time.time()
    model.fit(X_train, y_train)
    print(f"[RIDGE] Complete in {time.time()-start:.2f}s")
    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, params: Dict) -> lgb.LGBMRegressor:
    """Train LightGBM with early stopping."""
    print(f"\n[LIGHTGBM] Training (n_estimators={params.get('n_estimators')})...")
    model = lgb.LGBMRegressor(**params)
    
    start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=True),
            lgb.log_evaluation(period=500)
        ]
    )
    print(f"[LIGHTGBM] Complete in {time.time()-start:.2f}s | Best: {model.best_iteration_}")
    return model


# Dispatcher registry
TRAINERS = {
    'ridge': train_ridge,
    'lightgbm': train_lightgbm,
}


# ============================================================================
# MODEL MANAGER (Sklearn-Compatible)
# ============================================================================

class ModelManager(BaseEstimator):
    """
    Sklearn-compatible orchestrator for multi-model training.
    
    Manages:
        - Training multiple models
        - Performance tracking
        - Model selection
        - Artifact persistence
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.train_times: Dict[str, float] = {}
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        y_val_orig: pd.Series) -> Dict[str, Dict]:
        """Train all models defined in config."""
        print("\n" + "=" * 70)
        print(f" TRAINING PHASE")
        print("=" * 70)
        print(f"  Train:      {X_train.shape[0]:,} × {X_train.shape[1]} features")
        print(f"  Validation: {X_val.shape[0]:,} samples")
        
        for model_key in MODELS.keys():
            self._train_single(model_key, X_train, y_train, X_val, y_val, y_val_orig)
        
        return self.results
    
    def _train_single(self, model_key: str, X_train, y_train, X_val, y_val, y_val_orig):
        """Train one model and evaluate."""
        config = MODELS[model_key]
        params = config['params']
        
        # Dispatch to appropriate trainer
        start = time.time()
        if model_key not in TRAINERS:
            raise NotImplementedError(f"No trainer for '{model_key}'")
        
        trainer = TRAINERS[model_key]
        if model_key == 'ridge':
            model = trainer(X_train, y_train, params)
        else:  # lightgbm (needs validation)
            model = trainer(X_train, y_train, X_val, y_val, params)
        
        duration = time.time() - start
        
        # Evaluate (inverse log-transform)
        y_pred = np.expm1(model.predict(X_val))
        metrics = evaluate_model(y_val_orig, y_pred)
        
        # Store results
        self.models[model_key] = model
        self.results[model_key] = metrics
        self.train_times[model_key] = duration
        
        self._log_metrics(config['name'], metrics, duration)
    
    def _log_metrics(self, name: str, metrics: Dict, duration: float):
        """Print formatted metrics."""
        print(f"\n[{name.upper()}] Performance:")
        print(f"   Time: {duration:.2f}s")
        print(f"   MAE:  ${metrics['MAE']:>10,.2f}")
        print(f"   RMSE: ${metrics['RMSE']:>10,.2f}")
        print(f"   R²:   {metrics['R2']:>11.4f}")
    
    # ── Model Selection ──────────────────────────────────────────────────
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Return (key, model) for best performing model."""
        if not self.results:
            raise RuntimeError("No models trained")
        
        # Maximize R2, minimize errors
        if BEST_METRIC in ['R2', 'Accuracy']:
            best_key = max(self.results, key=lambda k: self.results[k][BEST_METRIC])
        else:
            best_key = min(self.results, key=lambda k: self.results[k][BEST_METRIC])
        
        return best_key, self.models[best_key]
    
    def compare_models(self) -> pd.DataFrame:
        """Return comparison DataFrame sorted by BEST_METRIC."""
        rows = [
            {'Model': MODELS[k]['name'], 'Train Time (s)': self.train_times[k], **self.results[k]}
            for k in self.results
        ]
        return pd.DataFrame(rows).sort_values(BEST_METRIC, ascending=False).reset_index(drop=True)
    
    # ── Persistence ──────────────────────────────────────────────────────
    
    def save_best_model(self, filename: str = "best_model.pkl") -> Path:
        """Save best model + metadata."""
        best_key, best_model = self.get_best_model()
        
        # Save model
        path_model = MODELS_DIR / filename
        joblib.dump(best_model, path_model)
        
        # Save metadata
        metadata = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_key': best_key,
            'model_name': MODELS[best_key]['name'],
            'metrics': self.results[best_key],
            'feature_names': self._get_feature_names(best_model)
        }
        path_meta = MODELS_DIR / filename.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, path_meta)
        
        print(f"\n[Persistence] Best model saved: {path_model.name}")
        return path_model
    
    def _get_feature_names(self, model: Any) -> List[str]:
        """Extract feature names if available."""
        return list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []
    
    # ── Feature Importance ───────────────────────────────────────────────
    
    def get_feature_importance(self, model_key: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Extract feature importance (tree: gain, linear: |coef|)."""
        if model_key not in self.models:
            return None
        
        model = self.models[model_key]
        features = self._get_feature_names(model)
        
        if not features:
            return None
        
        # Extract importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return None
        
        df = pd.DataFrame({'Feature': features, 'Importance': importance})
        return df.sort_values('Importance', ascending=False).head(top_n).reset_index(drop=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

def train_models(data: Dict[str, Any]) -> ModelManager:
    """Train, evaluate, and save models."""
    manager = ModelManager()
    
    # Train all
    manager.train_all_models(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['y_val_orig']
    )
    
    # Report leaderboard
    print("\n" + "=" * 70)
    print(" FINAL LEADERBOARD")
    print("=" * 70)
    print(manager.compare_models().to_string(index=False))
    
    # Save best
    best_key, _ = manager.get_best_model()
    best_name = MODELS[best_key]['name']
    best_r2 = manager.results[best_key]['R2']
    print(f"\n[Conclusion] Winner: {best_name} (R²: {best_r2:.4f})")
    manager.save_best_model()
    
    return manager