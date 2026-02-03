"""
model.py
========
Model training and management orchestration.

This module handles the lifecycle of machine learning models:
1. Instantiation (based on config)
2. Training (with specific logic for Linear vs Tree-based models)
3. Prediction (handling log-transform inversion)
4. Persistence (saving artifacts and metadata)

Design Pattern:
    Uses a 'ModelManager' to encapsulate state (trained models, results)
    and separate orchestration logic from specific training implementations.
"""

import time
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from sklearn.linear_model import Ridge

# Import configuration and evaluation
from .config import (
    MODELS,
    MODELS_DIR,
    PRIMARY_METRICS,
    BEST_METRIC
)
from .evaluation import evaluate_model


# ============================================================================
# 1. SPECIFIC TRAINING IMPLEMENTATIONS
# ============================================================================

def _train_ridge_impl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any]
) -> Ridge:
    """
    Implementation of Ridge training.
    Fast, interpretable, and serves as a robust baseline.
    """
    print(f"\n[RIDGE] Training with alpha={params.get('alpha', 1.0):.2f}...")
    
    model = Ridge(**params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start_time
    
    print(f"[RIDGE] Training complete in {duration:.2f}s")
    return model


def _train_lightgbm_impl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any]
) -> lgb.LGBMRegressor:
    """
    Implementation of LightGBM training with Early Stopping.
    
    Why Early Stopping?
    It prevents overfitting by monitoring the validation score. If the error
    doesn't decrease for N rounds, training stops. This is crucial for 
    gradient boosting on medium/large datasets.
    """
    print(f"\n[LIGHTGBM] Training (n_estimators={params.get('n_estimators')}, lr={params.get('learning_rate')})...")

    model = lgb.LGBMRegressor(**params)

    start_time = time.time()
    
    # Note: callbacks are the modern way to handle early stopping in LightGBM 4.x+
    callbacks = [
        lgb.early_stopping(stopping_rounds=150, verbose=True),
        lgb.log_evaluation(period=500)
    ]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=callbacks
    )
    
    duration = time.time() - start_time
    print(f"[LIGHTGBM] Training complete in {duration:.2f}s | Best Iteration: {model.best_iteration_}")
    return model


# ============================================================================
# 2. MODEL MANAGER (ORCHESTRATOR)
# ============================================================================

class ModelManager:
    """
    Central control unit for the modeling pipeline.
    
    Responsibilities:
        - Abstraction: Hides the complexity of training different model types.
        - State Management: Keeps track of trained models and their metrics.
        - Persistence: Handles saving/loading of model artifacts.
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.train_times: Dict[str, float] = {}

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        y_val_orig: pd.Series
    ) -> Dict[str, Dict]:
        """
        Iterates through defined models in config and trains them.
        """
        print("\n" + "=" * 70)
        print(f" TRAINING PHASE")
        print("=" * 70)
        print(f"  Training Set:   {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
        print(f"  Validation Set: {X_val.shape[0]:,} rows")

        # Dynamic iteration over config keys
        for model_key in MODELS.keys():
            self._train_and_evaluate(model_key, X_train, y_train, X_val, y_val, y_val_orig)

        return self.results

    def _train_and_evaluate(
        self,
        model_key: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        y_val_orig: pd.Series
    ) -> None:
        """
        Internal driver to train a specific model key, predict, and log metrics.
        """
        config = MODELS.get(model_key)
        if not config:
            raise ValueError(f"Configuration not found for model: {model_key}")

        params = config['params']
        start_time = time.time()

        # ─── 1. Dispatcher Logic ──────────────────────────────────────────────
        # This keeps the logic extensible. To add XGBoost, add a case here.
        if model_key == 'ridge':
            model = _train_ridge_impl(X_train, y_train, params)
        elif model_key == 'lightgbm':
            model = _train_lightgbm_impl(X_train, y_train, X_val, y_val, params)
        else:
            raise NotImplementedError(f"Training logic for '{model_key}' is not implemented.")
        
        duration = time.time() - start_time

        # ─── 2. Prediction (Log-Inverse Transform) ────────────────────────────
        # CRITICAL: Models are trained on log(price), so we must expm1() the output
        # to compare against real dollar values.
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)

        # ─── 3. Evaluation ────────────────────────────────────────────────────
        metrics = evaluate_model(y_val_orig, y_pred)

        # ─── 4. State Update ──────────────────────────────────────────────────
        self.models[model_key] = model
        self.results[model_key] = metrics
        self.train_times[model_key] = duration

        self._log_metrics(config['name'], metrics, duration)

    def _log_metrics(self, display_name: str, metrics: Dict[str, float], duration: float):
        """Helper to print consistent metric blocks."""
        print(f"\n[{display_name.upper()}] Performance:")
        print(f"   Time: {duration:.2f}s")
        print(f"   MAE:  ${metrics['MAE']:>10,.2f}")
        print(f"   RMSE: ${metrics['RMSE']:>10,.2f}")
        print(f"   R²:   {metrics['R2']:>11.4f}")

    # ─── Selection & Comparison ───────────────────────────────────────────────

    def get_best_model(self) -> Tuple[str, Any]:
        """Returns (model_key, model_object) based on BEST_METRIC optimization."""
        if not self.results:
            raise RuntimeError("Models not trained. Call train_all_models() first.")

        # Determine if we maximize or minimize the metric
        if BEST_METRIC in ['R2', 'Accuracy']:
            best_key = max(self.results, key=lambda k: self.results[k][BEST_METRIC])
        else:
            # For MAE, RMSE, etc., lower is better
            best_key = min(self.results, key=lambda k: self.results[k][BEST_METRIC])

        return best_key, self.models[best_key]

    def compare_models(self) -> pd.DataFrame:
        """Returns a sorted DataFrame comparing all models."""
        rows = []
        for key, metrics in self.results.items():
            row = {
                'Model': MODELS[key]['name'],
                'Train Time (s)': self.train_times[key],
                **metrics  # Unpack metrics (MAE, RMSE, R2)
            }
            rows.append(row)
        
        return pd.DataFrame(rows).sort_values(by=BEST_METRIC, ascending=False).reset_index(drop=True)

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save_best_model(self, filename: str = "best_model.pkl") -> Path:
        """Saves the best model + metadata (metrics, feature names) to disk."""
        best_key, best_model = self.get_best_model()
        
        # 1. Save Model Object
        path_model = MODELS_DIR / filename
        joblib.dump(best_model, path_model)
        
        # 2. Save Metadata (Audit Trail)
        # Allows us to know *how* this model performed without retraining
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
        """Safely extract feature names if supported by the estimator."""
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        return []

    # ─── Interpretability ─────────────────────────────────────────────────────

    def get_feature_importance(self, model_key: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Extracts feature importance in a standardized format.
        
        - Linear Models: Absolute value of coefficients.
        - Tree Models:   Gain or Split importance.
        """
        if model_key not in self.models:
            return None
            
        model = self.models[model_key]
        feature_names = self._get_feature_names(model)
        
        # If model doesn't store feature names, we can't map importance reliably
        if not feature_names:
            return None

        importances = None
        
        # 1. Tree-based importance (LightGBM, XGBoost, RF)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
        # 2. Linear importance (Ridge, Lasso)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)

        if importances is None:
            return None

        # Create DataFrame
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        return df.sort_values('Importance', ascending=False).head(top_n).reset_index(drop=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

def train_models(data: Dict[str, Any]) -> ModelManager:
    """
    Main pipeline function to train, evaluate, and save models.
    """
    manager = ModelManager()

    # Train
    manager.train_all_models(
        data['X_train'], data['y_train'],
        data['X_val'],   data['y_val'],
        data['y_val_orig']
    )

    # Report
    print("\n" + "=" * 70)
    print(" FINAL LEADERBOARD")
    print("=" * 70)
    print(manager.compare_models().to_string(index=False))

    # Persist
    best_key, _ = manager.get_best_model()
    best_name = MODELS[best_key]['name']
    best_r2 = manager.results[best_key]['R2']
    
    print(f"\n[Conclusion] Winner: {best_name} (R²: {best_r2:.4f})")
    manager.save_best_model()

    return manager