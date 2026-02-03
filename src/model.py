"""
model.py
========
Model training and management for machinery price prediction.

Implements two models with a clear performance gap by design:
    - Ridge Regression: linear baseline with L2 regularization.
      Useful for interpretability and as a lower-bound benchmark.
    - LightGBM: gradient-boosted trees with early stopping.
      Primary model — handles non-linearity, missing values natively,
      and scales well on the 114 MB dataset.

All models are trained on log-transformed target and predictions are
converted back to original scale (expm1) before evaluation.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib

from sklearn.linear_model import Ridge
import lightgbm as lgb

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
    """
    Train Ridge Regression model.

    Args:
        X_train: Training features (must be fully numeric, no NaN)
        y_train: Training target (log-transformed)
        params:  Hyperparameters — uses config defaults if None

    Returns:
        Fitted Ridge model
    """
    if params is None:
        params = MODELS['ridge']['params']

    print(f"\n[RIDGE] Training with alpha={params['alpha']:.2f}...")

    model = Ridge(**params)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"[RIDGE] Training complete in {train_time:.2f}s")
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

    Early stopping monitors validation RMSE and halts training when
    no improvement is seen for 150 consecutive rounds, preventing
    overfitting on the large feature space.

    Args:
        X_train: Training features
        y_train: Training target (log-transformed)
        X_val:   Validation features
        y_val:   Validation target (log-transformed)
        params:  Hyperparameters — uses config defaults if None

    Returns:
        Fitted LightGBM model (stopped at best iteration)
    """
    if params is None:
        params = MODELS['lightgbm']['params']

    print(f"\n[LIGHTGBM] Training with {params['n_estimators']} max estimators...")
    print(f"[LIGHTGBM] Learning rate: {params['learning_rate']}, Max depth: {params['max_depth']}")

    model = lgb.LGBMRegressor(**params)

    start_time = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=True),
            lgb.log_evaluation(period=500)
        ]
    )
    train_time = time.time() - start_time

    print(f"[LIGHTGBM] Training complete in {train_time:.2f}s")
    print(f"[LIGHTGBM] Best iteration: {model.best_iteration_}")
    return model


# ============================================================================
# MODEL MANAGER CLASS
# ============================================================================

class ModelManager:
    """
    Orchestrates training, evaluation, comparison, and persistence
    of all models in the pipeline.

    Usage:
        manager = ModelManager()
        manager.train_all_models(X_train, y_train, X_val, y_val, y_val_orig)
        manager.save_best_model()
    """

    def __init__(self):
        self.models: Dict[str, object] = {}
        self.results: Dict[str, Dict] = {}
        self.train_times: Dict[str, float] = {}

        # Lazy import to avoid circular dependency
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
        Train all configured models and evaluate each on the validation set.

        Args:
            X_train:    Training features
            y_train:    Training target (log-transformed)
            X_val:      Validation features
            y_val:      Validation target (log-transformed)
            y_val_orig: Validation target (original scale, for evaluation)

        Returns:
            Dictionary mapping model name -> evaluation metrics
        """
        print("\n" + "=" * 70)
        print(" TRAINING MODELS")
        print("=" * 70)
        print(f"  Train: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
        print(f"  Val:   {X_val.shape[0]:,} samples")

        self._train_single_model('ridge',    X_train, y_train, X_val, y_val, y_val_orig)
        self._train_single_model('lightgbm', X_train, y_train, X_val, y_val, y_val_orig)

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
        Train one model, predict on validation, evaluate, and store results.

        Args:
            model_name: Key in MODELS config ('ridge' or 'lightgbm')
            X_train:    Training features
            y_train:    Training target (log-transformed)
            X_val:      Validation features
            y_val:      Validation target (log-transformed)
            y_val_orig: Validation target (original scale)
        """
        start_time = time.time()

        if model_name == 'ridge':
            model = train_ridge(X_train, y_train)
        elif model_name == 'lightgbm':
            model = train_lightgbm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        train_time = time.time() - start_time

        # Predict and convert from log-scale back to dollars
        y_val_pred = np.expm1(model.predict(X_val))

        # Evaluate against original-scale targets
        metrics = self.evaluate_model(y_val_orig, y_val_pred)

        # Store everything
        self.models[model_name] = model
        self.results[model_name] = metrics
        self.train_times[model_name] = train_time

        self._print_model_results(model_name, metrics, train_time)

    def _print_model_results(self, model_name: str, metrics: Dict[str, float], train_time: float) -> None:
        """Print formatted validation results for one model."""
        print(f"\n[{model_name.upper()}] Validation Results:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   MAE:  ${metrics['MAE']:>10,.2f}")
        print(f"   RMSE: ${metrics['RMSE']:>10,.2f}")
        print(f"   R2:   {metrics['R2']:>11.4f}")

    # ── Model selection & comparison ──────────────────────────────────────

    def get_best_model(self) -> Tuple[str, object]:
        """
        Select the best model based on BEST_METRIC from config.

        Returns:
            Tuple of (model_name, fitted model object)
        """
        if not self.results:
            raise ValueError("No models have been trained yet.")

        if BEST_METRIC == 'R2':
            best_name = max(self.results, key=lambda k: self.results[k]['R2'])
        else:
            best_name = min(self.results, key=lambda k: self.results[k][BEST_METRIC])

        return best_name, self.models[best_name]

    def compare_models(self) -> pd.DataFrame:
        """
        Build a human-readable comparison table of all trained models.

        Returns:
            DataFrame sorted by R2 descending
        """
        if not self.results:
            raise ValueError("No models have been trained yet.")

        rows = []
        for name, metrics in self.results.items():
            rows.append({
                'Model':          MODELS[name]['name'],
                'MAE ($)':        f"${metrics['MAE']:,.0f}",
                'RMSE ($)':       f"${metrics['RMSE']:,.0f}",
                'R2':             f"{metrics['R2']:.4f}",
                'Train Time (s)': f"{self.train_times[name]:.2f}"
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(by='R2', ascending=False).reset_index(drop=True)
        return df

    # ── Persistence ───────────────────────────────────────────────────────

    def save_best_model(self, filename: str = "best_model.pkl") -> Path:
        """
        Save the best model and its metadata to MODELS_DIR.

        Args:
            filename: Filename for the model pickle

        Returns:
            Path to saved model file
        """
        best_name, best_model = self.get_best_model()
        filepath = MODELS_DIR / filename
        joblib.dump(best_model, filepath)
        print(f"\n[ModelManager] Best model ({MODELS[best_name]['name']}) saved to: {filepath}")

        # Metadata: model identity, metrics, feature list
        metadata = {
            'model_name':  best_name,
            'model_type':  MODELS[best_name]['name'],
            'metrics':     self.results[best_name],
            'train_time':  self.train_times[best_name],
            'features':    list(best_model.feature_names_in_) if hasattr(best_model, 'feature_names_in_') else None
        }
        metadata_path = MODELS_DIR / filename.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"[ModelManager] Metadata saved to: {metadata_path}")

        return filepath

    def save_all_models(self) -> None:
        """Save every trained model to MODELS_DIR."""
        print(f"\n[ModelManager] Saving all models to {MODELS_DIR}/")
        for name, model in self.models.items():
            filepath = MODELS_DIR / f"{name}_model.pkl"
            joblib.dump(model, filepath)
            print(f"[ModelManager]   Saved {MODELS[name]['name']}: {filepath.name}")

    # ── Feature importance ────────────────────────────────────────────────

    def get_feature_importance(self, model_name: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Extract feature importance from a trained model.

        For LightGBM: uses native feature_importances_ (gain-based).
        For Ridge:    uses absolute coefficient values as a proxy.

        Args:
            model_name: Key in self.models
            top_n:      Number of top features to return

        Returns:
            DataFrame with 'Feature' and 'Importance' columns, or None
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            # Tree-based models (LightGBM)
            features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else range(len(model.feature_importances_))
            df = pd.DataFrame({
                'Feature':    features,
                'Importance': model.feature_importances_
            })
        elif hasattr(model, 'coef_'):
            # Linear models (Ridge) — absolute coefficients
            features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else range(len(model.coef_))
            df = pd.DataFrame({
                'Feature':    features,
                'Importance': np.abs(model.coef_)
            })
        else:
            return None

        return df.sort_values('Importance', ascending=False).head(top_n).reset_index(drop=True)


# ============================================================================
# PIPELINE ENTRY POINT
# ============================================================================

def train_models(data: Dict) -> ModelManager:
    """
    Train all models, print comparison, save best.

    Args:
        data: Dictionary from prepare_data() containing train/val splits

    Returns:
        Fitted ModelManager instance (access models, results, etc.)
    """
    manager = ModelManager()

    manager.train_all_models(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        y_val_orig=data['y_val_orig']
    )

    # Print comparison table
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON")
    print("=" * 70)
    print(manager.compare_models().to_string(index=False))

    # Announce best model
    best_name, _ = manager.get_best_model()
    best_metrics = manager.results[best_name]
    print(f"\n  Best model: {MODELS[best_name]['name']}")
    print(f"  R2:  {best_metrics['R2']:.4f}")
    print(f"  MAE: ${best_metrics['MAE']:,.0f}")

    # Persist
    manager.save_best_model()

    return manager