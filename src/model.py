"""
model.py
--------
Defines the ModelTrainer class that handles:
  - Instantiation of candidate models
  - Hyperparameter tuning (RandomizedSearchCV)
  - Training the best configuration
  - Persisting / loading models to disk (joblib)
All models share the same interface so evaluation.py can loop over them
without knowing implementation details.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from typing import Any

from src.config import (
    HYPERPARAMS,
    PARAM_GRIDS,
    TUNING_CV_FOLDS,
    TUNING_N_ITER,
    RANDOM_STATE,
    TARGET_COLUMN,
    BASE_DIR,
)

# ─── Registry: add new models here ────────────────────────────────
MODEL_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


class ModelTrainer:
    """
    Train, tune, and persist one or more sklearn-compatible models.

    Usage:
        trainer = ModelTrainer()
        trainer.train_all(X_train, y_train)          # tune + fit
        trainer.save_all()                            # persist to disk
        predictions = trainer.predict("random_forest", X_test)
    """

    def __init__(self):
        self.models: dict[str, Any] = {}        # name -> fitted estimator
        self.best_params: dict[str, dict] = {}  # name -> best params after tuning

    # ── public API ────────────────────────────────────────────────
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Tune and fit every model in MODEL_REGISTRY."""
        for name in MODEL_REGISTRY:
            print(f"\n[ModelTrainer] ── Training: {name} ──")
            self._tune_and_fit(name, X_train, y_train)

    def predict(self, name: str, X: pd.DataFrame) -> np.ndarray:
        """Return predictions for an already-trained model."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not trained. Available: {list(self.models.keys())}")
        return self.models[name].predict(X)

    def save_all(self, directory: str = BASE_DIR) -> None:
        """Serialize all trained models to *directory*."""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(directory, f"{name}.joblib")
            joblib.dump(model, path)
            print(f"[ModelTrainer] Saved {name} → {path}")

    def load(self, name: str, directory: str = BASE_DIR) -> None:
        """Load a single persisted model back into memory."""
        path = os.path.join(directory, f"{name}.joblib")
        self.models[name] = joblib.load(path)
        print(f"[ModelTrainer] Loaded {name} from {path}")

    # ── internals ─────────────────────────────────────────────────
    def _tune_and_fit(self, name: str, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Run RandomizedSearchCV then refit the best estimator on full training data."""
        base_estimator = MODEL_REGISTRY[name](
            random_state=RANDOM_STATE, **HYPERPARAMS.get(name, {})
        )

        if name in PARAM_GRIDS:
            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=PARAM_GRIDS[name],
                n_iter=TUNING_N_ITER,
                cv=TUNING_CV_FOLDS,
                scoring="neg_mean_squared_error",
                refit=True,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1,
            )
            search.fit(X_train, y_train)
            self.best_params[name] = search.best_params_
            self.models[name] = search.best_estimator_
            print(f"[ModelTrainer] Best params for {name}: {search.best_params_}")
        else:
            # no grid defined — just fit with defaults
            base_estimator.fit(X_train, y_train)
            self.models[name] = base_estimator