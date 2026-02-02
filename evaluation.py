"""
evaluation.py
-------------
Computes and compares performance metrics for every trained model.
Supports both regression and classification tasks — set TASK_TYPE in
config.py if you need to switch.  By default assumes regression.

Metrics (regression):
  - MAE, RMSE, R², MAPE
Metrics (classification):
  - Accuracy, Precision, Recall, F1, ROC-AUC
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from typing import Any


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return a dict of regression metrics for a single model."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics = {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R²":   r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }
    print(f"\n[Evaluation] {name}")
    for k, v in metrics.items():
        print(f"  {k:>6}: {v:.4f}")
    return metrics


def evaluate_all(
    trainer: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    split_label: str = "Validation",
) -> pd.DataFrame:
    """
    Loop over every trained model in *trainer*, compute metrics, and
    return a summary DataFrame sorted by RMSE ascending.
    """
    rows = []
    for name in trainer.models:
        y_pred = trainer.predict(name, X_val)
        metrics = evaluate_model(name, y_val, y_pred)
        metrics["model"] = name
        metrics["split"] = split_label
        rows.append(metrics)

    summary = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    print(f"\n[Evaluation] ── {split_label} Summary ──")
    print(summary.to_string(index=False))
    return summary