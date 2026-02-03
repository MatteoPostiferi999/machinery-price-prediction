"""
evaluation.py
=============
Single source of truth for model evaluation metrics and reporting.

This module consolidates all metric calculations to ensure consistency across
validation and test phases. It handles:
  1. Inverse-transformation of log-predictions (Log -> Dollars)
  2. Calculation of regression metrics (MAE, RMSE, R², MAPE)
  3. Comparative reporting across multiple models
  4. Deep-dive error analysis (residuals by price bucket, worst outliers)

Design Pattern:
    The 'ModelEvaluator' class acts as a facade, abstracting away the raw
    math and string formatting required for professional reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import REPORTS_DIR


# ============================================================================
# CORE METRIC FUNCTIONS (Vectorized)
# ============================================================================

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Robust calculation that automatically handles division-by-zero
    by excluding rows where y_true is 0.

    Args:
        y_true: True values (original scale).
        y_pred: Predicted values (original scale).

    Returns:
        MAPE as a percentage (e.g., 12.5 for 12.5%).
    """
    # Vectorized boolean mask for safe division
    mask = y_true != 0
    
    if np.sum(mask) == 0:
        return np.nan  # Edge case: all targets are zero
        
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard regression metrics in a single pass.

    Args:
        y_true: True values (original dollar scale).
        y_pred: Predicted values (original dollar scale).

    Returns:
        Dictionary with keys: MAE, RMSE, R2, MAPE.
    """
    # Ensure numpy arrays for vectorized math
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    return {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2':   r2_score(y_true, y_pred),
        'MAPE': _mape(y_true, y_pred),
    }

# Alias for backward compatibility if needed
evaluate_model = compute_metrics


# ============================================================================
# MODEL EVALUATOR CLASS
# ============================================================================

class ModelEvaluator:
    """
    Orchestrator for model assessment and reporting.
    
    Encapsulates the logic for transforming predictions back to the original
    scale (expm1) and generating human-readable performance reports.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Args:
            feature_names: List of column names (useful for future feature-specific error analysis).
        """
        self.feature_names = feature_names or []

    # ── Single Model Evaluation ───────────────────────────────────────────

    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y_true_orig: pd.Series,
        label: str = "Model"
    ) -> Dict[str, float]:
        """
        Generate metrics for a single model.
        
        Handles the Log-transform inversion automatically.
        """
        # 1. Predict (Log Scale)
        y_pred_log = model.predict(X)
        
        # 2. Inverse Transform (Log -> Real Dollars)
        y_pred = np.expm1(y_pred_log)

        # 3. Calculate Metrics
        metrics = compute_metrics(y_true_orig.values, y_pred)

        self._print_metrics(metrics, title=f"{label} — Evaluation Results")
        return metrics

    # ── Multi-Model Comparison ────────────────────────────────────────────

    def compare_all(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y_true_orig: pd.Series
    ) -> pd.DataFrame:
        """
        Run evaluation on multiple models and return a leaderboard.

        Args:
            models: Dictionary of {name: trained_model_object}.
            X: Test features.
            y_true_orig: True target values (original scale).

        Returns:
            DataFrame sorted by R² score (descending).
        """
        rows = []
        for name, model in models.items():
            # Predict & Invert
            y_pred = np.expm1(model.predict(X))
            
            # Metric Calculation
            metrics = compute_metrics(y_true_orig.values, y_pred)
            metrics['Model'] = name
            rows.append(metrics)

        # Create Leaderboard
        df = pd.DataFrame(rows)
        cols = ['Model', 'MAE', 'RMSE', 'R2', 'MAPE']
        df = df[cols].sort_values('R2', ascending=False).reset_index(drop=True)

        # Console Report
        print("\n" + "=" * 65)
        print(" MODEL COMPARISON LEADERBOARD")
        print("=" * 65)
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print("=" * 65 + "\n")

        return df

    # ── Deep Dive: Residual Analysis ──────────────────────────────────────

    def residual_summary(
        self,
        model: Any,
        X: pd.DataFrame,
        y_true_orig: pd.Series
    ) -> pd.DataFrame:
        """
        Analyze errors across different price ranges (quintiles).
        
        This reveals if the model is biased (e.g., underpredicting expensive machines).
        """
        y_true = y_true_orig.values.astype(float)
        y_pred = np.expm1(model.predict(X))

        # Create 5 buckets (Quintiles) based on True Price
        labels  = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        buckets = pd.qcut(y_true, q=5, labels=labels, duplicates='drop')

        rows = []
        for label in labels:
            mask = (buckets == label)
            if mask.sum() == 0:
                continue
            
            # Calculate metrics for this specific bucket
            m = compute_metrics(y_true[mask], y_pred[mask])
            m['Price Range'] = label
            m['Count'] = int(mask.sum())
            rows.append(m)

        df = pd.DataFrame(rows)[['Price Range', 'Count', 'MAE', 'RMSE', 'R2', 'MAPE']]

        print("\n" + "=" * 65)
        print(" RESIDUAL ANALYSIS — BY PRICE SEGMENT")
        print("=" * 65)
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print("=" * 65 + "\n")

        return df

    # ── Deep Dive: Outlier Detection ──────────────────────────────────────

    def worst_predictions(
        self,
        model: Any,
        X: pd.DataFrame,
        y_true_orig: pd.Series,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Identify the N most erroneous predictions.
        Useful for debugging specific edge cases.
        """
        y_true = y_true_orig.values.astype(float)
        y_pred = np.expm1(model.predict(X))
        
        abs_errors = np.abs(y_true - y_pred)

        # Optimized Top-N selection (Argpartition is O(n), faster than Sort O(n log n))
        top_idx = np.argpartition(abs_errors, -n)[-n:]
        # Sort just the top N for display
        top_idx = top_idx[np.argsort(abs_errors[top_idx])[::-1]]

        df = pd.DataFrame({
            'True Price ($)':      y_true[top_idx],
            'Predicted Price ($)': y_pred[top_idx],
            'Error ($)':           abs_errors[top_idx],
            'Error (%)':           (abs_errors[top_idx] / y_true[top_idx]) * 100,
        }).round(2)

        print("\n" + "=" * 65)
        print(f" TOP {n} WORST PREDICTIONS (OUTLIERS)")
        print("=" * 65)
        print(df.to_string(index=False))
        print("=" * 65 + "\n")

        return df

    # ── Persistence ───────────────────────────────────────────────────────

    def save_results(
        self,
        comparison_df: pd.DataFrame,
        filename: str = "evaluation_results.csv"
    ) -> None:
        """Export evaluation results to CSV."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / filename
        comparison_df.to_csv(path, index=False)
        print(f"[Evaluation] Results saved to {path}")

    # ── Internal Helper ───────────────────────────────────────────────────

    @staticmethod
    def _print_metrics(metrics: Dict[str, float], title: str = "Evaluation") -> None:
        """Console printer for metric dictionaries."""
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")
        print(f"  MAE   : ${metrics['MAE']:>12,.2f}")
        print(f"  RMSE  : ${metrics['RMSE']:>12,.2f}")
        print(f"  R²    : {metrics['R2']:>13.4f}")
        print(f"  MAPE  : {metrics['MAPE']:>12.2f}%")
        print(f"{'=' * 50}\n")