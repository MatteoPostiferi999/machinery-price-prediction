"""
evaluation.py
=============
Single source of truth for model evaluation.

Handles the full evaluation flow:
  1. Predict on a dataset (log scale)
  2. Inverse-transform predictions back to original dollar scale
  3. Calculate MAE, RMSE, R², MAPE
  4. Compare multiple models side-by-side
  5. Residual + error analysis utilities

Usage in main.py:
    from src.evaluation import ModelEvaluator

    evaluator = ModelEvaluator(feature_names=data['X_val'].columns.tolist())

    # Single model on validation
    val_results = evaluator.evaluate(best_model, data['X_val'], data['y_val_orig'])

    # All models on test (final report)
    test_comparison = evaluator.compare_all(manager.models, data['X_test'], data['y_test_orig'])
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import REPORTS_DIR


# ============================================================================
# CORE METRIC FUNCTIONS  (vectorized, no loops)
# ============================================================================

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Measures average relative error as a percentage.
    Undefined when y_true contains zeros — those rows are excluded.

    Args:
        y_true: True values (original scale)
        y_pred: Predicted values (original scale)

    Returns:
        MAPE as a percentage (e.g. 12.3 means 12.3%)
    """
    mask = y_true != 0                          # exclude zeros vectorized
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all regression metrics in one pass.

    Keys are uppercase (MAE, RMSE, R2, MAPE) — consistent with
    the rest of the pipeline (config.py, model.py, main.py).

    Args:
        y_true: True values (original dollar scale)
        y_pred: Predicted values (original dollar scale)

    Returns:
        {'MAE': ..., 'RMSE': ..., 'R2': ..., 'MAPE': ...}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2':   r2_score(y_true, y_pred),
        'MAPE': _mape(y_true, y_pred),
    }

evaluate_model = compute_metrics

# ============================================================================
# MODEL EVALUATOR  (main class — use this everywhere)
# ============================================================================

class ModelEvaluator:
    """
    Handles prediction, inverse transform, metric calculation, and reporting.

    All public methods work on original dollar scale internally —
    you never have to remember to call expm1 yourself.

    Attributes:
        feature_names: Column names from training, used for residual analysis.
    """

    def __init__(self, feature_names: Optional[list] = None):
        """
        Args:
            feature_names: List of feature column names (from X_train.columns).
                           Used later for per-feature error analysis.
        """
        self.feature_names = feature_names or []

    # ── single-model evaluation ───────────────────────────────────────────

    def evaluate(
        self,
        model,
        X: pd.DataFrame,
        y_true_orig: pd.Series,
        label: str = "Model"
    ) -> Dict[str, float]:
        """
        Predict → inverse-transform → compute metrics → print.

        Args:
            model:        Trained sklearn-compatible model (predicts log scale).
            X:            Feature DataFrame (already preprocessed).
            y_true_orig:  True prices in original dollar scale.
            label:        Name shown in the printed summary.

        Returns:
            Metrics dict with keys MAE, RMSE, R2, MAPE.
        """
        # predict in log scale, convert back to dollars
        y_pred_log = model.predict(X)
        y_pred     = np.expm1(y_pred_log)

        metrics = compute_metrics(y_true_orig.values, y_pred)

        self._print_metrics(metrics, title=f"{label} — Evaluation Results")

        return metrics

    # ── multi-model comparison ────────────────────────────────────────────

    def compare_all(
        self,
        models: Dict[str, object],
        X: pd.DataFrame,
        y_true_orig: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate every model in the dict and return a comparison table.

        Args:
            models:       {name: trained_model} — e.g. manager.models
            X:            Feature DataFrame.
            y_true_orig:  True prices in original dollar scale.

        Returns:
            DataFrame sorted by R² descending, one row per model.
        """
        rows = []
        for name, model in models.items():
            y_pred = np.expm1(model.predict(X))
            m      = compute_metrics(y_true_orig.values, y_pred)
            m['Model'] = name
            rows.append(m)

        df = (
            pd.DataFrame(rows)[['Model', 'MAE', 'RMSE', 'R2', 'MAPE']]
            .sort_values('R2', ascending=False)
            .reset_index(drop=True)
        )

        # pretty-print
        print("\n" + "=" * 65)
        print(" MODEL COMPARISON")
        print("=" * 65)
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print("=" * 65 + "\n")

        return df

    # ── residual analysis ─────────────────────────────────────────────────

    def residual_summary(
        self,
        model,
        X: pd.DataFrame,
        y_true_orig: pd.Series
    ) -> pd.DataFrame:
        """
        Compute residuals and return a summary by price bucket.

        Buckets prices into quintiles so you can see where the model
        struggles (e.g. cheap machines vs expensive ones).

        Args:
            model:        Trained model.
            X:            Feature DataFrame.
            y_true_orig:  True prices in original dollar scale.

        Returns:
            DataFrame with columns: Price Range, Count, MAE, RMSE, R², MAPE
        """
        y_true = y_true_orig.values.astype(float)
        y_pred = np.expm1(model.predict(X))

        # vectorized quintile labeling
        labels  = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        buckets = pd.qcut(y_true, q=5, labels=labels, duplicates='drop')

        rows = []
        for label in labels:
            mask = (buckets == label)
            if mask.sum() == 0:
                continue
            m = compute_metrics(y_true[mask], y_pred[mask])
            m['Price Range'] = label
            m['Count']       = int(mask.sum())
            rows.append(m)

        df = pd.DataFrame(rows)[['Price Range', 'Count', 'MAE', 'RMSE', 'R2', 'MAPE']]

        print("\n" + "=" * 65)
        print(" RESIDUAL ANALYSIS — BY PRICE BUCKET")
        print("=" * 65)
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print("=" * 65 + "\n")

        return df

    # ── worst predictions ─────────────────────────────────────────────────

    def worst_predictions(
        self,
        model,
        X: pd.DataFrame,
        y_true_orig: pd.Series,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Return the N predictions with the largest absolute error.

        Useful for manual inspection during error analysis.

        Args:
            model:        Trained model.
            X:            Feature DataFrame.
            y_true_orig:  True prices in original dollar scale.
            n:            How many worst predictions to show.

        Returns:
            DataFrame sorted by Error descending.
        """
        y_true     = y_true_orig.values.astype(float)
        y_pred     = np.expm1(model.predict(X))
        abs_errors = np.abs(y_true - y_pred)

        # vectorized top-n selection (no sorting the full array)
        top_idx = np.argpartition(abs_errors, -n)[-n:]
        top_idx = top_idx[np.argsort(abs_errors[top_idx])[::-1]]

        df = pd.DataFrame({
            'True Price ($)':      y_true[top_idx],
            'Predicted Price ($)': y_pred[top_idx],
            'Error ($)':           abs_errors[top_idx],
            'Error (%)':           (abs_errors[top_idx] / y_true[top_idx]) * 100,
        }).round(2)

        print("\n" + "=" * 65)
        print(f" TOP {n} WORST PREDICTIONS")
        print("=" * 65)
        print(df.to_string(index=False))
        print("=" * 65 + "\n")

        return df

    # ── save results ──────────────────────────────────────────────────────

    def save_results(
        self,
        comparison_df: pd.DataFrame,
        filename: str = "evaluation_results.csv"
    ) -> None:
        """
        Persist the comparison table to reports/.

        Args:
            comparison_df: DataFrame from compare_all().
            filename:      Output filename.
        """
        path = REPORTS_DIR / filename
        comparison_df.to_csv(path, index=False)
        print(f"[evaluation] Results saved to {path}")

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _print_metrics(metrics: Dict[str, float], title: str = "Evaluation") -> None:
        """Pretty-print a single metrics dict."""
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")
        print(f"  MAE   : ${metrics['MAE']:>12,.2f}")
        print(f"  RMSE  : ${metrics['RMSE']:>12,.2f}")
        print(f"  R²    : {metrics['R2']:>13.4f}")
        print(f"  MAPE  : {metrics['MAPE']:>12.2f}%")
        print(f"{'=' * 50}\n")