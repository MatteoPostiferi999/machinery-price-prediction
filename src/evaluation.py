"""
evaluation.py - Model evaluation metrics and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import REPORTS_DIR


# ============================================================================
# CORE METRICS (Vectorized)
# ============================================================================

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (handles division by zero)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics (MAE, RMSE, R², MAPE)."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    
    return {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2':   r2_score(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
    }


# Alias for backward compatibility
evaluate_model = compute_metrics


# ============================================================================
# HELPER: Predict with Log-Inverse Transform
# ============================================================================

def predict_original_scale(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict and inverse log-transform (log -> dollars)."""
    return np.expm1(model.predict(X))


# ============================================================================
# MODEL EVALUATOR (Sklearn-Compatible)
# ============================================================================

class ModelEvaluator(BaseEstimator):
    """
    Sklearn-compatible evaluator for model assessment and reporting.
    
    Features:
        - Single model evaluation
        - Multi-model comparison (leaderboard)
        - Residual analysis by price range
        - Outlier detection
        - CSV export
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
    
    # ── Single Model ─────────────────────────────────────────────────────
    
    def evaluate(self, model: Any, X: pd.DataFrame, y_true_orig: pd.Series, label: str = "Model") -> Dict[str, float]:
        """Evaluate single model (handles log-inverse automatically)."""
        y_pred = predict_original_scale(model, X)
        metrics = compute_metrics(y_true_orig.values, y_pred)
        print_metrics(metrics, title=f"{label} — Evaluation")
        return metrics
    
    # ── Multi-Model Comparison ───────────────────────────────────────────
    
    def compare_all(self, models: Dict[str, Any], X: pd.DataFrame, y_true_orig: pd.Series) -> pd.DataFrame:
        """Evaluate multiple models and return leaderboard."""
        rows = [
            {**compute_metrics(y_true_orig.values, predict_original_scale(model, X)), 'Model': name}
            for name, model in models.items()
        ]
        
        df = pd.DataFrame(rows)[['Model', 'MAE', 'RMSE', 'R2', 'MAPE']].sort_values('R2', ascending=False).reset_index(drop=True)
        
        print("\n" + "=" * 65)
        print(" MODEL COMPARISON LEADERBOARD")
        print("=" * 65)
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print("=" * 65 + "\n")
        
        return df
    
    # ── Residual Analysis ────────────────────────────────────────────────
    
    def residual_summary(self, model: Any, X: pd.DataFrame, y_true_orig: pd.Series) -> pd.DataFrame:
        """Analyze errors by price quintiles (reveals bias)."""
        y_true = y_true_orig.values.astype(float)
        y_pred = predict_original_scale(model, X)
        
        # Create price buckets (quintiles)
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        buckets = pd.qcut(y_true, q=5, labels=labels, duplicates='drop')
        
        rows = []
        for label in labels:
            mask = (buckets == label)
            if mask.sum() == 0:
                continue
            
            m = compute_metrics(y_true[mask], y_pred[mask])
            m.update({'Price Range': label, 'Count': int(mask.sum())})
            rows.append(m)
        
        df = pd.DataFrame(rows)[['Price Range', 'Count', 'MAE', 'RMSE', 'R2', 'MAPE']]
        
        print("\n" + "=" * 65)
        print(" RESIDUAL ANALYSIS — BY PRICE SEGMENT")
        print("=" * 65)
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print("=" * 65 + "\n")
        
        return df
    
    # ── Outlier Detection ────────────────────────────────────────────────
    
    def worst_predictions(self, model: Any, X: pd.DataFrame, y_true_orig: pd.Series, n: int = 10) -> pd.DataFrame:
        """Identify N most erroneous predictions (for debugging edge cases)."""
        y_true = y_true_orig.values.astype(float)
        y_pred = predict_original_scale(model, X)
        abs_errors = np.abs(y_true - y_pred)
        
        # Optimized top-N selection (O(n) with argpartition)
        top_idx = np.argpartition(abs_errors, -n)[-n:]
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
    
    # ── Persistence ──────────────────────────────────────────────────────
    
    def save_results(self, df: pd.DataFrame, filename: str = "evaluation_results.csv") -> None:
        """Export evaluation results to CSV."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / filename
        df.to_csv(path, index=False)
        print(f"[Evaluation] Results saved to {path}")


# ============================================================================
# HELPER: Print Formatted Metrics
# ============================================================================

def print_metrics(metrics: Dict[str, float], title: str = "Evaluation") -> None:
    """Print metrics in formatted box."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  MAE   : ${metrics['MAE']:>12,.2f}")
    print(f"  RMSE  : ${metrics['RMSE']:>12,.2f}")
    print(f"  R²    : {metrics['R2']:>13.4f}")
    print(f"  MAPE  : {metrics['MAPE']:>12.2f}%")
    print(f"{'=' * 50}\n")