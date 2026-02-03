"""
evaluation.py
=============
Model evaluation functions with MAE, RMSE, and R² metrics.

All metrics are calculated on the original price scale (not log-transformed).
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import PRIMARY_METRICS, METRIC_DESCRIPTIONS


# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Formula: MAE = mean(|y_true - y_pred|)
    
    Interpretation: Average dollar amount of error.
    Lower is better.
    
    Args:
        y_true: True values (original scale)
        y_pred: Predicted values (original scale)
        
    Returns:
        MAE in dollars
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Formula: RMSE = sqrt(mean((y_true - y_pred)²))
    
    Interpretation: Penalizes large errors more than MAE.
    Lower is better.
    
    Args:
        y_true: True values (original scale)
        y_pred: Predicted values (original scale)
        
    Returns:
        RMSE in dollars
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (Coefficient of Determination).
    
    Formula: R² = 1 - (SS_res / SS_tot)
    
    Interpretation: Proportion of variance explained by the model.
    Range: (-inf, 1], where 1 is perfect prediction.
    Higher is better.
    
    Args:
        y_true: True values (original scale)
        y_pred: Predicted values (original scale)
        
    Returns:
        R² score
    """
    return r2_score(y_true, y_pred)


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list = PRIMARY_METRICS
) -> Dict[str, float]:
    """
    Evaluate model predictions using specified metrics.
    
    Args:
        y_true: True values (original scale, in dollars)
        y_pred: Predicted values (original scale, in dollars)
        metrics: List of metrics to calculate (default from config)
        
    Returns:
        Dictionary with metric names and values
    """
    results = {}
    
    for metric in metrics:
        if metric == 'MAE':
            results['MAE'] = calculate_mae(y_true, y_pred)
        elif metric == 'RMSE':
            results['RMSE'] = calculate_rmse(y_true, y_pred)
        elif metric == 'R2':
            results['R2'] = calculate_r2(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return results


def evaluate_model_detailed(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, any]:
    """
    Detailed evaluation with additional statistics.
    
    Args:
        y_true: True values (original scale)
        y_pred: Predicted values (original scale)
        
    Returns:
        Dictionary with metrics and additional statistics
    """
    # Calculate main metrics
    metrics = evaluate_model(y_true, y_pred)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Additional statistics
    metrics.update({
        'Mean_Error': np.mean(residuals),
        'Median_Error': np.median(np.abs(residuals)),
        'Max_Error': np.max(np.abs(residuals)),
        'Std_Error': np.std(residuals),
        'Residuals_Mean': np.mean(residuals),
        'Residuals_Std': np.std(residuals)
    })
    
    return metrics


def print_metrics(
    metrics: Dict[str, float],
    title: str = "Model Evaluation"
) -> None:
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for metric_name in PRIMARY_METRICS:
        if metric_name in metrics:
            value = metrics[metric_name]
            
            if metric_name in ['MAE', 'RMSE']:
                print(f"{metric_name:>10}: ${value:>12,.2f}")
            else:  # R2
                print(f"{metric_name:>10}: {value:>13.4f}")
    
    print(f"{'='*60}\n")


def compare_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_to_compare: str = 'R2'
) -> pd.DataFrame:
    """
    Compare a specific metric across multiple models.
    
    Args:
        metrics_dict: Dictionary of {model_name: {metric_name: value}}
        metric_to_compare: Which metric to compare
        
    Returns:
        DataFrame sorted by the specified metric
    """
    data = []
    
    for model_name, metrics in metrics_dict.items():
        data.append({
            'Model': model_name,
            'MAE': metrics.get('MAE', np.nan),
            'RMSE': metrics.get('RMSE', np.nan),
            'R2': metrics.get('R2', np.nan)
        })
    
    df = pd.DataFrame(data)
    
    # Sort by metric (descending for R2, ascending for errors)
    ascending = (metric_to_compare != 'R2')
    df = df.sort_values(metric_to_compare, ascending=ascending)
    
    return df


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentiles: list = [10, 25, 50, 75, 90]
) -> pd.DataFrame:
    """
    Analyze prediction errors by percentile.
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        percentiles: List of percentiles to calculate
        
    Returns:
        DataFrame with error statistics by percentile
    """
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)
    
    results = []
    
    for p in percentiles:
        results.append({
            'Percentile': f'{p}th',
            'True Price': f"${np.percentile(y_true, p):,.0f}",
            'Mean Error': f"${np.percentile(abs_errors, p):,.0f}",
            'Max Error': f"${np.max(abs_errors[y_true <= np.percentile(y_true, p)]):,.0f}"
        })
    
    return pd.DataFrame(results)


def get_worst_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 10
) -> pd.DataFrame:
    """
    Get the worst N predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n: Number of worst predictions to return
        
    Returns:
        DataFrame with worst predictions
    """
    abs_errors = np.abs(y_true - y_pred)
    worst_indices = np.argsort(abs_errors)[-n:][::-1]
    
    results = []
    for idx in worst_indices:
        results.append({
            'True Price': f"${y_true[idx]:,.0f}",
            'Predicted Price': f"${y_pred[idx]:,.0f}",
            'Error': f"${abs_errors[idx]:,.0f}",
            'Error %': f"{(abs_errors[idx] / y_true[idx] * 100):.1f}%"
        })
    
    return pd.DataFrame(results)