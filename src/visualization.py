"""
visualization.py
================
The 5 presentation-ready plots for the SHM machinery price prediction project.

Each function is self-contained: pass in the data, get a saved PNG in reports/.
Style is consistent across all plots (configured in config.py).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from scipy.stats import gaussian_kde

from .config import TARGET_COLUMN, REPORTS_DIR, PLOT_SETTINGS

# ─── Global Style Configuration ───────────────────────────────────────────
# Apply centralized style settings from config.py
sns.set_theme(
    style=PLOT_SETTINGS['STYLE'],
    font_scale=1.1,
    rc={"axes.grid": True, "grid.alpha": 0.6}
)
PALETTE = sns.color_palette(PLOT_SETTINGS['PALETTE'])


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to reports/ using project-wide DPI settings."""
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / filename
    
    fig.savefig(path, dpi=PLOT_SETTINGS['DPI'], bbox_inches="tight")
    print(f"[Visualization] Saved -> {path}")
    plt.close(fig)


# ============================================================================
# 1. TARGET DISTRIBUTION
# ============================================================================

def plot_target_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of Sales Price with summary statistics overlay."""
    prices = df[TARGET_COLUMN].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram
    ax.hist(prices, bins=80, color=PALETTE[0], edgecolor="black", alpha=0.7, label="Sales Price")

    # KDE overlay (secondary axis to match scale)
    kde_x = np.linspace(prices.min(), prices.max(), 500)
    kde = gaussian_kde(prices)
    ax2 = ax.twinx()
    ax2.plot(kde_x, kde(kde_x), color="#333333", linewidth=2, label="KDE")
    ax2.set_yticks([]) # Hide KDE axis ticks for cleaner look

    # Statistics lines
    mean_val = prices.mean()
    median_val = prices.median()
    
    ax.axvline(mean_val,   color="firebrick", linestyle="--", linewidth=1.5, label=f"Mean  ${mean_val:,.0f}")
    ax.axvline(median_val, color="orange",    linestyle="--", linewidth=1.5, label=f"Median ${median_val:,.0f}")

    # Stats box
    stats_text = (
        f"n = {len(prices):,}\n"
        f"mean  = ${mean_val:,.0f}\n"
        f"median = ${median_val:,.0f}\n"
        f"std   = ${prices.std():,.0f}"
    )
    ax.text(
        0.98, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc")
    )

    ax.set_title("Distribution of Sales Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sales Price ($)")
    ax.set_ylabel("Count")
    
    # Combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

    _save(fig, "plot_target_distribution.png")


# ============================================================================
# 2. FEATURE IMPORTANCE (top 15)
# ============================================================================

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart of the top-N most important features."""
    df = importance_df.head(top_n).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(df["Feature"], df["Importance"], color=PALETTE[1], edgecolor="black", alpha=0.8)

    # Value labels at end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{width:,.1f}", va="center", fontsize=9, color="#555555")

    ax.set_title(f"Feature Importance (Top {top_n})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance (Gain)")
    ax.set_xlim(0, df["Importance"].max() * 1.15)  # Add breathing room for labels

    _save(fig, "plot_feature_importance.png")


# ============================================================================
# 3. ACTUAL VS PREDICTED
# ============================================================================

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Scatter of actual vs predicted prices, colored by absolute error magnitude."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter colored by error magnitude
    sc = ax.scatter(
        y_true, y_pred,
        c=errors, cmap="inferno_r", # Darker = higher error
        alpha=0.6, edgecolors="none", s=25
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Absolute Error ($)", rotation=270, labelpad=15)

    # Ideal line (y = x)
    lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1.5, alpha=0.5, label="Ideal Fit (y=x)")

    # Performance metrics annotation
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    stats_text = (
        f"R² = {r2:.4f}\n"
        f"MAE = ${mae:,.0f}\n"
        f"RMSE = ${rmse:,.0f}"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc")
    )

    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc="lower right")

    _save(fig, "plot_actual_vs_predicted.png")


# ============================================================================
# 4. RESIDUAL DISTRIBUTION
# ============================================================================

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Histogram + KDE of residuals (Actual - Predicted) to check for bias."""
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(residuals, bins=70, color=PALETTE[2], edgecolor="black", alpha=0.7, density=True)

    # KDE overlay
    kde_x = np.linspace(residuals.min(), residuals.max(), 500)
    kde = gaussian_kde(residuals)
    ax.plot(kde_x, kde(kde_x), color="#333333", linewidth=2, label="KDE")

    # Zero line (Unbiased reference)
    ax.axvline(0, color="firebrick", linestyle="--", linewidth=2, label="Zero Bias")

    # Stats box
    stats_text = (
        f"Mean = ${residuals.mean():>+,.0f}\n"
        f"Std  = ${residuals.std():,.0f}\n"
        f"Skew = {pd.Series(residuals).skew():.2f}"
    )
    ax.text(
        0.02, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc")
    )

    ax.set_title(f"Residual Distribution — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Residual ($) [Actual - Predicted]")
    ax.set_ylabel("Density")
    ax.legend()

    _save(fig, "plot_residuals.png")


# ============================================================================
# 5. MODEL COMPARISON
# ============================================================================

def plot_model_comparison(results: Dict[str, Dict[str, float]], model_labels: Dict[str, str]) -> None:
    """Grouped bar chart comparing MAE, RMSE (Left Axis) and R² (Right Axis)."""
    names = [model_labels.get(k, k) for k in results]
    maes = [results[k]['MAE'] for k in results]
    rmses = [results[k]['RMSE'] for k in results]
    r2s = [results[k]['R2'] for k in results]

    x = np.arange(len(names))
    width = 0.3

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bars for Errors (Left Axis)
    rects1 = ax1.bar(x - width/2, maes, width, label='MAE ($)', color=PALETTE[0], edgecolor='black', alpha=0.9)
    rects2 = ax1.bar(x + width/2, rmses, width, label='RMSE ($)', color=PALETTE[1], edgecolor='black', alpha=0.9)

    # Right Axis for R2
    ax2 = ax1.twinx()
    # Plot R2 as points connected by a line instead of bars (cleaner visualization for dual-axis)
    ax2.plot(x, r2s, color=PALETTE[3], marker='o', markersize=10, linewidth=2, label='R² Score')
    
    # Formatting
    ax1.set_ylabel('Error ($)', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12, color=PALETTE[3])
    ax2.tick_params(axis='y', labelcolor=PALETTE[3])
    
    # Dynamic limit for R2 to avoid cutting off data if performance is poor
    min_r2 = min(r2s)
    bottom_lim = 0 if min_r2 > 0 else min_r2 - 0.1
    ax2.set_ylim(bottom_lim, 1.1)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11, fontweight='medium')
    ax1.legend(loc='upper left', frameon=True)
    ax2.legend(loc='upper right', frameon=True)

    # Value Labels for Bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'${height/1000:.0f}k',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)

    # Value Labels for R2 Points
    for i, r2 in enumerate(r2s):
        ax2.annotate(f'{r2:.3f}', (x[i], r2), xytext=(0, 10), 
                     textcoords="offset points", ha='center', color=PALETTE[3], fontweight='bold')

    ax1.set_title("Model Comparison Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()

    _save(fig, "plot_model_comparison.png")


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def generate_all_plots(
    df_raw: pd.DataFrame,
    manager: Any,  # Typed as Any to avoid circular import with ModelManager
    X_test: pd.DataFrame,
    y_test_orig: pd.Series,
) -> None:
    """Generate all 5 presentation plots in one call."""
    print("\n" + "=" * 70)
    print(" GENERATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Target Distribution
    plot_target_distribution(df_raw)

    # 2. Feature Importance (LightGBM preferred)
    importance_df = manager.get_feature_importance("lightgbm", top_n=15)
    if importance_df is not None:
        plot_feature_importance(importance_df)

    # 3 & 4. Predictions & Residuals (Best Model)
    best_name, best_model = manager.get_best_model()
    best_label = {"ridge": "Ridge Regression", "lightgbm": "LightGBM"}.get(best_name, best_name)

    # Predict: Assume model outputs Log Price, so we apply expm1 to get real Dollars
    # Note: Ensure this assumption matches your model training logic.
    y_pred = np.expm1(best_model.predict(X_test))
    y_true = y_test_orig.values.astype(float)

    plot_actual_vs_predicted(y_true, y_pred, best_label)
    plot_residuals(y_true, y_pred, best_label)

    # 5. Comparison
    model_labels = {"ridge": "Ridge Regression", "lightgbm": "LightGBM"}
    plot_model_comparison(manager.results, model_labels)

    print("\n[Visualization] All plots saved to reports/")