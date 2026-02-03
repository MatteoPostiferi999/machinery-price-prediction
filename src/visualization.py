"""
visualization.py
================
The 5 presentation-ready plots for the SHM machinery price prediction project.

Each function is self-contained: pass in the data, get a saved PNG in reports/.
Style is consistent across all plots (seaborn whitegrid, same palette, same DPI).

Plots generated:
    1. Target distribution          — histogram + KDE with summary stats
    2. Feature importance (top 15)  — horizontal bar chart from LightGBM
    3. Actual vs Predicted scatter  — best model, colored by absolute error
    4. Residual distribution        — histogram + KDE with zero-line
    5. Model comparison             — grouped bar chart (MAE, RMSE, R2)

How to call (from main.py):
    from src.visualization import generate_all_plots

    generate_all_plots(
        df_raw=df,                          # raw DataFrame (for target dist)
        manager=manager,                    # fitted ModelManager
        X_test=data['X_test'],              # test features
        y_test_orig=data['y_test_orig'],    # test target (original scale)
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
from scipy.stats import gaussian_kde


from .config import TARGET_COLUMN, REPORTS_DIR, PLOT_DPI

# ─── global style ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2")


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to reports/ and close it."""
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"[Visualization] Saved -> {path}")
    plt.close(fig)


# ============================================================================
# 1. TARGET DISTRIBUTION
# ============================================================================

def plot_target_distribution(df: pd.DataFrame) -> None:
    """
    Histogram + KDE of Sales Price with summary statistics overlay.

    Shows mean, median, and std as vertical/text annotations so the
    viewer immediately understands the shape and spread of the target.

    Args:
        df: Raw DataFrame containing TARGET_COLUMN.
    """
    prices = df[TARGET_COLUMN].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram
    ax.hist(prices, bins=80, color=PALETTE[0], edgecolor="black", alpha=0.7, label="Sales Price")

    # KDE overlay
    kde_x = np.linspace(prices.min(), prices.max(), 500)
    kde = gaussian_kde(prices)
    ax.twinx().plot(kde_x, kde(kde_x), color="black", linewidth=1.5, label="KDE")

    # Statistics lines
    mean_val   = prices.mean()
    median_val = prices.median()
    ax.axvline(mean_val,   color="red",    linestyle="--", linewidth=1.5, label=f"Mean  ${mean_val:,.0f}")
    ax.axvline(median_val, color="orange", linestyle="--", linewidth=1.5, label=f"Median ${median_val:,.0f}")

    # Stats box
    stats_text = (
        f"n = {len(prices):,}\n"
        f"mean  = ${mean_val:,.0f}\n"
        f"median = ${median_val:,.0f}\n"
        f"std   = ${prices.std():,.0f}"
    )
    ax.text(
        0.98, 0.95, stats_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax.set_title("Distribution of Sales Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sales Price ($)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")

    _save(fig, "plot_target_distribution.png")


# ============================================================================
# 2. FEATURE IMPORTANCE (top 15)
# ============================================================================

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Horizontal bar chart of the top-N most important features.

    Expects a DataFrame with columns 'Feature' and 'Importance'
    — exactly what ModelManager.get_feature_importance() returns.

    Args:
        importance_df: DataFrame from manager.get_feature_importance().
        top_n:         Number of features to show (default 15).
    """
    df = importance_df.head(top_n).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(df["Feature"], df["Importance"], color=PALETTE[1], edgecolor="black")

    # Value labels at end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{width:,.0f}", va="center", fontsize=8)

    ax.set_title("Feature Importance — LightGBM (Top 15)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance (gain)")
    ax.set_xlim(0, df["Importance"].max() * 1.12)  # room for labels

    _save(fig, "plot_feature_importance.png")


# ============================================================================
# 3. ACTUAL VS PREDICTED
# ============================================================================

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """
    Scatter of actual vs predicted prices, colored by absolute error.

    The color gradient makes it immediately visible where the model
    struggles — dark points = large errors.

    Args:
        y_true:     True prices (original dollar scale).
        y_pred:     Predicted prices (original dollar scale).
        model_name: Label for the plot title.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter colored by error magnitude
    sc = ax.scatter(
        y_true, y_pred,
        c=errors, cmap="YlOrRd",
        alpha=0.6, edgecolors="black", linewidths=0.4, s=20
    )
    fig.colorbar(sc, ax=ax, label="Absolute Error ($)")

    # Ideal line (y = x)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y = x)")

    # R² and RMSE annotation
    from sklearn.metrics import r2_score, mean_squared_error
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(
        0.05, 0.95,
        f"R² = {r2:.4f}\nRMSE = ${rmse:,.0f}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.legend(loc="lower right")

    _save(fig, "plot_actual_vs_predicted.png")


# ============================================================================
# 4. RESIDUAL DISTRIBUTION
# ============================================================================

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """
    Histogram + KDE of residuals (actual - predicted) with zero-line.

    A well-behaved model produces residuals centered on zero with
    roughly symmetric spread. Skew or heavy tails signal systematic bias.

    Args:
        y_true:     True prices (original dollar scale).
        y_pred:     Predicted prices (original dollar scale).
        model_name: Label for the plot title.
    """
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram
    ax.hist(residuals, bins=60, color=PALETTE[2], edgecolor="black", alpha=0.7)

    # KDE overlay
    kde_x = np.linspace(residuals.min(), residuals.max(), 500)
    kde = gaussian_kde(residuals)
    ax.twinx().plot(kde_x, kde(kde_x), color="black", linewidth=1.5)

    # Zero line
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)

    # Stats box
    stats_text = (
        f"mean  = ${residuals.mean():>+,.0f}\n"
        f"std   = ${residuals.std():,.0f}\n"
        f"skew  = {pd.Series(residuals).skew():.2f}"
    )
    ax.text(
        0.02, 0.95, stats_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax.set_title(f"Residual Distribution — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Residual: Actual − Predicted ($)")
    ax.set_ylabel("Count")

    _save(fig, "plot_residuals.png")


# ============================================================================
# 5. MODEL COMPARISON
# ============================================================================

def plot_model_comparison(results: Dict[str, Dict[str, float]], model_labels: Dict[str, str]) -> None:
    """
    Grouped bar chart comparing MAE, RMSE, and R² across all models.

    MAE and RMSE share the left axis (dollars).  R² is on the right axis
    (0–1 scale) so both scales are readable in one chart.

    Args:
        results:       {model_key: {'MAE': ..., 'RMSE': ..., 'R2': ...}}
                       — exactly manager.results
        model_labels:  {model_key: display_name}
                       — e.g. {'ridge': 'Ridge Regression', 'lightgbm': 'LightGBM'}
    """
    names  = [model_labels.get(k, k) for k in results]
    maes   = [results[k]['MAE']  for k in results]
    rmses  = [results[k]['RMSE'] for k in results]
    r2s    = [results[k]['R2']   for k in results]

    x      = np.arange(len(names))
    width  = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # MAE and RMSE bars (left axis — dollars)
    bars_mae  = ax1.bar(x - width, maes,  width, label="MAE ($)",  color=PALETTE[0], edgecolor="black")
    bars_rmse = ax1.bar(x,         rmses, width, label="RMSE ($)", color=PALETTE[1], edgecolor="black")

    ax1.set_ylabel("Error ($)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)

    # Value labels on MAE/RMSE bars
    for bar in bars_mae + bars_rmse:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                 f"${h:,.0f}", ha="center", va="bottom", fontsize=8)

    # R² bars (right axis — 0 to 1)
    ax2 = ax1.twinx()
    bars_r2 = ax2.bar(x + width, r2s, width, label="R²", color=PALETTE[3], edgecolor="black")
    ax2.set_ylabel("R² Score")
    ax2.set_ylim(0, 1.15)  # headroom for labels

    for bar in bars_r2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()

    _save(fig, "plot_model_comparison.png")


# ============================================================================
# ORCHESTRATOR — called from main.py
# ============================================================================

def generate_all_plots(
    df_raw: pd.DataFrame,
    manager,          # ModelManager instance
    X_test: pd.DataFrame,
    y_test_orig: pd.Series,
) -> None:
    """
    Generate all 5 presentation plots in one call.

    Args:
        df_raw:      Raw DataFrame (for target distribution).
        manager:     Fitted ModelManager (holds models, results, feature importance).
        X_test:      Test feature matrix (preprocessed).
        y_test_orig: Test target in original dollar scale.
    """
    print("\n" + "=" * 70)
    print(" GENERATING VISUALIZATIONS")
    print("=" * 70)

    # 1 — Target distribution
    plot_target_distribution(df_raw)

    # 2 — Feature importance (LightGBM)
    importance_df = manager.get_feature_importance("lightgbm", top_n=15)
    if importance_df is not None:
        plot_feature_importance(importance_df)

    # 3 & 4 — Actual vs Predicted + Residuals (best model on test set)
    best_name, best_model = manager.get_best_model()
    best_label = {"ridge": "Ridge Regression", "lightgbm": "LightGBM"}.get(best_name, best_name)

    y_pred = np.expm1(best_model.predict(X_test))
    y_true = y_test_orig.values.astype(float)

    plot_actual_vs_predicted(y_true, y_pred, best_label)
    plot_residuals(y_true, y_pred, best_label)

    # 5 — Model comparison (all models)
    model_labels = {"ridge": "Ridge Regression", "lightgbm": "LightGBM"}
    plot_model_comparison(manager.results, model_labels)

    print("\n[Visualization] All plots saved to reports/")