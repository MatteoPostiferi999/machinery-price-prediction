"""
visualization.py
----------------
All plotting lives here — both EDA (exploratory) and post-training
performance charts.  Every function saves its figure to the report/
folder so the final report can reference them directly.

EDA:
  - Target distribution
  - Correlation heatmap
  - Missing-value bar chart
  - Feature distributions (top N by variance)

Performance:
  - Actual vs Predicted scatter (per model)
  - Residual distribution (per model)
  - Model comparison bar chart (RMSE)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

from config import TARGET_COLUMN, REPORT_DIR

# ─── style ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 5)


def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to report/ and close it."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Visualization] Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  EDA
# ═══════════════════════════════════════════════════════════════════

def plot_target_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of the target column."""
    fig, ax = plt.subplots()
    df[TARGET_COLUMN].hist(bins=50, ax=ax, edgecolor="black", alpha=0.7)
    ax.set_title(f"Distribution of {TARGET_COLUMN}")
    ax.set_xlabel(TARGET_COLUMN)
    ax.set_ylabel("Count")
    _save(fig, "eda_target_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of pairwise correlations (numerical columns only)."""
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(max(8, corr.shape[0] * 0.6), max(6, corr.shape[0] * 0.5)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    _save(fig, "eda_correlation_heatmap.png")


def plot_missing_values(df: pd.DataFrame) -> None:
    """Bar chart of missing-value counts (only columns that have any)."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        print("[Visualization] No missing values — skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.4)))
    missing.plot.barh(ax=ax, color="salmon", edgecolor="black")
    ax.set_title("Missing Values per Column")
    ax.set_xlabel("Count")
    _save(fig, "eda_missing_values.png")


def plot_feature_distributions(df: pd.DataFrame, top_n: int = 6) -> None:
    """Pairplot-style grid of the top-N numerical features by variance."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in num_cols:
        num_cols.remove(TARGET_COLUMN)
    # pick top N by variance
    top_cols = df[num_cols].var().nlargest(top_n).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(top_cols[:6]):
        df[col].hist(bins=40, ax=axes[i], edgecolor="black", alpha=0.7)
        axes[i].set_title(col)
    for j in range(i + 1, 6):
        axes[j].set_visible(False)
    fig.suptitle("Top Feature Distributions (by variance)", y=1.02)
    fig.tight_layout()
    _save(fig, "eda_feature_distributions.png")


# ═══════════════════════════════════════════════════════════════════
#  Performance
# ═══════════════════════════════════════════════════════════════════

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Scatter of actual vs predicted with the ideal 1:1 line."""
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="black", linewidths=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y = x)")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    _save(fig, f"perf_actual_vs_predicted_{model_name}.png")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Histogram of residuals (actual − predicted)."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"Residual Distribution — {model_name}")
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Count")
    _save(fig, f"perf_residuals_{model_name}.png")


def plot_model_comparison(summary: pd.DataFrame) -> None:
    """Horizontal bar chart comparing RMSE across models."""
    fig, ax = plt.subplots()
    summary_sorted = summary.sort_values("RMSE", ascending=True)
    ax.barh(summary_sorted["model"], summary_sorted["RMSE"], color="steelblue", edgecolor="black")
    ax.set_xlabel("RMSE")
    ax.set_title("Model Comparison — RMSE")
    ax.invert_yaxis()  # best (lowest) on top
    fig.tight_layout()
    _save(fig, "perf_model_comparison.png")