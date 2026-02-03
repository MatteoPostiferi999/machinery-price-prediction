"""
visualization.py - Presentation-ready plots for ML project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from scipy.stats import gaussian_kde

from .config import TARGET_COLUMN, REPORTS_DIR, PLOT_SETTINGS

# Global style
sns.set_theme(
    style=PLOT_SETTINGS['STYLE'],
    font_scale=1.1,
    rc={"axes.grid": True, "grid.alpha": 0.6}
)
PALETTE = sns.color_palette(PLOT_SETTINGS['PALETTE'])


# ============================================================================
# HELPER
# ============================================================================

def save_plot(fig: plt.Figure, filename: str) -> None:
    """Save figure to reports/ with project-wide DPI."""
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / filename
    fig.savefig(path, dpi=PLOT_SETTINGS['DPI'], bbox_inches="tight")
    print(f"[Plot] Saved → {path.name}")
    plt.close(fig)


def add_stats_box(ax: plt.Axes, text: str, loc: str = "upper right") -> None:
    """Add white stats box to plot (DRY helper)."""
    positions = {
        "upper right": (0.98, 0.95, "top", "right"),
        "upper left": (0.02, 0.95, "top", "left"),
        "lower left": (0.02, 0.05, "bottom", "left"),
    }
    x, y, va, ha = positions.get(loc, positions["upper right"])
    
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=10,
        verticalalignment=va, horizontalalignment=ha,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#ccc")
    )


# ============================================================================
# 1. TARGET DISTRIBUTION
# ============================================================================

def plot_target_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of sales price with summary statistics."""
    prices = df[TARGET_COLUMN].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Histogram
    ax.hist(prices, bins=80, color=PALETTE[0], edgecolor="black", alpha=0.7, label="Sales Price")
    
    # KDE overlay
    kde_x = np.linspace(prices.min(), prices.max(), 500)
    kde = gaussian_kde(prices)
    ax2 = ax.twinx()
    ax2.plot(kde_x, kde(kde_x), color="#333", linewidth=2, label="KDE")
    ax2.set_yticks([])
    
    # Stats lines
    mean_val, median_val = prices.mean(), prices.median()
    ax.axvline(mean_val, color="firebrick", linestyle="--", linewidth=1.5, label=f"Mean ${mean_val:,.0f}")
    ax.axvline(median_val, color="orange", linestyle="--", linewidth=1.5, label=f"Median ${median_val:,.0f}")
    
    # Stats box
    stats = f"n={len(prices):,}\nmean=${mean_val:,.0f}\nmedian=${median_val:,.0f}\nstd=${prices.std():,.0f}"
    add_stats_box(ax, stats)
    
    ax.set_title("Distribution of Sales Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sales Price ($)")
    ax.set_ylabel("Count")
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    save_plot(fig, "plot_target_distribution.png")


# ============================================================================
# 2. FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart of top-N features."""
    df = importance_df.head(top_n).sort_values("Importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df["Feature"], df["Importance"], color=PALETTE[1], edgecolor="black", alpha=0.8)
    
    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{width:,.1f}", va="center", fontsize=9, color="#555")
    
    ax.set_title(f"Feature Importance (Top {top_n})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance (Gain)")
    ax.set_xlim(0, df["Importance"].max() * 1.15)
    
    save_plot(fig, "plot_feature_importance.png")


# ============================================================================
# 3. ACTUAL VS PREDICTED
# ============================================================================

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Scatter plot colored by absolute error magnitude."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter
    sc = ax.scatter(y_true, y_pred, c=errors, cmap="inferno_r", alpha=0.6, s=25, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Absolute Error ($)", rotation=270, labelpad=15)
    
    # Ideal line
    lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1.5, alpha=0.5, label="Ideal (y=x)")
    
    # Metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    stats = f"R²={r2:.4f}\nMAE=${mae:,.0f}\nRMSE=${rmse:,.0f}"
    add_stats_box(ax, stats, loc="upper left")
    
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc="lower right")
    
    save_plot(fig, "plot_actual_vs_predicted.png")


# ============================================================================
# 4. RESIDUAL DISTRIBUTION
# ============================================================================

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Histogram + KDE of residuals to check bias."""
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram + KDE
    ax.hist(residuals, bins=70, color=PALETTE[2], edgecolor="black", alpha=0.7, density=True)
    kde_x = np.linspace(residuals.min(), residuals.max(), 500)
    kde = gaussian_kde(residuals)
    ax.plot(kde_x, kde(kde_x), color="#333", linewidth=2, label="KDE")
    
    # Zero line
    ax.axvline(0, color="firebrick", linestyle="--", linewidth=2, label="Zero Bias")
    
    # Stats
    stats = f"Mean=${residuals.mean():>+,.0f}\nStd=${residuals.std():,.0f}\nSkew={pd.Series(residuals).skew():.2f}"
    add_stats_box(ax, stats, loc="upper left")
    
    ax.set_title(f"Residual Distribution — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Residual ($) [Actual - Predicted]")
    ax.set_ylabel("Density")
    ax.legend()
    
    save_plot(fig, "plot_residuals.png")


# ============================================================================
# 5. MODEL COMPARISON
# ============================================================================

def plot_model_comparison(results: Dict[str, Dict[str, float]], model_labels: Dict[str, str]) -> None:
    """Grouped bar chart: MAE/RMSE (left) + R² (right)."""
    names = [model_labels.get(k, k) for k in results]
    maes = [results[k]['MAE'] for k in results]
    rmses = [results[k]['RMSE'] for k in results]
    r2s = [results[k]['R2'] for k in results]
    
    x = np.arange(len(names))
    width = 0.3
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bars
    rects1 = ax1.bar(x - width/2, maes, width, label='MAE ($)', color=PALETTE[0], edgecolor='black', alpha=0.9)
    rects2 = ax1.bar(x + width/2, rmses, width, label='RMSE ($)', color=PALETTE[1], edgecolor='black', alpha=0.9)
    
    # R² line (right axis)
    ax2 = ax1.twinx()
    ax2.plot(x, r2s, color=PALETTE[3], marker='o', markersize=10, linewidth=2, label='R²')
    
    # Formatting
    ax1.set_ylabel('Error ($)', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12, color=PALETTE[3])
    ax2.tick_params(axis='y', labelcolor=PALETTE[3])
    ax2.set_ylim(0 if min(r2s) > 0 else min(r2s) - 0.1, 1.1)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11, fontweight='medium')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Value labels
    def add_labels(rects, ax):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'${h/1000:.0f}k', xy=(rect.get_x() + rect.get_width() / 2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    add_labels(rects1, ax1)
    add_labels(rects2, ax1)
    
    # R² labels
    for i, r2 in enumerate(r2s):
        ax2.annotate(f'{r2:.3f}', (x[i], r2), xytext=(0, 10),
                    textcoords="offset points", ha='center', color=PALETTE[3], fontweight='bold')
    
    ax1.set_title("Model Comparison Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    save_plot(fig, "plot_model_comparison.png")


# ============================================================================
# 6. RESIDUALS VS PREDICTED (NEW - Diagnostic Plot)
# ============================================================================

def plot_residuals_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Scatter: predicted vs residuals (check heteroscedasticity)."""
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color=PALETTE[2], edgecolors="none")
    ax.axhline(0, color="firebrick", linestyle="--", linewidth=2, label="Zero Line")
    
    # Add smoothed trend line to detect patterns
    from scipy.interpolate import UnivariateSpline
    sorted_idx = np.argsort(y_pred)
    spl = UnivariateSpline(y_pred[sorted_idx], residuals[sorted_idx], s=len(y_pred)*1000, k=3)
    y_smooth = spl(np.sort(y_pred))
    ax.plot(np.sort(y_pred), y_smooth, color="orange", linewidth=2, label="Trend")
    
    ax.set_title(f"Residuals vs Predicted — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Price ($)")
    ax.set_ylabel("Residual ($)")
    ax.legend()
    
    save_plot(fig, "plot_residuals_vs_predicted.png")


# ============================================================================
# 7. BUSINESS IMPACT SUMMARY (NEW - Executive Dashboard)
# ============================================================================

def plot_business_impact(baseline_mae: float, model_mae: float, model_name: str,
                        annual_transactions: int = 10_000) -> None:
    """Visual summary of business value: baseline vs ML model."""
    improvement = baseline_mae - model_mae
    improvement_pct = (improvement / baseline_mae) * 100
    annual_savings = improvement * annual_transactions
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Bar comparison
    methods = ['Current\n(Expert)', f'ML Model\n({model_name})']
    maes = [baseline_mae, model_mae]
    colors = ['#d62728', '#2ca02c']
    
    bars = ax1.bar(methods, maes, color=colors, edgecolor='black', alpha=0.8, width=0.5)
    ax1.set_ylabel('Mean Absolute Error ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Pricing Accuracy Comparison', fontsize=14, fontweight='bold')
    
    # Value labels
    for bar, mae in zip(bars, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, mae * 1.05,
                f'${mae:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Improvement arrow
    ax1.annotate('', xy=(1, model_mae), xytext=(0, baseline_mae),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax1.text(0.5, (baseline_mae + model_mae)/2, f'-{improvement_pct:.1f}%',
            ha='center', fontsize=14, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Right: Business metrics
    ax2.axis('off')
    metrics_text = (
        f"📊 BUSINESS IMPACT\n\n"
        f"Improvement per Transaction:\n"
        f"  ${improvement:,.0f} ({improvement_pct:.1f}%)\n\n"
        f"Annual Savings:\n"
        f"  ${annual_savings:,.0f}\n"
        f"  (Based on {annual_transactions:,} transactions/year)\n\n"
        f"Model Confidence:\n"
        f"  Typical error ±${model_mae:,.0f}"
    )
    ax2.text(0.5, 0.5, metrics_text, ha='center', va='center',
            fontsize=13, family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3, edgecolor='blue', lw=2))
    
    fig.suptitle('Business Value of ML Model', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_plot(fig, "plot_business_impact.png")


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def generate_all_plots(df_raw: pd.DataFrame, manager: Any, X_test: pd.DataFrame, y_test_orig: pd.Series) -> None:
    """Generate all 7 presentation plots."""
    print("\n" + "=" * 70)
    print(" GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Core plots
    plot_target_distribution(df_raw)
    
    importance_df = manager.get_feature_importance("lightgbm", top_n=15)
    if importance_df is not None:
        plot_feature_importance(importance_df)
    
    best_name, best_model = manager.get_best_model()
    best_label = {"ridge": "Ridge", "lightgbm": "LightGBM"}.get(best_name, best_name)
    
    y_pred = np.expm1(best_model.predict(X_test))
    y_true = y_test_orig.values.astype(float)
    
    plot_actual_vs_predicted(y_true, y_pred, best_label)
    plot_residuals(y_true, y_pred, best_label)
    plot_residuals_vs_predicted(y_true, y_pred, best_label)
    
    model_labels = {"ridge": "Ridge", "lightgbm": "LightGBM"}
    plot_model_comparison(manager.results, model_labels)
    
    # Business impact (estimate baseline)
    baseline_mae = np.median(y_true)  # Simple median predictor
    model_mae = manager.results[best_name]['MAE']
    plot_business_impact(baseline_mae, model_mae, best_label)
    
    print(f"\n[Visualization] 7 plots saved to {REPORTS_DIR}/")