"""
analyze_features.py
===================
Standalone script — run ONCE to understand your 7361 features.

Produces:
  exploration/feature_importance_lgbm.png   (top 30 by importance)
  exploration/feature_importance_bottom.png (bottom 30 — candidates to drop)
  exploration/cardinality_report.txt        (which categoricals exploded OHE)
  exploration/feature_action_plan.txt       (concrete drop / keep / replace list)

Usage:
  python analyze_features.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# ── Point to your project root ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DATA,
    TARGET_COLUMN,
    MODELS_DIR,
)
from src.preprocessing import load_data, engineer_features

# ============================================================================
# 1.  LOAD THE TRAINED LIGHTGBM MODEL  (already saved)
# ============================================================================

def load_best_model():
    """Load the saved LightGBM model."""
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run main.py first!"
        )
    model = joblib.load(model_path)
    print(f"[load] Loaded model: {type(model).__name__}")
    return model


# ============================================================================
# 2.  CARDINALITY ANALYSIS  (before encoding — on raw data)
# ============================================================================

def analyze_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every object/string column count unique values.
    This tells us exactly which columns exploded into thousands
    of OHE columns.
    """
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    rows = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        null_pct = df[col].isna().mean() * 100
        rows.append({
            "column": col,
            "n_unique": n_unique,
            "null_pct": round(null_pct, 2),
            "sample_values": df[col].dropna().unique()[:5].tolist(),
        })

    card_df = (
        pd.DataFrame(rows)
        .sort_values("n_unique", ascending=False)
        .reset_index(drop=True)
    )
    return card_df


# ============================================================================
# 3.  IMPORTANCE EXTRACTION + PLOTTING
# ============================================================================

def get_importances(model) -> pd.DataFrame:
    """Pull feature importances directly from the LightGBM model."""
    importances = model.feature_importances_
    names = model.feature_names_in_

    df = pd.DataFrame({
        "feature": names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Add cumulative importance (percentage)
    total = df["importance"].sum()
    df["importance_pct"] = (df["importance"] / total * 100).round(2)
    df["cumulative_pct"] = df["importance_pct"].cumsum().round(2)

    return df


def plot_top_bottom(imp_df: pd.DataFrame, out_dir: Path):
    """Save top-30 and bottom-30 importance bar charts."""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── TOP 30 ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    top30 = imp_df.head(30).copy()
    sns.barplot(
        data=top30,
        x="importance",
        y="feature",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Top 30 Features — LightGBM Importance", fontsize=16, pad=15)
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(out_dir / "feature_importance_lgbm.png", dpi=150)
    plt.close(fig)
    print(f"[plot] Saved feature_importance_lgbm.png")

    # ── BOTTOM 30 ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    bottom30 = imp_df.tail(30).copy().sort_values("importance", ascending=True)
    sns.barplot(
        data=bottom30,
        x="importance",
        y="feature",
        palette="magma",
        ax=ax,
    )
    ax.set_title("Bottom 30 Features — Candidates to DROP", fontsize=16, pad=15)
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(out_dir / "feature_importance_bottom.png", dpi=150)
    plt.close(fig)
    print(f"[plot] Saved feature_importance_bottom.png")


# ============================================================================
# 4.  BUILD THE ACTION PLAN
# ============================================================================

def build_action_plan(imp_df: pd.DataFrame, card_df: pd.DataFrame) -> str:
    """
    Produce a human-readable action plan.

    Rules (engineering judgment):
      - Features with importance == 0          → DROP (zero signal)
      - Features with importance in bottom 5%  → DROP (noise)
      - OHE columns from high-cardinality cols → REPLACE with ordinal / target encoding
      - Top-50 features                        → KEEP
    """
    total = imp_df["importance"].sum()

    # ── zero-importance features ──────────────────────────────
    zero = imp_df[imp_df["importance"] == 0]

    # ── bottom 5 % threshold ──────────────────────────────────
    threshold_5pct = imp_df["importance"].quantile(0.05)
    bottom_5pct = imp_df[imp_df["importance"] <= threshold_5pct]

    # ── high-cardinality source columns (>50 unique) ──────────
    high_card = card_df[card_df["n_unique"] > 50]

    # ── build text report ─────────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append("FEATURE ACTION PLAN")
    lines.append("=" * 80)

    # --- summary stats ---
    lines.append(f"\nTotal features after OHE : {len(imp_df):,}")
    lines.append(f"Features with importance 0: {len(zero):,}")
    lines.append(f"Features in bottom 5 %    : {len(bottom_5pct):,}")
    lines.append(f"Top-50 cumulative importance: {imp_df.head(50)['importance_pct'].sum():.1f} %\n")

    # --- high-cardinality categoricals ---
    lines.append("-" * 80)
    lines.append("HIGH-CARDINALITY CATEGORICALS  →  replace OHE with target/ordinal encoding")
    lines.append("-" * 80)
    for _, row in high_card.iterrows():
        lines.append(
            f"  {row['column']:40s}  unique={row['n_unique']:>6,}   "
            f"nulls={row['null_pct']:>5.1f} %   samples={row['sample_values']}"
        )

    # --- drop list ---
    lines.append("\n" + "-" * 80)
    lines.append("FEATURES TO DROP  (importance == 0 or bottom 5 %)")
    lines.append("-" * 80)
    # show first 40 for readability
    for _, row in bottom_5pct.head(40).iterrows():
        lines.append(f"  {row['feature']:60s}  imp={row['importance']}")
    if len(bottom_5pct) > 40:
        lines.append(f"  ... and {len(bottom_5pct) - 40} more (see full CSV)")

    # --- keep list (top 50) ---
    lines.append("\n" + "-" * 80)
    lines.append("FEATURES TO KEEP  (top 50 — drive 90 %+ of model performance)")
    lines.append("-" * 80)
    for _, row in imp_df.head(50).iterrows():
        lines.append(
            f"  {row['feature']:60s}  imp_pct={row['importance_pct']:>6.2f} %   "
            f"cum={row['cumulative_pct']:>6.1f} %"
        )

    lines.append("\n" + "=" * 80)
    lines.append("EXPECTED OUTCOME AFTER CLEANUP")
    lines.append("=" * 80)
    lines.append(f"  Current features : {len(imp_df):,}")
    estimated_drop = len(zero) + (
        high_card["n_unique"].sum() if len(high_card) > 0 else 0
    )
    lines.append(f"  Features to drop : ~{len(zero):,} (zero importance)")
    lines.append(f"  High-card cols   : {len(high_card)} columns → replace with 1 encoded col each")
    lines.append(f"  Estimated final  : ~50–200 features")
    lines.append(f"\n  This should:")
    lines.append(f"    • Cut training time by 60-80 %")
    lines.append(f"    • Maintain or IMPROVE R² (less noise)")
    lines.append(f"    • Make feature importance chart readable")

    return "\n".join(lines)


# ============================================================================
# 5.  MAIN
# ============================================================================

def main():
    out_dir = PROJECT_ROOT / "exploration"
    out_dir.mkdir(exist_ok=True)

    # ── load raw data (pre-encoding) to get cardinality ──────
    print("\n[1/4] Loading raw data for cardinality analysis...")
    data_file = TRAIN_DATA
    df_raw = load_data(data_file)
    df_raw = engineer_features(df_raw)   # same features as pipeline, but NO encoding yet
    card_df = analyze_cardinality(df_raw)
    print(card_df[["column", "n_unique", "null_pct"]].to_string(index=False))

    # ── load trained model ────────────────────────────────────
    print("\n[2/4] Loading trained LightGBM model...")
    model = load_best_model()

    # ── extract importances ───────────────────────────────────
    print("\n[3/4] Extracting feature importances...")
    imp_df = get_importances(model)
    imp_df.to_csv(out_dir / "feature_importances_full.csv", index=False)
    print(f"  → Saved full importance table ({len(imp_df)} features)")

    # ── plots ─────────────────────────────────────────────────
    print("\n[4/4] Generating plots and action plan...")
    plot_top_bottom(imp_df, out_dir)

    # ── action plan ───────────────────────────────────────────
    plan_text = build_action_plan(imp_df, card_df)
    print(plan_text)

    (out_dir / "feature_action_plan.txt").write_text(plan_text)
    print(f"\n[done] All outputs in exploration/")


if __name__ == "__main__":
    main()