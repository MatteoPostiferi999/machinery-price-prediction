"""
main.py
-------
Entry point. Run the entire ML pipeline end-to-end:
    python main.py

Pipeline steps (in order):
  1. Load data
  2. EDA visualizations
  3. Train / val / test split
  4. Preprocessing (fit on train, transform all splits)
  5. Train & tune models
  6. Evaluate on validation set
  7. Evaluate best model on held-out test set
  8. Save performance plots
  9. Print final summary
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_FILE,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
)
from preprocessing import load_data, Preprocessor
from model import ModelTrainer
from evaluation import evaluate_all
from visualization import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_feature_distributions,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_model_comparison,
)


def main():
    # ── 1. Load ───────────────────────────────────────────────────
    df = load_data(DATA_FILE)

    # ── 2. EDA plots (on raw data) ────────────────────────────────
    print("\n[main] Generating EDA visualizations …")
    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_missing_values(df)
    plot_feature_distributions(df)

    # ── 3. Split ──────────────────────────────────────────────────
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n[main] Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── 4. Preprocess ─────────────────────────────────────────────
    print("\n[main] Preprocessing …")
    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_val   = preprocessor.transform(X_val)
    X_test  = preprocessor.transform(X_test)

    # ── 5. Train ──────────────────────────────────────────────────
    print("\n[main] Training models …")
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)
    trainer.save_all()

    # ── 6. Validate ───────────────────────────────────────────────
    print("\n[main] Evaluating on validation set …")
    val_summary = evaluate_all(trainer, X_val, y_val.values, split_label="Validation")

    # ── 7. Test (best model only) ─────────────────────────────────
    best_model_name = val_summary.iloc[0]["model"]
    print(f"\n[main] Evaluating best model ({best_model_name}) on test set …")
    test_summary = evaluate_all(trainer, X_test, y_test.values, split_label="Test")

    # ── 8. Performance plots ──────────────────────────────────────
    print("\n[main] Generating performance visualizations …")
    for name in trainer.models:
        y_pred = trainer.predict(name, X_val)
        plot_actual_vs_predicted(y_val.values, y_pred, name)
        plot_residuals(y_val.values, y_pred, name)
    plot_model_comparison(val_summary)

    # ── 9. Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"\n  Best model (validation): {best_model_name}")
    best_test = test_summary[test_summary["model"] == best_model_name].iloc[0]
    print(f"  Test RMSE : {best_test['RMSE']:.4f}")
    print(f"  Test MAE  : {best_test['MAE']:.4f}")
    print(f"  Test R²   : {best_test['R²']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()