"""
main.py
=======
Entry point for the SHM machinery price prediction pipeline.

Run:
    python main.py

Pipeline stages:
    1. Load data
    2. Preprocessing (engineer → split → encode)
    3. Model training (Ridge + LightGBM)
    4. Evaluation (validation + test)
    5. Visualizations
    6. Summary
"""

import sys
from pathlib import Path

from src.config import TRAIN_DATA, RANDOM_STATE, MODELS
from src.preprocessing import load_data, prepare_data
from src.model import train_models
from src.evaluation import ModelEvaluator
from src.visualization import generate_all_plots


def main():
    """Run the full pipeline end-to-end."""

    print("\n" + "=" * 70)
    print(" SHM MACHINERY PRICE PREDICTION — ML PIPELINE")
    print("=" * 70)

    # ── STEP 1: Load Data ───────────────────────────────────────────────
    print("\n[STEP 1/6] Loading data...")
    if not TRAIN_DATA.exists():
        print(f"[ERROR] Data file not found: {TRAIN_DATA}")
        sys.exit(1)
    df = load_data(TRAIN_DATA)

    # ── STEP 2: Preprocessing ───────────────────────────────────────────
    print("\n[STEP 2/6] Preprocessing...")
    data = prepare_data(df, random_state=RANDOM_STATE)

    # ── STEP 3: Model Training ──────────────────────────────────────────
    print("\n[STEP 3/6] Training models...")
    manager = train_models(data)

    # ── STEP 4: Evaluation ──────────────────────────────────────────────
    print("\n[STEP 4/6] Evaluating...")
    evaluator = ModelEvaluator(feature_names=data['X_val'].columns.tolist())

    # Validation — model selection
    val_comparison = evaluator.compare_all(
        manager.models, data['X_val'], data['y_val_orig']
    )
    evaluator.save_results(val_comparison, filename="val_evaluation.csv")

    # Residual analysis on best model
    _, best_model = manager.get_best_model()
    evaluator.residual_summary(best_model, data['X_val'], data['y_val_orig'])

    # Test — final evaluation (unseen data)
    test_comparison = evaluator.compare_all(
        manager.models, data['X_test'], data['y_test_orig']
    )
    evaluator.save_results(test_comparison, filename="test_evaluation.csv")

    # ── STEP 5: Visualizations ──────────────────────────────────────────
    print("\n[STEP 5/6] Generating plots...")
    generate_all_plots(
        df_raw=df,
        manager=manager,
        X_test=data['X_test'],
        y_test_orig=data['y_test_orig'],
    )

    # ── STEP 6: Summary ─────────────────────────────────────────────────
    best_name, _ = manager.get_best_model()
    best_label   = MODELS[best_name]['name']

    best_val  = val_comparison[val_comparison['Model'] == best_name].iloc[0]
    best_test = test_comparison[test_comparison['Model'] == best_name].iloc[0]

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Best model : {best_label}")
    print(f"  Features   : {data['X_train'].shape[1]}")
    print(f"  Train size : {data['X_train'].shape[0]:,} samples")
    print()
    print(f"  {'Metric':<6} | {'Validation':>12} | {'Test':>12}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'R2':<6} | {best_val['R2']:>12.4f} | {best_test['R2']:>12.4f}")
    print(f"  {'MAE':<6} | ${best_val['MAE']:>10,.0f} | ${best_test['MAE']:>10,.0f}")
    print(f"  {'RMSE':<6} | ${best_val['RMSE']:>10,.0f} | ${best_test['RMSE']:>10,.0f}")
    print(f"  {'MAPE':<6} | {best_val['MAPE']:>11.2f}% | {best_test['MAPE']:>11.2f}%")
    print()

    return manager, data


if __name__ == "__main__":
    main()