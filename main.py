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
    6. Summary & Artifact Persistence
"""

import sys
import joblib
from pathlib import Path

# Import custom modules
from src.config import TRAIN_DATA, RANDOM_STATE, MODELS, MODELS_DIR
from src.preprocessing import load_data, prepare_data
from src.model import train_models
from src.evaluation import ModelEvaluator
from src.visualization import generate_all_plots


def main():
    """Run the full pipeline end-to-end."""

    print("\n" + "=" * 70)
    print(" SHM MACHINERY PRICE PREDICTION — ML PIPELINE")
    print("=" * 70)

    try:
        # ── STEP 1: Load Data ───────────────────────────────────────────────
        print("\n[STEP 1/6] Loading data...")
        if not TRAIN_DATA.exists():
            print(f"[ERROR] Data file not found: {TRAIN_DATA}")
            print(f"Please ensure the data is placed in 'data/train_full.csv'.")
            sys.exit(1)
            
        df = load_data(TRAIN_DATA)

        # ── STEP 2: Preprocessing ───────────────────────────────────────────
        print("\n[STEP 2/6] Preprocessing...")
        # Note: This returns a dictionary containing split data AND the fitted preprocessor
        data = prepare_data(df, random_state=RANDOM_STATE)

        # ── STEP 3: Model Training ──────────────────────────────────────────
        print("\n[STEP 3/6] Training models...")
        # ModelManager handles training, comparison, and saving the best model artifact
        manager = train_models(data)

        # ── STEP 4: Evaluation ──────────────────────────────────────────────
        print("\n[STEP 4/6] Evaluating...")
        # Initialize evaluator with feature names for deep-dive error analysis
        evaluator = ModelEvaluator(feature_names=data['X_val'].columns.tolist())

        # 4a. Validation — Model Selection
        val_comparison = evaluator.compare_all(
            manager.models, data['X_val'], data['y_val_orig']
        )
        evaluator.save_results(val_comparison, filename="val_evaluation.csv")

        # 4b. Deep Dive — Residuals of the Champion Model
        best_name, best_model = manager.get_best_model()
        print(f"\n[Deep Dive] Analyzing residuals for best model: {MODELS[best_name]['name']}")
        evaluator.residual_summary(best_model, data['X_val'], data['y_val_orig'])

        # 4c. Test — Final Performance on Unseen Data
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

        # ── STEP 6: Summary & Artifact Persistence ──────────────────────────
        # Retrieve best metrics for final report
        best_val  = val_comparison[val_comparison['Model'] == best_name].iloc[0]
        best_test = test_comparison[test_comparison['Model'] == best_name].iloc[0]
        best_label = MODELS[best_name]['name']

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

        # CRITICAL: Save the preprocessor to allow future inference
        print("[Artifacts] Saving fitted preprocessor...")
        preprocessor_path = MODELS_DIR / "preprocessor.pkl"
        joblib.dump(data['preprocessor'], preprocessor_path)
        print(f"  -> Saved to {preprocessor_path}")
        print(f"  -> System ready for inference (load preprocessor + best_model).")
        print()

        return manager, data

    except Exception as e:
        print("\n" + "!" * 70)
        print(f" [CRITICAL ERROR] Pipeline failed: {e}")
        print("!" * 70 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()