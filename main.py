"""
main.py
=======
Entry point for the SHM machinery price prediction pipeline.

Pipeline stages:
    1. Load and engineer features
    2. Split data (70/15/15)
    3. Preprocessing (fit on train, transform all)
    4. Model training (Ridge + LightGBM)
    5. Evaluation (validation + test)
    6. Visualizations
    7. Summary & Artifacts
"""

import sys
import joblib
import numpy as np
from pathlib import Path

# Import custom modules
from src.config import TRAIN_DATA, RANDOM_STATE, MODELS, MODELS_DIR
from src.data import load_raw_data, load_and_split
from src.preprocessing import build_preprocessor
from src.model import train_models
from src.evaluation import ModelEvaluator
from src.visualization import generate_all_plots


def main():
    """Run the full pipeline end-to-end."""

    print("\n" + "=" * 70)
    print(" SHM MACHINERY PRICE PREDICTION — ML PIPELINE")
    print("=" * 70)

    try:
        # ── STEP 1: Load and Split Data ─────────────────────────────────────
        print("\n[STEP 1/6] Loading and splitting data...")
        if not TRAIN_DATA.exists():
            print(f"[ERROR] Data file not found: {TRAIN_DATA}")
            sys.exit(1)
        
        # Load raw data for visualizations (keep original)
        df_raw = load_raw_data(TRAIN_DATA)
        
        # Load, engineer, and split
        splits = load_and_split(TRAIN_DATA, return_separated=False)
        train_df = splits['train']
        val_df = splits['val']
        test_df = splits['test']

        # ── STEP 2: Preprocessing ───────────────────────────────────────────
        print("\n[STEP 2/6] Preprocessing...")
        
        # Build preprocessor
        preprocessor = build_preprocessor()
        
        # Fit on train, transform all splits
        train_processed = preprocessor.fit_transform(train_df, is_train=True)
        val_processed = preprocessor.transform(val_df, is_train=False)
        test_processed = preprocessor.transform(test_df, is_train=False)
        
        # Extract features and target
        from src.config import TARGET_COLUMN, TARGET_LOG
        
        # TRAIN: TARGET_LOG is created by preprocessor (is_train=True)
        X_train = train_processed.drop(columns=[TARGET_COLUMN, TARGET_LOG], errors='ignore')
        y_train = train_processed[TARGET_LOG]
        y_train_orig = train_df[TARGET_COLUMN]
        
        # VAL: TARGET_LOG not created by preprocessor (is_train=False)
        # We create it manually from original dataframe
        X_val = val_processed.drop(columns=[TARGET_COLUMN], errors='ignore')
        y_val = np.log1p(val_df[TARGET_COLUMN])  # Manual log-transform
        y_val_orig = val_df[TARGET_COLUMN]
        
        # TEST: Same as val
        X_test = test_processed.drop(columns=[TARGET_COLUMN], errors='ignore')
        y_test = np.log1p(test_df[TARGET_COLUMN])  # Manual log-transform
        y_test_orig = test_df[TARGET_COLUMN]
        
        # Assemble data dict (compatible with model.py)
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'y_train_orig': y_train_orig,
            'y_val_orig': y_val_orig,
            'y_test_orig': y_test_orig,
            'preprocessor': preprocessor
        }

        # ── STEP 3: Model Training ──────────────────────────────────────────
        print("\n[STEP 3/6] Training models...")
        manager = train_models(data)

        # ── STEP 4: Evaluation ──────────────────────────────────────────────
        print("\n[STEP 4/6] Evaluating...")
        evaluator = ModelEvaluator(feature_names=X_val.columns.tolist())

        # 4a. Validation evaluation
        val_comparison = evaluator.compare_all(
            manager.models, X_val, y_val_orig
        )
        evaluator.save_results(val_comparison, filename="val_evaluation.csv")

        # 4b. Best model residual analysis
        best_name, best_model = manager.get_best_model()
        print(f"\n[Deep Dive] Analyzing residuals for: {MODELS[best_name]['name']}")
        evaluator.residual_summary(best_model, X_val, y_val_orig)

        # 4c. Test evaluation (final unseen data)
        test_comparison = evaluator.compare_all(
            manager.models, X_test, y_test_orig
        )
        evaluator.save_results(test_comparison, filename="test_evaluation.csv")

        # ── STEP 5: Visualizations ──────────────────────────────────────────
        print("\n[STEP 5/6] Generating plots...")
        generate_all_plots(
            df_raw=df_raw,
            manager=manager,
            X_test=X_test,
            y_test_orig=y_test_orig,
        )

        # ── STEP 6: Final Summary ───────────────────────────────────────────
        best_val = val_comparison[val_comparison['Model'] == best_name].iloc[0]
        best_test = test_comparison[test_comparison['Model'] == best_name].iloc[0]
        best_label = MODELS[best_name]['name']

        print("\n" + "=" * 70)
        print(" PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\n  Best model : {best_label}")
        print(f"  Features   : {X_train.shape[1]}")
        print(f"  Train size : {X_train.shape[0]:,} samples")
        print()
        print(f"  {'Metric':<6} | {'Validation':>12} | {'Test':>12}")
        print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}")
        print(f"  {'R2':<6} | {best_val['R2']:>12.4f} | {best_test['R2']:>12.4f}")
        print(f"  {'MAE':<6} | ${best_val['MAE']:>10,.0f} | ${best_test['MAE']:>10,.0f}")
        print(f"  {'RMSE':<6} | ${best_val['RMSE']:>10,.0f} | ${best_test['RMSE']:>10,.0f}")
        print(f"  {'MAPE':<6} | {best_val['MAPE']:>11.2f}% | {best_test['MAPE']:>11.2f}%")
        print()

        # Save preprocessor for future inference
        print("[Artifacts] Saving fitted preprocessor...")
        preprocessor_path = MODELS_DIR / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
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