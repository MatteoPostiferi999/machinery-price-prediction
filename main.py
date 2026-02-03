"""
main.py
=======
Orchestrator for the Blue Book for Bulldozers ML pipeline.
Runs the complete workflow from data loading to model evaluation.
"""

import sys
from pathlib import Path

from src.config import TRAIN_DATA, RANDOM_STATE
from src.preprocessing import load_data, prepare_data
from src.model import train_models
from src.evaluation import ModelEvaluator
import pandas as pd
from deep_forensic import deep_forensic_analysis # Assicurati che il file sia nella stessa cartella o src



def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("BLUE BOOK FOR BULLDOZERS - ML PIPELINE")
    print("=" * 80)

    # ── STEP 1: Load Data ─────────────────────────────────────────
    print("\n[STEP 1] LOADING DATA")
    print("-" * 80)

    if not TRAIN_DATA.exists():
        print(f"❌ ERROR: Data file not found: {TRAIN_DATA}")
        sys.exit(1)

    df = load_data(TRAIN_DATA)

    # ── STEP 2: Data Preparation ─────────────────────────────────
    print("\n[STEP 2] DATA PREPARATION")
    print("-" * 80)
    data = prepare_data(df, random_state=RANDOM_STATE)

    # ── STEP 3: Model Training ────────────────────────────────────
    print("\n[STEP 3] MODEL TRAINING")
    print("-" * 80)
    manager = train_models(data)

    # ── STEP 3.5: Feature Importance ──────────────────────────────
    print("\n[STEP 3.5] FEATURE IMPORTANCE ANALYSIS")
    print("-" * 80)

    best_name, _ = manager.get_best_model()
    importance_df = manager.get_feature_importance(best_name, top_n=20)

    if importance_df is not None:
        print(f"\n[ANALYSIS] Top 20 features for {best_name.upper()}:")
        print(importance_df.to_string(index=False))

        report_path = Path("reports") / "feature_importance.csv"
        importance_df.to_csv(report_path, index=False)
        print(f"\n💾 Feature importance saved to: {report_path}")
    else:
        print("Feature importance not available for this model.")

    # ── STEP 4: Validation Evaluation ─────────────────────────────
    print("\n[STEP 4] VALIDATION EVALUATION")
    print("-" * 80)

    # Build {name: model} dict from manager.results
    models = manager.models
    evaluator = ModelEvaluator(feature_names=data['X_val'].columns.tolist())

    # 4a — Compare all models side by side on validation set
    print("\n📊 VALIDATION SET")
    val_comparison = evaluator.compare_all(models, data['X_val'], data['y_val_orig'])

    # 4b — Residual analysis: where does the best model struggle?
    _, best_model = manager.get_best_model()
    evaluator.residual_summary(best_model, data['X_val'], data['y_val_orig'])

    # 4c — Worst predictions: inspect the biggest errors
    evaluator.worst_predictions(best_model, data['X_val'], data['y_val_orig'], n=20)

    # 4d — Persist validation results
    evaluator.save_results(val_comparison, filename="val_evaluation.csv")

    # ── STEP 5: Test Set Evaluation (final, unseen data) ─────────
    print("\n[STEP 5] TEST SET EVALUATION")
    print("-" * 80)

    test_comparison = evaluator.compare_all(models, data['X_test'], data['y_test_orig'])
    evaluator.save_results(test_comparison, filename="test_evaluation.csv")

    # PROVAAAAAAAAAAAAAAAAA
    # ── STEP 6: Forensic Analysis ─────────────────────────────────
    print("\n[STEP 6] FORENSIC ANALYSIS")
    print("-" * 80)
    
    
    #best_name, _ = manager.get_best_model()  
    best_model = manager.models[best_name]
    # Passiamo 'df' che è il dataset originale caricato allo Step 1
    # Nota: 'df' deve essere ancora disponibile in memoria nel main
    manager.analyze_worst_errors(
        model=best_model, 
        X_val=data['X_val'], 
        y_val_orig=data['y_val_orig'],
        df_raw=df,  # <--- ECCO IL TRUCCO: Passiamo il dataset originale
        top_n=200
    )

    # ── STEP 7: Deep Forensic (Opzionale) ─────────────────────────
        # Carichiamo i moduli necessari
        
        # Carica gli ID dei cattivi dal file appena creato
    try:
        worst_df_loaded = pd.read_csv("reports/worst_errors_forensic.csv")
        worst_ids_list = worst_df_loaded['Sales ID'].tolist()
            
            # Lancia l'analisi comparativa
        deep_forensic_analysis(df, worst_ids_list)
            
    except FileNotFoundError:
            print("⚠️ File worst_errors_forensic.csv non trovato. Salta questo step.")
            
    # ── FINAL SUMMARY ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)

    print(f"\n📊 Results Summary:")
    print(f"   ✅ Preprocessed {data['X_train'].shape[0]:,} training samples")
    print(f"   ✅ Trained {len(models)} models ({', '.join(models.keys())})")
    print(f"   ✅ Best model saved to models/")
    print(f"   ✅ Evaluation results saved to reports/")

    # Pull test metrics for best model
    best_test_row = test_comparison[test_comparison['Model'] == best_name].iloc[0]

    # Pull validation metrics for best model
    best_val_row = val_comparison[val_comparison['Model'] == best_name].iloc[0]

    print(f"\n🏆 Best Model: {best_name.upper()}")

    print("\n📊 PERFORMANCE COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<10} | {'VALIDATION':>15} | {'TEST':>15}")
    print("-" * 80)
    print(f"{'R²':<10} | {best_val_row['R2']:>15.4f} | {best_test_row['R2']:>15.4f}")
    print(f"{'MAE':<10} | ${best_val_row['MAE']:>14,.0f} | ${best_test_row['MAE']:>14,.0f}")
    print(f"{'RMSE':<10} | ${best_val_row['RMSE']:>14,.0f} | ${best_test_row['RMSE']:>14,.0f}")
    print(f"{'MAPE':<10} | {best_val_row['MAPE']:>14.2f}% | {best_test_row['MAPE']:>14.2f}%")

    print("\n📁 Next Steps:")
    print("   1. Generate visualizations (visualization.py)")
    print("   2. Create final report (report/report.md)")

    return manager, data


if __name__ == "__main__":
    manager, data = main()