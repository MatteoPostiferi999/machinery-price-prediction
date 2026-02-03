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


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("BLUE BOOK FOR BULLDOZERS - ML PIPELINE")
    print("="*80)
    
    # ── STEP 1: Load Data ─────────────────────────────────────────
    print("\n[STEP 1] LOADING DATA")
    print("-" * 80)
    data_file = TRAIN_DATA
    
    if not data_file.exists():
        print(f"❌ ERROR: Data file not found: {data_file}")
        sys.exit(1)
    
    df = load_data(data_file)
    
    # ── STEP 2: Data Preparation ─────────────────────────────────
    print("\n[STEP 2] DATA PREPARATION")
    print("-" * 80)
    data = prepare_data(df, random_state=RANDOM_STATE)
    
    # ── STEP 3: Train Models ──────────────────────────────────────
    print("\n[STEP 3] MODEL TRAINING")
    print("-" * 80)
    manager = train_models(data)
    
    # ── STEP 4: Final Summary ─────────────────────────────────────
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n📊 Results Summary:")
    print(f"   ✅ Preprocessed {data['X_train'].shape[0]:,} training samples")
    print(f"   ✅ Trained 2 models (Ridge, LightGBM)")
    print(f"   ✅ Best model saved to models/")
    
    best_name, _ = manager.get_best_model()
    best_metrics = manager.results[best_name]
    print(f"\n🏆 Best Model: {best_name.upper()}")
    print(f"   R²: {best_metrics['R2']:.4f}")
    print(f"   MAE: ${best_metrics['MAE']:,.0f}")
    print(f"   RMSE: ${best_metrics['RMSE']:,.0f}")
    
    print("\n📁 Next Steps:")
    print("   1. Check models/ for saved models")
    print("   2. Run evaluation on test set")
    print("   3. Generate visualizations")
    print("   4. Create final report")
    
    return manager, data


if __name__ == "__main__":
    manager, data = main()