"""
main.py
=======
Orchestrator for the Blue Book for Bulldozers ML pipeline.
Runs the complete workflow from data loading to model evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


from src.config import TRAIN_DATA, SAMPLE_DATA, RANDOM_STATE
from src.preprocessing import load_data, prepare_data
# from src.model import train_models  # When ready
# from src.evaluation import evaluate_model  # When ready


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("BLUE BOOK FOR BULLDOZERS - ML PIPELINE")
    print("="*80)
    
    # Load data
    print("\n[STEP 1] LOADING DATA")
    data_file = SAMPLE_DATA if SAMPLE_DATA.exists() else TRAIN_DATA
    df = load_data(data_file)
    
    # Prepare data
    print("\n[STEP 2] DATA PREPARATION")
    data = prepare_data(df, random_state=RANDOM_STATE)
    
    # Summary
    print("\n[STEP 3] PIPELINE SUMMARY")
    print(f"✅ Train: {data['X_train'].shape}")
    print(f"✅ Val:   {data['X_val'].shape}")
    print(f"✅ Test:  {data['X_test'].shape}")
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    
    return data


if __name__ == "__main__":
    data = main()