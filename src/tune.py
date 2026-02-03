"""
tune.py
=======
Automated Hyperparameter Tuning using Optuna.

Performs Bayesian Optimization to find the optimal set of hyperparameters
for LightGBM. Uses pruning to kill unpromising trials early.

Usage:
    python -m src.tune
"""

import optuna
import lightgbm as lgb
import numpy as np
import joblib
import sys
from pathlib import Path
from optuna.integration import LightGBMPruningCallback

# Import internal modules
# Note: We use relative imports, so this script must be run as a module
from .config import TRAIN_DATA, RANDOM_STATE, MODELS_DIR
from .preprocessing import load_data, prepare_data

# Set logging to see progress clearly
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial: optuna.Trial, data: dict) -> float:
    """
    Optuna objective function.
    1. Suggests hyperparameters.
    2. Trains LightGBM model.
    3. Returns RMSE (to be minimized).
    """
    
    # ── 1. Define Search Space ────────────────────────────────────────────
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 2000,  # Allow many trees, Early Stopping will cut it short
        
        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # ── 2. Prepare Data for LightGBM ──────────────────────────────────────
    dtrain = lgb.Dataset(data['X_train'], label=data['y_train'])
    dval   = lgb.Dataset(data['X_val'],   label=data['y_val'], reference=dtrain)

    # ── 3. Train with Pruning ─────────────────────────────────────────────
    # This callback kills the trial if it looks bad compared to others
    pruning_callback = LightGBMPruningCallback(trial, "rmse")

    model = lgb.train(
        param,
        dtrain,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            pruning_callback
        ]
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    # Predict on validation set
    preds = model.predict(data['X_val'])
    rmse = np.sqrt(np.mean((data['y_val'] - preds) ** 2))
    
    return rmse

def run_tuning(n_trials=50):
    print("\n" + "="*70)
    print(" 🚀 STARTING HYPERPARAMETER TUNING (OPTUNA)")
    print("="*70)

    # 1. Load Data
    print("[tune] Loading and preparing data...")
    if not TRAIN_DATA.exists():
        print(f"[ERROR] Data not found: {TRAIN_DATA}")
        return
    
    df_raw = load_data(TRAIN_DATA)
    data = prepare_data(df_raw, random_state=RANDOM_STATE)

    # 2. Create Study
    print(f"\n[tune] Running {n_trials} trials. Please wait...")
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    try:
        study.optimize(lambda trial: objective(trial, data), n_trials=n_trials)
    except KeyboardInterrupt:
        print("\n[tune] Tuning interrupted by user. Saving current best results...")

    # 3. Show Results
    print("\n" + "="*70)
    print(" TUNING COMPLETE")
    print("="*70)
    print(f"Best RMSE: {study.best_value:.4f}")
    print("Best Params (COPY THESE TO CONFIG.PY):")
    print("-" * 30)
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
    print("-" * 30)
    
    # 4. Save to file
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "best_params_optuna.pkl"
    joblib.dump(study.best_params, out_path)
    print(f"\n[tune] Parameters backed up to: {out_path}")

if __name__ == "__main__":
    run_tuning(n_trials=50)