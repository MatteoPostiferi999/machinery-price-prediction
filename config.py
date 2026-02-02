"""
config.py
---------
Centralized configuration: file paths, train/val/test split ratios,
random seeds, and model hyperparameters. Edit this file to change
any project-wide setting without touching other modules.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_DIR = os.path.join(BASE_DIR, "report")

DATA_FILE = os.path.join(DATA_DIR, "dataset.csv")  # <-- change filename here

# ─── Reproducibility ──────────────────────────────────────────────
RANDOM_STATE = 42

# ─── Split ratios ─────────────────────────────────────────────────
TEST_SIZE = 0.2       # fraction reserved for final evaluation
VAL_SIZE = 0.25       # fraction of the remaining data used for validation
                      # effective val size on full data = (1 - TEST_SIZE) * VAL_SIZE

# ─── Target ───────────────────────────────────────────────────────
TARGET_COLUMN = "target"  # <-- name of the column to predict

# ─── Hyperparameters ──────────────────────────────────────────────
# Edit or extend per model as needed.
HYPERPARAMS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
    },
    "gradient_boosting": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_samples_leaf": 10,
    },
}

# ─── Tuning ───────────────────────────────────────────────────────
PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_leaf": [3, 5, 10],
    },
    "gradient_boosting": {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
    },
}

TUNING_CV_FOLDS = 3
TUNING_N_ITER = 15  # used only for RandomizedSearchCV