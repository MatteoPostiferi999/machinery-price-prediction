# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python ML pipeline for regression tasks with hyperparameter tuning, model comparison, and visualization. Uses scikit-learn models (Random Forest, Gradient Boosting) with a modular architecture.

## Commands

```bash
# Setup environment (creates venv at ~/ML)
bash setup_env.sh
source ~/ML/bin/activate

# Run full pipeline
python main.py
```

No test framework or linting is configured.

## Architecture

**Pipeline flow** (main.py):
1. Load CSV → 2. EDA plots → 3. Train/val/test split → 4. Preprocess → 5. Train & tune → 6. Evaluate → 7. Generate plots

**Key modules:**
- `config.py` - Central configuration (paths, hyperparameters, split ratios, target column)
- `preprocessing.py` - `Preprocessor` class with fit/transform pattern (imputation, encoding, scaling)
- `model.py` - `ModelTrainer` class with model registry; uses `RandomizedSearchCV` for tuning
- `evaluation.py` - Computes MAE, RMSE, R², MAPE; returns comparison DataFrames
- `visualization.py` - EDA and performance plots saved to `report/`

**Configuration-driven:** All hyperparameters, paths, and split ratios are in `config.py`. Models are registered in `MODEL_REGISTRY` in model.py.

**Data expectations:**
- Input: `data/dataset.csv` with a column named by `TARGET_COLUMN` in config.py (default: "target")
- Auto-detects numeric vs categorical columns
- Handles missing values (median/mode imputation)

**Outputs:**
- Trained models saved as `.joblib` files in project root
- Plots saved to `report/` directory as PNG files
