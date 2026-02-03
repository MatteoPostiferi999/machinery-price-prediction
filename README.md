# Machinery Price Prediction - ML Pipeline

**Candidate:** Matteo Postiferi  

---

## How to Run the Program

### Step 1: Prerequisites

Ensure you have **Python 3.11+** installed:
```bash
python --version  # Should show Python 3.11 or higher
```

### Step 2: Install Dependencies

Navigate to the project folder and install required packages:
```bash
cd Bit_Assessment
pip install -r requirements.txt
```

**Verify installation:**
```bash
pip list | grep -E "lightgbm|pandas|scikit-learn"
```

You should see:
- `lightgbm` (4.6.0 or higher)
- `pandas` (3.0.0 or higher)
- `scikit-learn` (1.8.0 or higher)

### Step 3: Prepare Dataset

**Place the dataset file:**

The dataset should be placed in the `data/` directory with the filename `train_full.csv`.
```bash
# If data/ directory doesn't exist, create it:
mkdir -p data

# Rename and move your dataset to the correct location:
# (the file provided should be renamed to train_full.csv)
mv your_dataset_file.csv data/train_full.csv

# Verify the file is in place:
ls -lh data/train_full.csv
# Should show: train_full.csv (~114 MB)
```

**Note:** The dataset was provided as part of this assignment and is not included in the ZIP due to file size limitations.

### Step 4: Run the Pipeline

Execute the complete ML pipeline:
```bash
python main.py
```

**Expected runtime:** ~2 minutes (depending on your hardware)

**Expected console output:**
```
======================================================================
 SHM MACHINERY PRICE PREDICTION — ML PIPELINE
======================================================================

[STEP 1/6] Loading and splitting data...
[Data] Loaded 412,698 rows
[Data] Train=288,888, Val=61,905, Test=61,905

[STEP 2/6] Preprocessing...
[Preprocessor] Output shape: (288888, 75)

[STEP 3/6] Training models...
[LIGHTGBM] Training...
[LIGHTGBM] Complete in 69.94s | Best: 3726

[STEP 4/6] Evaluating...
✅ Test R² = 0.915, MAE = $4,156

[STEP 5/6] Generating plots...
✅ 7 plots saved to reports/

[STEP 6/6] Pipeline complete!
```

### Step 5: View Results

**Generated visualizations** (in `reports/`):
- `plot_target_distribution.png` - Price distribution analysis
- `plot_feature_importance.png` - Top 15 predictive features
- `plot_actual_vs_predicted.png` - Model accuracy scatter plot
- `plot_residuals.png` - Error distribution (bias check)
- `plot_residuals_vs_predicted.png` - Heteroscedasticity diagnostic
- `plot_model_comparison.png` - Ridge vs LightGBM benchmark
- `plot_business_impact.png` - ROI analysis

**Evaluation metrics** (in `reports/`):
- `val_evaluation.csv` - Validation set scores
- `test_evaluation.csv` - Test set scores (final results)

**Trained model artifacts** (in `models/`):
- `best_model.pkl` - Trained LightGBM model
- `preprocessor.pkl` - Fitted preprocessing pipeline
- `best_model_metadata.pkl` - Training metadata

---

## Expected Performance

### Final Test Set Results (61,905 samples)

| Metric | Value |
|--------|-------|
| **R²** | 0.9146 |
| **MAE** | $4,156 |
| **RMSE** | $6,732 |
| **MAPE** | 14.63% |

**Model:** LightGBM with 3,726 iterations (early stopping)  
**Training time:** ~70 seconds on MacBook Air M4  
**Improvement vs baseline:** 42% error reduction

---

## Project Structure
```
Bit_Assessment/
│
├── main.py                    # ← RUN THIS FILE
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── data/
│   └── train_full.csv         # Dataset (place here before running)
│
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── config.py             # Configuration
│   ├── data.py               # Data loading + feature engineering
│   ├── preprocessing.py      # Sklearn-compatible transformers
│   ├── model.py              # LightGBM training
│   ├── evaluation.py         # Metrics calculation
│   ├── visualization.py      # Plot generation
│   └── tune.py               # Hyperparameter optimization (optional)
│
├── reports/                   # Generated outputs (after running main.py)
│   ├── plot_*.png            # 7 visualization plots
│   ├── val_evaluation.csv    # Validation metrics
│   └── test_evaluation.csv   # Test metrics
│
└── models/                    # Trained artifacts (after running main.py)
    ├── best_model.pkl        # LightGBM model
    ├── preprocessor.pkl      # Preprocessing pipeline
    └── metadata files        # Training metadata
```
---

## Technical Details

### Dependencies (requirements.txt)
```
joblib==1.5.3          # Model serialization
lightgbm==4.6.0        # Gradient boosting
matplotlib==3.10.8     # Plotting
numpy==2.4.2           # Numerical operations
optuna(4.1.0 or higher)
pandas==3.0.0          # Data manipulation
scikit-learn==1.8.0    # ML framework
scipy==1.17.0          # Scientific computing
seaborn==0.13.2        # Statistical visualization
```

### Key Features
- **Target Encoding** for high-cardinality features (5,281 models → 1 feature)
- **Early Stopping** prevents overfitting (3,726 optimal iterations)
- **Log-Transform** on target variable (stabilizes variance)
- **73 engineered features** from 54 raw columns



---
