"""
config.py
=========
Single source of truth for all configuration parameters.
All paths, constants, and hyperparameters are defined here.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
print("PROJECT_ROOT:", PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Data files
TRAIN_DATA = DATA_DIR / "train_full.csv"
SAMPLE_DATA = DATA_DIR / "sample_train.csv"

# ============================================================================
# DATA PARAMETERS
# ============================================================================
TARGET_COLUMN = "Sales Price"
TARGET_LOG = "Log_Sales_Price"  # Transformed target
RANDOM_STATE = 42

# Train/Val/Test split ratios
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# Columns to drop (IDs and non-predictive metadata)
DROP_COLUMNS = [
    'Unnamed: 0', 
    'Sales ID', 
    'Machine ID', 
    'Model ID', 
    'Auctioneer ID', 
    'datasource',
    'Sales date'  # We extract features from this, then drop
]

# Features to extract from Product Class Description
EXTRACT_PATTERNS = {
    'Horsepower': r'(\d+\.?\d*)\s+to\s+\d+\.?\d*\s+Horsepower',
    'Weight_Tons': r'(\d+\.?\d*)\s+to\s+\d+\.?\d*\s+Metric Tons',
    'Digging_Depth_Ft': r'(\d+\.?\d*)\s+to\s+\d+\.?\d*\s+Ft Standard Digging Depth'
}

# Features to impute per Product Group (group-aware imputation)
GROUP_IMPUTE_FEATURES = [
    'MachineHours CurrentMeter',
    'Horsepower',
    'Weight_Tons',
    'Year Made'
]

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
# Outlier capping (based on EDA)
PRICE_LOWER_BOUND = 6100   # 1st percentile
PRICE_UPPER_BOUND = 107500  # 99th percentile

# Year Made filtering
MIN_YEAR = 1950  # Filter out data errors (years like 1000)

# Missing value thresholds
DROP_MISSING_THRESHOLD = 0.80  # Drop columns with >80% missing

# ============================================================================
# ENCODING STRATEGY
# ============================================================================
# Low cardinality features (<10 unique) → One-Hot Encoding
ONEHOT_FEATURES = [
    'Product Group',
    'Enclosure',
    'Hydraulics',
    'Transmission',
    'Drive_System'
]

# High cardinality features (>50 unique) → Target Encoding
TARGET_ENCODE_FEATURES = [
    'Model Description',
    'Base Model',
    'State of Usage',
    'Product Class Description'
]

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
# Baseline models to train
MODELS = {
    'linear': {
        'name': 'Linear Regression',
        'params': {}
    },
    'rf': {
        'name': 'Random Forest',
        'params': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'n_jobs': -1,
            'random_state': RANDOM_STATE
        }
    },
    'xgb': {
        'name': 'XGBoost',
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'random_state': RANDOM_STATE
        }
    }
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
METRICS = ['MAE', 'RMSE', 'R2', 'MAPE']

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
PLOT_DPI = 150
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
COLOR_PALETTE = 'Set2'