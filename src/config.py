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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist immediately upon import
# This fails early if permissions are wrong.
for path in [MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Data files
TRAIN_DATA = DATA_DIR / "train_full.csv"

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
# Columns to drop (IDs, leakage, redundant, and low-signal features)
DROP_FEATURES = (
    # --- Leakage & Identifiers (no predictive value) ---
    'Unnamed: 0',              # Index = proxy for date (leakage)
    'Sales ID',                # Unique sale identifier
    'Machine ID',              # Unique machine identifier
    'Model ID',                # 1264+ unique values, no direct price signal
    'datasource',              # Constant (1 unique value)
    'Auctioneer ID',           # Constant in this subset
    'Sales date',              # Already extracted to Sale_Year/Sale_Month
    'Product Group Description',  # Redundant with Product Group (same info as text)

    # --- >90% null, virtually no signal ---
    'Extra features',
    'Machine Width',
    'Clean Room',
    'Engine Horsepower',
    'Push Block',
    'Scarifier',
    'Tip Control',

    # --- >80% null AND zero feature importance ---
    'Touchpad Type',
    'Turbo Charged',
    'Couple System',
    'Grouser Tracks',
    'Hydraulics Flow',
    'Backhoe Mounting',
    'Blade Type',
    'Travel Possibilities',
    'Differential Type',
    'Steering Controls',
)

# Features to extract from Product Class Description
EXTRACT_PATTERNS = {
    'Horsepower': r'(\d+\.?\d*)\s+to\s+\d+\.?\d*\s+Horsepower',
    'Weight_Tons': r'(\d+\.?\d*)\s+to\s+\d+\.?\d*\s+Metric Tons',
    'Digging_Depth_Ft': r'(\d+\.?\d*)\s+to\s+\d+\.?\d*\s+Ft Standard Digging Depth'
}

# Features to impute per Product Group (group-aware imputation)
GROUP_IMPUTE_FEATURES = (
    'MachineHours CurrentMeter',
    'Horsepower',
    'Weight_Tons',
    'Year Made'
)

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
# Outlier capping (based on EDA)
PRICE_LOWER_BOUND = 6100
PRICE_UPPER_BOUND = 107500

# Year Made filtering
MIN_YEAR = 1950  # Filter out data errors (years like 1000)

# Missing value thresholds
DROP_MISSING_THRESHOLD = 0.80  # Drop columns with >80% missing

# Scaling (disabled: LightGBM is tree-based, doesn't benefit from scaling)
APPLY_SCALING = False

# ============================================================================
# ENCODING STRATEGY
# ============================================================================
# High cardinality features → Target Encoding
TARGET_ENCODE_FEATURES = (
    'Model Description',
    'Base Model',
    'Secondary Description',
    'Product Class Description',
    'State of Usage',
    'Hydraulics',
    'Screen Size',
    'Screen Size.1',
    'Stick Length',
)

# Low cardinality features → One-Hot Encoding
ONEHOT_FEATURES = (
    'Usage Band',
    'Machine Size',
    'Product Group',
    'Driver System',
    'Enclosure',
    'Control',
    'Control Type',
    'Version',
    'Transmission',
    'Ripper',
    'Coupler',
    'Thumb',
    'Pattern Changer',
    'Grouser Type',
)

# Threshold for rare label grouping (categories with < N occurrences → "Rare")
RARE_LABEL_THRESHOLD = 10

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Models to train
MODELS = {
    'ridge': {
        'name': 'Ridge Regression',
        'description': 'Simple linear model with L2 regularization',
        'params': {
            'alpha': 1.0,  # Regularization strength
            'random_state': RANDOM_STATE
        }
    },
    'lightgbm': {
        'name': 'LightGBM',
        'description': 'Gradient boosting optimized for large datasets (110MB)',
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.01,
            'num_leaves': 256,
            'max_depth': 12,
            'min_child_samples': 40,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'reg_alpha': 1.5,   # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'n_jobs': -1,
            'random_state': RANDOM_STATE,
            'verbose': -1
        }
    }
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Primary metrics for model evaluation
PRIMARY_METRICS = ['MAE', 'RMSE', 'R2']

# Metric for selecting best model (will be maximized for R2, minimized for MAE/RMSE)
BEST_METRIC = 'R2'

# Metric descriptions for reporting
METRIC_DESCRIPTIONS = {
    'MAE': 'Mean Absolute Error (lower is better)',
    'RMSE': 'Root Mean Squared Error (lower is better)',
    'R2': 'R² Score (higher is better)'
}

# Metric formatting for display
METRIC_FORMATS = {
    'MAE': '${:,.0f}',
    'RMSE': '${:,.0f}',
    'R2': '{:.4f}'
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
PLOT_SETTINGS = {
    'DPI': 150,
    'STYLE': 'whitegrid',
    'PALETTE': 'Set2'
}