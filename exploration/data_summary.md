# DATA EXPLORATION REPORT
Generated: 2026-02-02 13:45:17.125048

## Dataset Overview
- **File**: data/train_full.csv
- **Shape**: 412,698 rows × 54 columns
- **Memory**: 830.96 MB

## Feature Types
- **Numeric**: 9 columns
- **Categorical**: 45 columns

## Missing Values Summary
- **Columns with missing values**: 40 / 54
- **Critical (>80% missing)**: 21 columns
- **High (50-80% missing)**: 14 columns
- **Medium (20-50% missing)**: 3 columns
- **Low (<20% missing)**: 2 columns

## Target Variable: Sales Price
- **Mean**: $31,215.18
- **Median**: $24,000.00
- **Std**: $23,141.74
- **Min**: $4,750.00
- **Max**: $142,000.00
- **Skewness**: 1.52
- **Outliers**: 23,648 (5.73%)

## Data Quality Issues
- **Duplicate rows**: 0
- **Constant columns**: 0
- **Unrealistic years (<1950)**: 39,542

## Recommendations

### Features to DROP:
- **ID columns** (5): Sales ID, Machine ID, Model ID, Auctioneer ID, Machine Width
- **Critical missing >80%** (21): Clean Room, Engine Horsepower, Machine Width, Extra features, Push Block, Scarifier, Tip Control, Grouser Tracks, Hydraulics Flow, Couple System ... and 11 more

### Features to KEEP:

**Numeric (5 useful):**
- Year Made
- MachineHours CurrentMeter (needs imputation - 64% missing)

**Categorical - Low cardinality (One-Hot Encoding):**
- Usage Band (3 values)
- Machine Size (6 values)
- Product Group (6 values)
- Product Group Description (6 values)
- Driver System (4 values)
- Enclosure (6 values)
- Control (2 values)
- Touchpad Type (4 values)
- Control Type (3 values)
- Version (2 values)

**Categorical - High cardinality (Target Encoding):**
- Sales date (4013 values)
- Model Description (5059 values)
- Base Model (1961 values)
- Secondary Description (177 values)
- Model Series (120 values)
- Model Descriptor (137 values)
- Product Class Description (74 values)
- State of Usage (53 values)

### Preprocessing Strategy

### 1. Data Cleaning
- **Drop IDs**: Remove `SalesID`, `MachineID`, and `Unnamed: 0`.
- **Drop Constants**: Remove columns with only 1 unique value.
- **Missing Threshold**: Drop columns with >80% missing values.
- **Quality Filter**: Correct or filter `Year Made < 1950` (detected placeholders).

### 2. Outliers & Imputation
- **Target Capping**: Cap `Sales Price` at 1st and 99th percentiles to handle extremes.
- **Numeric Imputation**: Use **Median** for all numeric columns. 
  * *Note: Impute `MachineHours` raggruppated by `Product Group`.*
- **Categorical Imputation**: Fill missing text values with **"Unknown"**.

### 3. Encoding & Transformation
- **Target Transformation**: Apply **Log Transformation** (`np.log1p`) to `Sales Price` to correct right-skewness.
- **Low Cardinality (<10)**: Use **One-Hot Encoding** (e.g., `Product Group`, `Enclosure`).
- **High Cardinality (>50)**: Use **Target Encoding** (e.g., `Base Model`, `Model Descriptor`).

## 🚀 Next Steps
1. ✅ Review generated plots in `exploration/`
2. ✅ Update `config.py` with the engineered feature list.
3. ✅ Build the `preprocessing.py` pipeline using the strategy above.
4. ✅ Train an initial Baseline Model (Random Forest recommended).
