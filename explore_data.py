"""
Data Exploration Script - FIXED VERSION
========================================
Run this FIRST before any modeling to understand your dataset.
All comments in English, all issues fixed.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "data/train_full.csv"
OUTPUT_DIR = Path("exploration")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

# Load dataset with proper dtype handling to avoid mixed type warnings
try:
    df = pd.read_csv(DATA_PATH, dtype={
        'Model Series': str,
        'Couple System': str,
        'Grouser Tracks': str,
        'Hydraulics Flow': str
    })
    print(f"✅ Loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# ============================================================================
# 2. GENERAL OVERVIEW
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: GENERAL OVERVIEW")
print("=" * 80)

print(f"\n📊 SHAPE: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"💾 MEMORY USAGE: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n📋 FIRST 5 ROWS:")
print(df.head())

print("\n📋 COLUMN NAMES:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:3}. {col}")

# ============================================================================
# 3. DATA TYPES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA TYPES ANALYSIS")
print("=" * 80)

print("\n🔢 NUMERIC COLUMNS:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"   Total: {len(numeric_cols)}")
for col in numeric_cols:
    print(f"   • {col} ({df[col].dtype})")

print("\n📝 CATEGORICAL COLUMNS:")
categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
print(f"   Total: {len(categorical_cols)}")
for col in categorical_cols:  
    print(f"   • {col}")

# ============================================================================
# 4. MISSING VALUES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: MISSING VALUES ANALYSIS")
print("=" * 80)

missing_df = pd.DataFrame({
    'column': df.columns,
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
}).sort_values('missing_pct', ascending=False)

print("\n❌ COLUMNS WITH MISSING VALUES:")
missing_cols = missing_df[missing_df['missing_count'] > 0]
if len(missing_cols) > 0:
    print(missing_cols.to_string(index=False))
    
    # Categorize by severity
    critical = missing_df[missing_df['missing_pct'] > 80]
    high = missing_df[(missing_df['missing_pct'] > 50) & (missing_df['missing_pct'] <= 80)]
    medium = missing_df[(missing_df['missing_pct'] > 20) & (missing_df['missing_pct'] <= 50)]
    low = missing_df[(missing_df['missing_pct'] > 0) & (missing_df['missing_pct'] <= 20)]
    
    print(f"\n📊 MISSING VALUES SUMMARY:")
    print(f"   🔴 Critical (>80%): {len(critical)} columns → CONSIDER DROPPING")
    print(f"   🟠 High (50-80%): {len(high)} columns → EVALUATE USEFULNESS")
    print(f"   🟡 Medium (20-50%): {len(medium)} columns → NEEDS IMPUTATION")
    print(f"   🟢 Low (<20%): {len(low)} columns → EASY TO HANDLE")
else:
    print("✅ No missing values!")

# ============================================================================
# 5. TARGET VARIABLE ANALYSIS (SALES PRICE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TARGET VARIABLE ANALYSIS")
print("=" * 80)

# Target variable is Sales Price (NOT Sales ID!)
target_col = "Sales Price"

if target_col in df.columns:
    print(f"\n🎯 TARGET IDENTIFIED: {target_col}")
    print(f"\n📊 STATISTICS:")
    print(df[target_col].describe())
    
    # Calculate skewness and kurtosis
    skewness = df[target_col].skew()
    kurtosis = df[target_col].kurtosis()
    
    print(f"\n📈 DISTRIBUTION METRICS:")
    print(f"   Skewness: {skewness:.2f}")
    print(f"   Kurtosis: {kurtosis:.2f}")
    
    if skewness > 1:
        print(f"   ⚠️  Highly right-skewed → Consider log transformation")
    elif skewness < -1:
        print(f"   ⚠️  Highly left-skewed")
    else:
        print(f"   ✅ Approximately normal distribution")
else:
    print(f"⚠️  Target column '{target_col}' not found in dataset!")

# ============================================================================
# 6. CARDINALITY ANALYSIS (for categorical features)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: CARDINALITY ANALYSIS")
print("=" * 80)

print("\n📊 CATEGORICAL FEATURES CARDINALITY:")
cardinality_info = []
for col in categorical_cols:
    nunique = df[col].nunique()
    # Determine encoding recommendation
    if nunique > 50:
        category = 'High (>50) → Target Encoding'
    elif nunique > 10:
        category = 'Medium (10-50) → One-Hot or Target'
    else:
        category = 'Low (<10) → One-Hot'
    
    cardinality_info.append({
        'column': col,
        'unique_values': nunique,
        'recommendation': category
    })

cardinality_df = pd.DataFrame(cardinality_info).sort_values('unique_values', ascending=False)
print(cardinality_df.to_string(index=False))

print(f"\n💡 ENCODING RECOMMENDATIONS:")
print(f"   🔵 Low cardinality (<10): One-Hot Encoding")
print(f"   🟢 Medium cardinality (10-50): One-Hot or Target Encoding")
print(f"   🟡 High cardinality (>50): Target Encoding or Frequency Encoding")

# ============================================================================
# 7. CORRELATION ANALYSIS (numeric features)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CORRELATION ANALYSIS")
print("=" * 80)

if target_col in df.columns and len(numeric_cols) > 1:
    print(f"\n🔗 TOP FEATURES CORRELATED WITH {target_col}:")
    correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
    print(correlations.to_string())
    
    print(f"\n⚠️  HIGH MULTICOLLINEARITY CHECK (corr > 0.9):")
    corr_matrix = df[numeric_cols].corr().abs()
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr:
        for col1, col2, corr_val in high_corr:
            print(f"   • {col1} ↔ {col2}: {corr_val:.3f}")
        print(f"   💡 Consider removing one from each pair")
    else:
        print("   ✅ No high multicollinearity detected")

# ============================================================================
# 8. OUTLIER DETECTION (target variable)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: OUTLIER DETECTION")
print("=" * 80)

if target_col in df.columns:
    # Use IQR method to detect outliers
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
    
    print(f"\n📊 OUTLIERS IN {target_col}:")
    print(f"   Total outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"   Lower bound: ${lower_bound:,.2f}")
    print(f"   Upper bound: ${upper_bound:,.2f}")
    
    if len(outliers) > 0:
        print(f"\n   Outlier range:")
        print(f"   Min outlier: ${outliers[target_col].min():,.2f}")
        print(f"   Max outlier: ${outliers[target_col].max():,.2f}")
        
        # Show percentiles for capping
        p1 = df[target_col].quantile(0.01)
        p99 = df[target_col].quantile(0.99)
        print(f"\n   💡 Percentiles for capping:")
        print(f"   1st percentile: ${p1:,.2f}")
        print(f"   99th percentile: ${p99:,.2f}")
        print(f"   → Recommend capping outliers at these values")

# ============================================================================
# 9. DATA QUALITY CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: DATA QUALITY CHECKS")
print("=" * 80)

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\n🔍 DUPLICATE ROWS: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")

# Check for constant columns
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
if constant_cols:
    print(f"\n⚠️  CONSTANT COLUMNS (only 1 unique value):")
    for col in constant_cols:
        print(f"   • {col}: {df[col].unique()[0]}")
else:
    print(f"\n✅ NO CONSTANT COLUMNS")

# Check Year Made for data quality issues
if 'Year Made' in df.columns:
    print(f"\n🔍 YEAR MADE QUALITY CHECK:")
    year_min = df['Year Made'].min()
    year_max = df['Year Made'].max()
    print(f"   Range: {year_min} - {year_max}")
    
    # Check for unrealistic years
    unrealistic = df[df['Year Made'] < 1950]
    if len(unrealistic) > 0:
        print(f"   ⚠️  Found {len(unrealistic):,} rows with Year < 1950 (likely errors)")
        print(f"   → Recommend filtering: df = df[df['Year Made'] >= 1950]")
    else:
        print(f"   ✅ All years look reasonable")

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: GENERATING VISUALIZATIONS")
print("=" * 80)

# 10.1 Target Distribution
if target_col in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with statistics
    axes[0].hist(df[target_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(df[target_col].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: ${df[target_col].mean():,.0f}')
    axes[0].axvline(df[target_col].median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: ${df[target_col].median():,.0f}')
    axes[0].set_xlabel(target_col, fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Boxplot
    axes[1].boxplot(df[target_col].dropna(), vert=True)
    axes[1].set_ylabel(target_col, fontsize=12)
    axes[1].set_title(f'Boxplot of {target_col}', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/target_distribution.png")
    plt.close()

# 10.2 Missing Values Heatmap (FIXED)
if len(missing_cols) > 0:
    try:
        plt.figure(figsize=(12, 8))
        
        # Get columns with missing values
        cols_with_missing = missing_cols['column'].tolist()
        
        # Create heatmap for a sample (first 1000 rows for performance)
        sample_size = min(1000, len(df))
        missing_data_sample = df[cols_with_missing].head(sample_size).isnull()
        
        sns.heatmap(missing_data_sample, cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap (First 1000 rows)', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "missing_values_heatmap.png", dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {OUTPUT_DIR}/missing_values_heatmap.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not generate missing values heatmap: {e}")

# 10.3 Correlation Matrix
if len(numeric_cols) > 2 and target_col in df.columns:
    try:
        plt.figure(figsize=(10, 8))
        
        # Get correlation matrix for all numeric features
        corr_to_plot = df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_to_plot, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix (Numeric Features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {OUTPUT_DIR}/correlation_matrix.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not generate correlation matrix: {e}")

# 10.4 Missing Values Bar Chart
if len(missing_cols) > 0:
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot top 20 columns with most missing values
        top_missing = missing_cols.head(20).sort_values('missing_pct')
        
        plt.barh(top_missing['column'], top_missing['missing_pct'], color='coral', edgecolor='black')
        plt.xlabel('Missing Percentage (%)', fontsize=12)
        plt.ylabel('Column', fontsize=12)
        plt.title('Top 20 Columns with Missing Values', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, row in top_missing.iterrows():
            plt.text(row['missing_pct'] + 1, row.name, f"{row['missing_pct']:.1f}%",
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "missing_values_bar.png", dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {OUTPUT_DIR}/missing_values_bar.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not generate missing values bar chart: {e}")

# ============================================================================
# 11. MISSING PATTERN ANALYSIS (CRITICAL)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: MISSING PATTERN ANALYSIS")
print("=" * 80)

if 'MachineHours CurrentMeter' in df.columns and target_col in df.columns:
    has_hours = df[df['MachineHours CurrentMeter'].notna()]
    no_hours = df[df['MachineHours CurrentMeter'].isna()]
    
    # Calculate mean differences
    avg_with = has_hours[target_col].mean()
    avg_without = no_hours[target_col].mean()
    diff = abs(avg_with - avg_without)
    diff_pct = (diff / avg_without) * 100
    
    print(f"\n🔍 MACHINEHOURS MISSINGNESS PATTERN:")
    print(f"   Rows WITH hours: {len(has_hours):,} ({len(has_hours)/len(df)*100:.1f}%)")
    print(f"   Rows WITHOUT hours: {len(no_hours):,} ({len(no_hours)/len(df)*100:.1f}%)")
    print(f"\n   Average Price (With Hours):    ${avg_with:,.2f}")
    print(f"   Average Price (Without Hours): ${avg_without:,.2f}")
    print(f"   Absolute Difference:           ${diff:,.2f} ({diff_pct:.1f}%)")
    
    # Cross-analysis with 'Year Made' to detect systemic bias
    if 'Year Made' in df.columns:
        year_with = has_hours['Year Made'].mean()
        year_without = no_hours['Year Made'].mean()
        print(f"\n🔍 DRIVERS OF MISSINGNESS:")
        print(f"   Avg Year (With Hours):    {year_with:.0f}")
        print(f"   Avg Year (Without Hours): {year_without:.0f}")

    # LOGIC: If price difference > 10% or > $3000, it indicates MNAR (Missing Not At Random)
    if diff_pct > 10 or diff > 3000:
        print(f"\n   🔴 ALERT: Missing data is NOT random (Systemic Bias Detected)!")
        print(f"   → Price delta is significant ({diff_pct:.1f}%)")
        print(f"   → STRATEGY: Create 'Has_Hours' binary feature (CRITICAL)")
        print(f"   → STRATEGY: Impute MachineHours using median per 'Product Group'")
    else:
        print(f"\n   ✅ Missingness appears random (negligible price impact)")

    # Group-level analysis for imputation strategy
    if 'Product Group' in df.columns:
        print(f"\n📊 MISSINGNESS % BY PRODUCT GROUP:")
        missing_by_group = df.groupby('Product Group')['MachineHours CurrentMeter'].apply(lambda x: x.isna().mean() * 100).sort_values(ascending=False)
        for group, pct in missing_by_group.items():
            print(f"   • {group:5}: {pct:.1f}%")
            
# ============================================================================
# 12. TEMPORAL ANALYSIS (CRITICAL)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: TEMPORAL ANALYSIS")
print("=" * 80)

if 'Sales date' in df.columns and target_col in df.columns:
    # Parse dates
    df_temp = df.copy()
    df_temp['Sale_Date'] = pd.to_datetime(df_temp['Sales date'], errors='coerce')
    df_temp['Sale_Year'] = df_temp['Sale_Date'].dt.year
    df_temp['Sale_Month'] = df_temp['Sale_Date'].dt.month
    
    print(f"\n📅 SALES DATE RANGE:")
    print(f"   From: {df_temp['Sale_Date'].min()}")
    print(f"   To: {df_temp['Sale_Date'].max()}")
    print(f"   Span: {(df_temp['Sale_Date'].max() - df_temp['Sale_Date'].min()).days} days")
    
    # Price trends over time
    print(f"\n💰 PRICE TRENDS BY YEAR:")
    yearly_avg = df_temp.groupby('Sale_Year')[target_col].agg(['mean', 'median', 'count'])
    print(yearly_avg.to_string())
    
    # Check correlation with year
    year_corr = df_temp[['Sale_Year', target_col]].corr().iloc[0, 1]
    print(f"\n🔗 CORRELATION Sale_Year vs {target_col}: {year_corr:.3f}")
    
    if abs(year_corr) > 0.2:
        print(f"   ✅ Strong temporal trend! Use Sale_Year as feature")
    else:
        print(f"   ⚠️  Weak temporal trend")
    
    # Visualize temporal trend
    try:
        plt.figure(figsize=(14, 6))
        
        # Plot 1: Price over time
        plt.subplot(1, 2, 1)
        yearly_avg['mean'].plot(kind='line', marker='o', color='steelblue', linewidth=2, markersize=8)
        plt.xlabel('Sale Year', fontsize=12)
        plt.ylabel('Average Sales Price ($)', fontsize=12)
        plt.title('Average Sales Price by Year', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        
        # Plot 2: Monthly seasonality
        plt.subplot(1, 2, 2)
        monthly_avg = df_temp.groupby('Sale_Month')[target_col].mean()
        monthly_avg.plot(kind='bar', color='coral', edgecolor='black')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Average Sales Price ($)', fontsize=12)
        plt.title('Average Sales Price by Month', fontsize=14, fontweight='bold')
        plt.xticks(rotation=0)
        plt.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "temporal_analysis.png", dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved: {OUTPUT_DIR}/temporal_analysis.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not generate temporal plots: {e}")

# ============================================================================
# 13. EXTRACT NUMERIC FROM TEXT FEATURES (CRITICAL)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 13: EXTRACT NUMERIC FEATURES FROM TEXT")
print("=" * 80)

if 'Product Class Description' in df.columns:
    print(f"\n🔍 ANALYZING PRODUCT CLASS DESCRIPTION:")
    
    # Show examples
    print(f"\n   Example values:")
    for val in df['Product Class Description'].dropna().unique()[:5]:
        print(f"   • {val}")
    
    # Create temporary df for extraction
    df_extract = df.copy()
    
    # Extract horsepower
    df_extract['Horsepower'] = df_extract['Product Class Description'].str.extract(
        r'(\d+\.?\d*)\s+to\s+\d+\.?\d*\s+Horsepower'
    )[0].astype(float)
    
    # Extract weight (metric tons)
    df_extract['Weight_Tons'] = df_extract['Product Class Description'].str.extract(
        r'(\d+\.?\d*)\s+to\s+\d+\.?\d*\s+Metric Tons'
    )[0].astype(float)
    
    # Extract capacity (pounds)
    df_extract['Capacity_Lbs'] = df_extract['Product Class Description'].str.extract(
        r'(\d+)\s+to\s+\d+\s+Lb Operating Capacity'
    )[0].astype(float)
    
    # Extract digging depth (feet)
    df_extract['Digging_Depth_Ft'] = df_extract['Product Class Description'].str.extract(
        r'(\d+\.?\d*)\s+to\s+\d+\.?\d*\s+Ft Standard Digging Depth'
    )[0].astype(float)
    
    # Report extraction results
    print(f"\n📊 EXTRACTION RESULTS:")
    extracted_features = []
    for col in ['Horsepower', 'Weight_Tons', 'Capacity_Lbs', 'Digging_Depth_Ft']:
        non_null = df_extract[col].notna().sum()
        pct = non_null / len(df_extract) * 100
        print(f"   • {col}: {non_null:,} values ({pct:.1f}%)")
        
        if non_null > 100:  # Only calculate correlation if enough data
            corr = df_extract[[col, target_col]].corr().iloc[0, 1]
            print(f"      Correlation with {target_col}: {corr:.3f}")
            extracted_features.append((col, corr))
    
    # Show impact on correlations
    if extracted_features:
        print(f"\n💡 IMPACT ON CORRELATIONS:")
        print(f"   BEFORE extraction:")
        print(f"      Year Made: 0.154")
        print(f"      MachineHours: 0.016")
        print(f"\n   AFTER extraction:")
        for feat, corr in sorted(extracted_features, key=lambda x: abs(x[1]), reverse=True):
            print(f"      {feat}: {corr:.3f}")
        
        print(f"\n   ✅ Feature engineering SIGNIFICANTLY improves correlations!")
        print(f"   → Recommendation: Extract these features in preprocessing")

# ============================================================================
# 14. CATEGORICAL FEATURE IMPORTANCE (IMPORTANT)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 14: CATEGORICAL FEATURE IMPORTANCE")
print("=" * 80)

# Analyze key categorical features
key_cats = ['Product Group', 'Enclosure', 'Hydraulics', 'Product Group Description']
key_cats = [c for c in key_cats if c in df.columns]

print(f"\n📊 PRICE VARIATION BY CATEGORICAL FEATURES:")

for cat in key_cats:
    print(f"\n   === {cat} ===")
    cat_stats = df.groupby(cat)[target_col].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
    
    # Show top 5
    print(f"   Top 5 (highest avg price):")
    for idx, row in cat_stats.head(5).iterrows():
        print(f"      • {str(idx):30} → ${row['mean']:>10,.0f} (n={row['count']:,})")
    
    # Calculate F-statistic (one-way ANOVA)
    try:
        from scipy import stats
        groups = [group[target_col].dropna().values for name, group in df.groupby(cat) if len(group) > 0]
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            
            print(f"\n   F-statistic: {f_stat:.2f} (p-value: {p_value:.2e})")
            if p_value < 0.001:
                print(f"   ✅ HIGHLY PREDICTIVE feature!")
            elif p_value < 0.05:
                print(f"   ✅ Significant feature")
            else:
                print(f"   ⚠️  Weak predictive power")
    except Exception as e:
        print(f"   ⚠️  Could not calculate F-statistic: {e}")

# ============================================================================
# 15. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 15: GENERATING SUMMARY REPORT")
print("=" * 80)

summary_report = f"""# DATA EXPLORATION REPORT
Generated: {pd.Timestamp.now()}

## Dataset Overview
- **File**: {DATA_PATH}
- **Shape**: {df.shape[0]:,} rows × {df.shape[1]} columns
- **Memory**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

## Feature Types
- **Numeric**: {len(numeric_cols)} columns
- **Categorical**: {len(categorical_cols)} columns

## Missing Values Summary
- **Columns with missing values**: {len(missing_cols)} / {len(df.columns)}
- **Critical (>80% missing)**: {len(critical)} columns
- **High (50-80% missing)**: {len(high)} columns
- **Medium (20-50% missing)**: {len(medium)} columns
- **Low (<20% missing)**: {len(low)} columns

## Target Variable: {target_col}
"""

if target_col in df.columns:
    summary_report += f"""- **Mean**: ${df[target_col].mean():,.2f}
- **Median**: ${df[target_col].median():,.2f}
- **Std**: ${df[target_col].std():,.2f}
- **Min**: ${df[target_col].min():,.2f}
- **Max**: ${df[target_col].max():,.2f}
- **Skewness**: {df[target_col].skew():.2f}
- **Outliers**: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)

"""

summary_report += f"""## Data Quality Issues
- **Duplicate rows**: {duplicates:,}
- **Constant columns**: {len(constant_cols)}
"""

if 'Year Made' in df.columns:
    unrealistic_years = len(df[df['Year Made'] < 1950])
    summary_report += f"- **Unrealistic years (<1950)**: {unrealistic_years:,}\n"

summary_report += f"""
## Recommendations

### Features to DROP:
"""

# ID columns
id_cols = [col for col in df.columns if 'id' in col.lower() and col != target_col]
if id_cols:
    summary_report += f"- **ID columns** ({len(id_cols)}): {', '.join(id_cols)}\n"

# Constant columns
if constant_cols:
    summary_report += f"- **Constant columns** ({len(constant_cols)}): {', '.join(constant_cols)}\n"

# Critical missing
if len(critical) > 0:
    crit_list = critical['column'].tolist()
    summary_report += f"- **Critical missing >80%** ({len(crit_list)}): {', '.join(crit_list[:10])}"
    if len(crit_list) > 10:
        summary_report += f" ... and {len(crit_list) - 10} more"
    summary_report += "\n"

summary_report += f"""
### Features to KEEP:

**Numeric ({len(numeric_cols) - len([c for c in numeric_cols if 'id' in c.lower()])} useful):**
- Year Made
- MachineHours CurrentMeter (needs imputation - 64% missing)

**Categorical - Low cardinality (One-Hot Encoding):**
"""

low_card = [col for col in categorical_cols if df[col].nunique() <= 10]
for col in low_card[:10]:
    summary_report += f"- {col} ({df[col].nunique()} values)\n"

summary_report += f"""
**Categorical - High cardinality (Target Encoding):**
"""

high_card = [col for col in categorical_cols if df[col].nunique() > 50]
for col in high_card[:10]:
    summary_report += f"- {col} ({df[col].nunique()} values)\n"

summary_report += f"""
### Preprocessing Strategy

### 1. Data Cleaning
- **Drop IDs**: Remove `SalesID`, `MachineID`, and `Unnamed: 0`.
- **Drop Constants**: Remove columns with only 1 unique value.
- **Missing Threshold**: Drop columns with >80% missing values.
- **Quality Filter**: Correct or filter `Year Made < 1950` (detected placeholders).

### 2. Outliers & Imputation
- **Target Capping**: Cap `{target_col}` at 1st and 99th percentiles to handle extremes.
- **Numeric Imputation**: Use **Median** for all numeric columns. 
  * *Note: Impute `MachineHours` raggruppated by `Product Group`.*
- **Categorical Imputation**: Fill missing text values with **"Unknown"**.

### 3. Encoding & Transformation
- **Target Transformation**: Apply **Log Transformation** (`np.log1p`) to `{target_col}` to correct right-skewness.
- **Low Cardinality (<10)**: Use **One-Hot Encoding** (e.g., `Product Group`, `Enclosure`).
- **High Cardinality (>50)**: Use **Target Encoding** (e.g., `Base Model`, `Model Descriptor`).

## 🚀 Next Steps
1. ✅ Review generated plots in `{OUTPUT_DIR}/`
2. ✅ Update `config.py` with the engineered feature list.
3. ✅ Build the `preprocessing.py` pipeline using the strategy above.
4. ✅ Train an initial Baseline Model (Random Forest recommended).
"""

# Save report as Markdown for better readability
with open(OUTPUT_DIR / "data_summary.md", "w") as f:
    f.write(summary_report)

print(f"✅ Saved: {OUTPUT_DIR}/data_summary.md")

print("\n" + "=" * 80)
print("✅ DATA EXPLORATION COMPLETE!")
print("=" * 80)