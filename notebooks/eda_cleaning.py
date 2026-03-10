# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Configure display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("Libraries imported successfully")
print("="*70)

# ============================================================================
# SECTION 1: LOAD DATASET
# ============================================================================

# Loading the loan default dataset
loan_data = pd.read_csv('train.csv')

print("\nDataset loaded successfully")
print(f"Total rows: {loan_data.shape[0]:,}")
print(f"Total columns: {loan_data.shape[1]}")
print("="*70)

# ============================================================================
# SECTION 2: DATASET OVERVIEW
# ============================================================================

# Displaying first few rows
print("\nFirst 5 rows of the dataset:")
print(loan_data.head())

# Displaying column names
print("\nColumn names:")
for i, col in enumerate(loan_data.columns, 1):
    print(f"{i}. {col}")

# Checking data types and non-null counts
print("\nDataset information:")
print(loan_data.info())

# Calculating memory usage
memory_usage_mb = loan_data.memory_usage(deep=True).sum() / (1024**2)
print(f"\nMemory usage: {memory_usage_mb:.2f} MB")

# Identifying numerical and categorical columns
numerical_cols = loan_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = loan_data.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print("="*70)

# ============================================================================
# SECTION 3: MISSING VALUES ANALYSIS
# ============================================================================

# Checking for missing values
print("\nChecking for missing values:")
missing_counts = loan_data.isnull().sum()
missing_percentages = (loan_data.isnull().sum() / len(loan_data)) * 100

missing_summary = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Percentage': missing_percentages
}).sort_values('Missing_Count', ascending=False)

# Displaying columns with missing values
columns_with_missing = missing_summary[missing_summary['Missing_Count'] > 0]
if len(columns_with_missing) > 0:
    print(columns_with_missing)
    print(f"\nTotal missing values: {missing_counts.sum()}")
else:
    print("No missing values found in the dataset")

# Visualizing missing values with heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(loan_data.isnull(), cbar=True, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap', fontsize=14)
plt.xlabel('Columns')
plt.tight_layout()
plt.savefig('missing_values_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nMissing values heatmap saved as 'missing_values_heatmap.png'")
print("="*70)

# ============================================================================
# SECTION 4: TARGET VARIABLE ANALYSIS
# ============================================================================

# Analyzing target variable distribution
target_column = 'Loan Status'

print(f"\nAnalyzing target variable: {target_column}")
print("\nValue counts:")
print(loan_data[target_column].value_counts())

print("\nPercentage distribution:")
target_distribution = loan_data[target_column].value_counts(normalize=True) * 100
print(target_distribution)

# Calculating class imbalance ratio
value_counts_target = loan_data[target_column].value_counts()
majority_class_count = value_counts_target.iloc[0]
minority_class_count = value_counts_target.iloc[1]
imbalance_ratio = majority_class_count / minority_class_count

print(f"\nClass imbalance analysis:")
print(f"Majority class count: {majority_class_count:,} ({majority_class_count/len(loan_data)*100:.1f}%)")
print(f"Minority class count: {minority_class_count:,} ({minority_class_count/len(loan_data)*100:.1f}%)")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
print("\nNote: Dataset shows significant class imbalance")
print("Solution: Use stratified split and class_weight='balanced' in models")

# Visualizing target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
loan_data[target_column].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Loan Status Distribution', fontsize=12)
axes[0].set_xlabel('Loan Status (0=Paid, 1=Default)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Fully Paid', 'Default'], rotation=0)

# Adding value labels on bars
for i, v in enumerate(loan_data[target_column].value_counts()):
    axes[0].text(i, v + 1000, str(v), ha='center', fontweight='bold')

# Pie chart
colors_pie = ['green', 'red']
loan_data[target_column].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                              colors=colors_pie, startangle=90)
axes[1].set_title('Loan Status Distribution', fontsize=12)
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nTarget distribution plots saved as 'target_distribution.png'")
print("="*70)

# ============================================================================
# SECTION 5: DESCRIPTIVE STATISTICS
# ============================================================================

# Generating descriptive statistics for numerical features
print("\nDescriptive statistics for numerical features:")
numerical_stats = loan_data[numerical_cols].describe()
print(numerical_stats)

# Calculating skewness and kurtosis for key numerical features
print("\nSkewness and kurtosis for numerical features:")
key_numerical_features = numerical_cols[:10]
for col in key_numerical_features:
    skewness_value = skew(loan_data[col].dropna())
    kurtosis_value = kurtosis(loan_data[col].dropna())
    print(f"{col}:")
    print(f"  Skewness: {skewness_value:.2f}")
    print(f"  Kurtosis: {kurtosis_value:.2f}")

# Summarizing categorical features
print("\nCategorical features summary:")
for col in categorical_cols:
    unique_count = loan_data[col].nunique()
    most_common_value = loan_data[col].mode()[0] if len(loan_data[col].mode()) > 0 else 'N/A'
    most_common_count = loan_data[col].value_counts().iloc[0] if len(loan_data[col].value_counts()) > 0 else 0
    print(f"{col}:")
    print(f"  Unique values: {unique_count}")
    print(f"  Most common: {most_common_value} ({most_common_count} occurrences)")
print("="*70)

# ============================================================================
# SECTION 6: UNIVARIATE ANALYSIS - NUMERICAL FEATURES
# ============================================================================

# Selecting key numerical features for detailed analysis
key_numerical_cols = ['Loan Amount', 'Interest Rate', 'Debit to Income',
                      'Revolving Balance', 'Revolving Utilities', 'Total Accounts',
                      'Open Account', 'Delinquency - two years', 'Inquires - six months']

# Filtering to existing columns
key_numerical_cols = [col for col in key_numerical_cols if col in loan_data.columns]

print(f"\nAnalyzing {len(key_numerical_cols)} key numerical features")

# Creating histograms for numerical features
num_features_plot = len(key_numerical_cols)
n_cols_plot = 3
n_rows_plot = (num_features_plot + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(18, n_rows_plot*4))
axes_flat = axes.flatten()

for idx, col in enumerate(key_numerical_cols):
    axes_flat[idx].hist(loan_data[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes_flat[idx].set_title(f'{col} Distribution', fontsize=10)
    axes_flat[idx].set_xlabel(col)
    axes_flat[idx].set_ylabel('Frequency')
    axes_flat[idx].grid(axis='y', alpha=0.3)
    
    # Adding mean and median lines
    mean_value = loan_data[col].mean()
    median_value = loan_data[col].median()
    axes_flat[idx].axvline(mean_value, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_value:.2f}')
    axes_flat[idx].axvline(median_value, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_value:.2f}')
    axes_flat[idx].legend(fontsize=8)

# Hiding empty subplots
for idx in range(num_features_plot, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Numerical feature distributions saved as 'numerical_distributions.png'")

# Creating box plots for outlier detection
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(18, n_rows_plot*4))
axes_flat = axes.flatten()

for idx, col in enumerate(key_numerical_cols):
    axes_flat[idx].boxplot(loan_data[col].dropna(), vert=True, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
    axes_flat[idx].set_title(f'{col} - Outliers', fontsize=10)
    axes_flat[idx].set_ylabel(col)
    axes_flat[idx].grid(axis='y', alpha=0.3)

# Hiding empty subplots
for idx in range(num_features_plot, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig('numerical_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("Box plots for outlier detection saved as 'numerical_boxplots.png'")

# Identifying outliers using IQR method
print("\nOutlier analysis using IQR method:")
for col in key_numerical_cols:
    Q1 = loan_data[col].quantile(0.25)
    Q3 = loan_data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_data = loan_data[(loan_data[col] < lower_bound) | (loan_data[col] > upper_bound)][col]
    outlier_count = len(outliers_data)
    outlier_percentage = (outlier_count / len(loan_data)) * 100
    
    print(f"{col}:")
    print(f"  Outliers: {outlier_count} ({outlier_percentage:.2f}%)")
    print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
print("="*70)

# ============================================================================
# SECTION 7: UNIVARIATE ANALYSIS - CATEGORICAL FEATURES
# ============================================================================

# Selecting key categorical features for analysis
key_categorical_cols = ['Grade', 'Home Ownership', 'Verification Status',
                        'Payment Plan', 'Initial List Status', 'Application Type']

# Filtering to existing columns
key_categorical_cols = [col for col in key_categorical_cols if col in loan_data.columns]

print(f"\nAnalyzing {len(key_categorical_cols)} categorical features")

# Creating bar charts for categorical features
num_cat_features = len(key_categorical_cols)
n_cols_cat = 2
n_rows_cat = (num_cat_features + n_cols_cat - 1) // n_cols_cat

fig, axes = plt.subplots(n_rows_cat, n_cols_cat, figsize=(16, n_rows_cat*4))
axes_flat_cat = axes.flatten()

for idx, col in enumerate(key_categorical_cols):
    value_counts_cat = loan_data[col].value_counts()
    axes_flat_cat[idx].bar(range(len(value_counts_cat)), value_counts_cat.values,
                           color=sns.color_palette('husl', len(value_counts_cat)))
    axes_flat_cat[idx].set_title(f'{col} Distribution', fontsize=10)
    axes_flat_cat[idx].set_xlabel(col)
    axes_flat_cat[idx].set_ylabel('Count')
    axes_flat_cat[idx].set_xticks(range(len(value_counts_cat)))
    axes_flat_cat[idx].set_xticklabels(value_counts_cat.index, rotation=45, ha='right')
    axes_flat_cat[idx].grid(axis='y', alpha=0.3)
    
    # Adding value labels
    for i, v in enumerate(value_counts_cat.values):
        axes_flat_cat[idx].text(i, v + 100, str(v), ha='center', fontsize=8)

# Hiding empty subplots
for idx in range(num_cat_features, len(axes_flat_cat)):
    axes_flat_cat[idx].set_visible(False)

plt.tight_layout()
plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Categorical feature distributions saved as 'categorical_distributions.png'")

# Displaying value counts and percentages
print("\nValue counts and percentages for categorical features:")
for col in key_categorical_cols:
    print(f"\n{col}:")
    value_counts_summary = loan_data[col].value_counts()
    percentages_summary = loan_data[col].value_counts(normalize=True) * 100
    
    summary_table = pd.DataFrame({
        'Count': value_counts_summary,
        'Percentage': percentages_summary
    })
    print(summary_table)
print("="*70)

# ============================================================================
# SECTION 8: BIVARIATE ANALYSIS - CORRELATION
# ============================================================================

# Calculating correlation matrix for numerical features
print("\nCalculating correlation matrix for numerical features")
correlation_matrix_data = loan_data[numerical_cols].corr()

# Creating correlation heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix_data, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Correlation heatmap saved as 'correlation_heatmap.png'")

# Finding top correlations with target variable
if target_column in numerical_cols:
    target_correlations = correlation_matrix_data[target_column].sort_values(ascending=False)
    
    print(f"\nTop 15 features correlated with {target_column}:")
    print(target_correlations.head(16))
    
    # Visualizing top correlations
    plt.figure(figsize=(10, 8))
    target_correlations[1:16].plot(kind='barh', color='teal')
    plt.title('Top 15 Features Correlated with Loan Default', fontsize=12)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('target_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Target correlations plot saved as 'target_correlations.png'")
print("="*70)

# ============================================================================
# SECTION 9: BIVARIATE ANALYSIS - FEATURES VS TARGET
# ============================================================================

# Comparing numerical features by loan status
features_comparison = ['Loan Amount', 'Interest Rate', 'Debit to Income',
                       'Revolving Balance', 'Revolving Utilities', 'Total Accounts']
features_comparison = [f for f in features_comparison if f in loan_data.columns]

print(f"\nComparing {len(features_comparison)} numerical features by loan status")

num_comparison_features = len(features_comparison)
n_cols_comparison = 2
n_rows_comparison = (num_comparison_features + n_cols_comparison - 1) // n_cols_comparison

fig, axes = plt.subplots(n_rows_comparison, n_cols_comparison, figsize=(16, n_rows_comparison*4))
axes_flat_comparison = axes.flatten()

for idx, col in enumerate(features_comparison):
    loan_data.boxplot(column=col, by=target_column, ax=axes_flat_comparison[idx], patch_artist=True)
    axes_flat_comparison[idx].set_title(f'{col} by Loan Status', fontsize=10)
    axes_flat_comparison[idx].set_xlabel('Loan Status (0=Paid, 1=Default)')
    axes_flat_comparison[idx].set_ylabel(col)
    axes_flat_comparison[idx].get_figure().suptitle('')

# Hiding empty subplots
for idx in range(num_comparison_features, len(axes_flat_comparison)):
    axes_flat_comparison[idx].set_visible(False)

plt.tight_layout()
plt.savefig('features_by_target.png', dpi=300, bbox_inches='tight')
plt.close()
print("Features by target plots saved as 'features_by_target.png'")
print("="*70)

# ============================================================================
# SECTION 10: CATEGORICAL FEATURES VS TARGET
# ============================================================================

# Analyzing default rate by grade
if 'Grade' in loan_data.columns:
    print("\nAnalyzing default rate by loan grade")
    grade_default_rate = loan_data.groupby('Grade')[target_column].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    grade_default_rate.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Default Rate by Loan Grade', fontsize=12)
    plt.xlabel('Grade')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    # Adding value labels
    for i, v in enumerate(grade_default_rate.values):
        plt.text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('default_rate_by_grade.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Default rate by grade saved as 'default_rate_by_grade.png'")
    
    print("\nDefault rate by grade:")
    print(grade_default_rate)

# Analyzing default rate by home ownership
if 'Home Ownership' in loan_data.columns:
    print("\nAnalyzing default rate by home ownership")
    
    # Handling numerical or categorical home ownership
    if loan_data['Home Ownership'].dtype in ['int64', 'float64']:
        print("Note: Home Ownership is numerical, analyzing as-is")
    
    home_default_rate = loan_data.groupby('Home Ownership')[target_column].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    home_default_rate.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Default Rate by Home Ownership', fontsize=12)
    plt.xlabel('Home Ownership')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Adding value labels
    for i, v in enumerate(home_default_rate.values):
        plt.text(i, v + 0.005, f'{v:.2%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('default_rate_by_home_ownership.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Default rate by home ownership saved as 'default_rate_by_home_ownership.png'")
    
    print("\nDefault rate by home ownership:")
    print(home_default_rate)

# Analyzing default rate by verification status
if 'Verification Status' in loan_data.columns:
    print("\nAnalyzing default rate by verification status")
    verification_default_rate = loan_data.groupby('Verification Status')[target_column].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    verification_default_rate.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Default Rate by Verification Status', fontsize=12)
    plt.xlabel('Verification Status')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Adding value labels
    for i, v in enumerate(verification_default_rate.values):
        plt.text(i, v + 0.005, f'{v:.2%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('default_rate_by_verification.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Default rate by verification status saved as 'default_rate_by_verification.png'")
    
    print("\nDefault rate by verification status:")
    print(verification_default_rate)
print("="*70)

# ============================================================================
# SECTION 11: SCATTER PLOTS - KEY RELATIONSHIPS
# ============================================================================

# Creating scatter plots for key relationships
if 'Interest Rate' in loan_data.columns and 'Loan Amount' in loan_data.columns:
    print("\nCreating scatter plot: Interest Rate vs Loan Amount")
    
    plt.figure(figsize=(12, 6))
    
    for status_value, color_value, label_text in [(0, 'green', 'Fully Paid'), (1, 'red', 'Default')]:
        subset_data = loan_data[loan_data[target_column] == status_value]
        plt.scatter(subset_data['Interest Rate'], subset_data['Loan Amount'],
                   alpha=0.3, c=color_value, label=label_text, s=10)
    
    plt.title('Interest Rate vs Loan Amount by Loan Status', fontsize=12)
    plt.xlabel('Interest Rate')
    plt.ylabel('Loan Amount')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('scatter_interest_vs_amount.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Scatter plot saved as 'scatter_interest_vs_amount.png'")

if 'Debit to Income' in loan_data.columns and 'Interest Rate' in loan_data.columns:
    print("\nCreating scatter plot: Debit to Income vs Interest Rate")
    
    plt.figure(figsize=(12, 6))
    
    for status_value, color_value, label_text in [(0, 'blue', 'Fully Paid'), (1, 'orange', 'Default')]:
        subset_data = loan_data[loan_data[target_column] == status_value]
        plt.scatter(subset_data['Debit to Income'], subset_data['Interest Rate'],
                   alpha=0.3, c=color_value, label=label_text, s=10)
    
    plt.title('Debit to Income vs Interest Rate by Loan Status', fontsize=12)
    plt.xlabel('Debit to Income Ratio')
    plt.ylabel('Interest Rate')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('scatter_dti_vs_interest.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Scatter plot saved as 'scatter_dti_vs_interest.png'")
print("="*70)

# ============================================================================
# SECTION 12: DATA CLEANING - IDENTIFYING ISSUES
# ============================================================================

print("\nPerforming data cleaning checks")

# Checking for duplicate rows
duplicate_count = loan_data.duplicated().sum()
print(f"\nChecking for duplicate rows:")
print(f"Duplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    print("Removing duplicate rows")
    loan_data = loan_data.drop_duplicates()
    print(f"Rows after removing duplicates: {len(loan_data)}")

# Checking for columns with single unique value
single_value_columns = [col for col in loan_data.columns if loan_data[col].nunique() == 1]

if single_value_columns:
    print(f"\nColumns with single unique value: {single_value_columns}")
    print("Recommendation: Consider dropping these columns")
else:
    print("\nNo columns with single unique value found")

# Checking for highly correlated features
print("\nChecking for highly correlated features (correlation > 0.95)")
high_correlation_pairs = []
correlation_matrix_abs = loan_data[numerical_cols].corr().abs()

for i in range(len(correlation_matrix_abs.columns)):
    for j in range(i+1, len(correlation_matrix_abs.columns)):
        if correlation_matrix_abs.iloc[i, j] > 0.95:
            col_name_1 = correlation_matrix_abs.columns[i]
            col_name_2 = correlation_matrix_abs.columns[j]
            correlation_value = correlation_matrix_abs.iloc[i, j]
            high_correlation_pairs.append((col_name_1, col_name_2, correlation_value))

if high_correlation_pairs:
    print("Highly correlated feature pairs found:")
    for col1, col2, corr_val in high_correlation_pairs:
        print(f"  {col1} <-> {col2}: {corr_val:.3f}")
    print("Recommendation: Consider dropping one feature from each pair")
else:
    print("No highly correlated features found")

# Identifying features requiring encoding
print("\nIdentifying categorical features requiring encoding:")
for col in categorical_cols:
    unique_count_cat = loan_data[col].nunique()
    print(f"{col}: {unique_count_cat} unique values")
    if unique_count_cat <= 5:
        print(f"  Recommendation: One-Hot Encoding")
    elif unique_count_cat <= 20:
        print(f"  Recommendation: Label Encoding or Target Encoding")
    else:
        print(f"  Recommendation: Consider grouping or Target Encoding")

# Identifying features requiring scaling
print("\nChecking feature ranges for scaling requirement:")
for col in key_numerical_cols:
    min_value_col = loan_data[col].min()
    max_value_col = loan_data[col].max()
    range_value_col = max_value_col - min_value_col
    print(f"{col}: Range [{min_value_col:.2f}, {max_value_col:.2f}] (span: {range_value_col:.2f})")

print("\nRecommendation: Apply StandardScaler or MinMaxScaler to all numerical features")
print("="*70)

# ============================================================================
# SECTION 13: SUMMARY AND KEY INSIGHTS
# ============================================================================

print("\n" + "="*70)
print("KEY INSIGHTS FROM EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\n1. DATASET OVERVIEW:")
print(f"   Total loans: {len(loan_data):,}")
print(f"   Total features: {loan_data.shape[1]}")
print(f"   Numerical features: {len(numerical_cols)}")
print(f"   Categorical features: {len(categorical_cols)}")
print(f"   Missing values: {'None' if loan_data.isnull().sum().sum() == 0 else loan_data.isnull().sum().sum()}")
print(f"   Duplicate rows: {duplicate_count}")

print("\n2. TARGET VARIABLE:")
print(f"   Default rate: {loan_data[target_column].mean()*100:.1f}%")
print(f"   Class imbalance ratio: {imbalance_ratio:.1f}:1")
print("   Issue: Highly imbalanced dataset requiring special handling")

print("\n3. DATA QUALITY:")
print(f"   Missing values: {'None found' if loan_data.isnull().sum().sum() == 0 else 'Found - requires handling'}")
print(f"   Duplicate rows: {'None found' if duplicate_count == 0 else f'{duplicate_count} found'}")
print(f"   Single-value columns: {'None found' if len(single_value_columns) == 0 else f'{len(single_value_columns)} found'}")
print(f"   Highly correlated pairs: {'None found' if len(high_correlation_pairs) == 0 else f'{len(high_correlation_pairs)} found'}")

print("\n4. KEY PREDICTORS:")
if 'Grade' in loan_data.columns:
    worst_grade = loan_data.groupby('Grade')[target_column].mean().idxmax()
    worst_rate = loan_data.groupby('Grade')[target_column].mean().max()
    print(f"   Grade {worst_grade} shows highest default rate: {worst_rate:.1%}")

if 'Interest Rate' in numerical_cols:
    high_rate_threshold = loan_data['Interest Rate'].quantile(0.75)
    high_rate_default_rate = loan_data[loan_data['Interest Rate'] > high_rate_threshold][target_column].mean()
    print(f"   High interest rate loans show {high_rate_default_rate:.1%} default rate")

if 'Debit to Income' in numerical_cols:
    high_dti_default_rate = loan_data[loan_data['Debit to Income'] > 40][target_column].mean()
    print(f"   Loans with DTI > 40% show {high_dti_default_rate:.1%} default rate")

print("\n5. RECOMMENDATIONS FOR MODELING:")
print("   Use stratified train-test split to maintain class distribution")
print("   Apply class_weight='balanced' in classification models")
print("   Focus on Recall metric for catching defaults")
print("   Encode categorical variables using appropriate methods")
print("   Scale numerical features using StandardScaler")
print("   Consider feature engineering for loan-to-income ratio")

print("\n" + "="*70)
print("EDA AND DATA CLEANING COMPLETED")
print("="*70)

# Saving cleaned data
# loan_data.to_csv('cleaned_train.csv', index=False)
# print("\nCleaned data saved as 'cleaned_train.csv'")

print("\nAll visualizations saved successfully")
print("Process complete")
