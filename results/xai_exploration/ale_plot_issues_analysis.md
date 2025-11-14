# ALE Plot Issues Analysis

## Problem Summary

The ALE plots show several problematic patterns:
1. **Extreme outliers** causing plots to have strange scales and clipping
2. **Categorical feature** (`bransch_sni071_konv`) being treated as numeric
3. **Step functions** in plots that don't make economic sense

---

## Issue 1: Extreme Outliers in Financial Features

### Affected Features:

#### 1. `ratio_cash_liquidity`
- **Range**: -117 to 16,047.5
- **99th percentile**: 6.66
- **Problem**: 82 extreme outliers above mean + 5*std
- **Cause**: Division creating extreme ratios when denominator is very small
- **Impact**: ALE plot dominated by outlier range, obscuring normal behavior

#### 2. `rr01_ntoms_yoy_abs` (Revenue YoY absolute change)
- **Range**: -11.45M to 5.70M
- **99th percentile**: 60,751
- **Problem**: 354 extreme high outliers
- **Cause**: Absolute changes in revenue vary wildly with company size
- **Impact**: Step function appearance in ALE plot

#### 3. `dpo_days_yoy_diff` (Days payable outstanding change)
- **Range**: -468,660 to 603,162
- **99th percentile**: 241
- **Problem**: 60 extreme outliers
- **Cause**: Likely data errors or extreme business changes
- **Impact**: Unreasonable values (can't have >600k days payable!)

### Recommended Solutions:

**Option A: Winsorizing (RECOMMENDED)**
```python
def winsorize_features(df, features, lower=0.01, upper=0.99):
    """Cap extreme values at specified percentiles"""
    for feature in features:
        q_low = df[feature].quantile(lower)
        q_high = df[feature].quantile(upper)
        df[feature] = df[feature].clip(lower=q_low, upper=q_high)
    return df
```

Apply to: `ratio_cash_liquidity`, `rr01_ntoms_yoy_abs`, `dpo_days_yoy_diff`

**Option B: Log transformation**
For highly skewed features like `ratio_cash_liquidity`, use `log(1 + x)` transformation

**Option C: Robust scaling in feature engineering**
Replace division-based ratios with rank-based or quantile-based features

---

## Issue 2: Categorical Feature Treated as Numeric

### `bransch_sni071_konv` (Swedish SNI Industry Codes)

**Current state:**
- Type: `float64`
- Unique values: 814 different industry codes
- Range: 0 to 99,000
- Examples: 70220 (Management consulting), 56100 (Restaurants), 68201 (Renting property)

**Problems:**
1. **Treated as continuous**: LightGBM is splitting on numeric values (e.g., industry < 50000)
2. **No categorical handling**: Not declared in `categorical_feature` parameter
3. **Logistic regression**: Treats as single continuous feature (meaningless)
4. **ALE plot meaningless**: Plotting a line for categorical codes makes no sense

**Current configuration:**
- `CATEGORICAL_COLS` defined in `config.py` but **NOT passed to LightGBM**
- `DEFAULT_PARAMS` in `train.py` has **no categorical_feature parameter**

### Recommended Solutions:

**Option A: One-Hot Encoding (Logit compatibility)**
```python
# In feature engineering
from sklearn.preprocessing import OneHotEncoder

# Group rare industries (< 1% of data) into "Other"
industry_counts = df['bransch_sni071_konv'].value_counts()
rare_industries = industry_counts[industry_counts < len(df) * 0.01].index
df['bransch_sni071_konv'] = df['bransch_sni071_konv'].replace(rare_industries, -1)

# One-hot encode
encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
industry_dummies = encoder.fit_transform(df[['bransch_sni071_konv']])
```

**Pros**: Works for both models, interpretable
**Cons**: Adds ~50-100 features (increases dimensionality)

**Option B: LightGBM Native Categorical + Target Encoding for Logit** (RECOMMENDED)
```python
# For LightGBM: Declare as categorical
# In train.py, modify fit call:
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    categorical_feature=['bransch_sni071_konv'],  # ADD THIS
    callbacks=[early_stopping, log_evaluation]
)

# For Logistic Regression: Target encoding
from category_encoders import TargetEncoder
encoder = TargetEncoder()
X_train['bransch_sni071_konv_encoded'] = encoder.fit_transform(
    X_train['bransch_sni071_konv'], y_train
)
```

**Pros**: Best of both worlds, preserves interpretability
**Cons**: Requires different preprocessing pipelines for each model

**Option C: Remove from model** (if not important)
Check feature importance - if low, simply drop it

---

## Issue 3: ALE Plot Implementation for Current Data

### Problems:

1. **Quantile-based grid breaks on outliers**: With extreme values, quantiles are unstable
2. **Categorical features**: Should not be plotted with ALE
3. **Missing value handling**: Not explicitly handled in ALE computation

### Recommended Changes:

#### 1. Filter features for ALE plots
```python
# Only plot continuous, non-outlier-dominated features
features_for_ale = [
    f for f in top_15_features
    if f not in ['bransch_sni071_konv']  # Exclude categorical
    and X_val[f].std() / X_val[f].mean() < 10  # Exclude extreme outliers
]
```

#### 2. Add robust ALE computation
```python
def compute_ale_1d_robust(model, X, feature, grid_size=50, predict_fn=None,
                          winsorize_pct=0.99):
    """ALE with outlier protection"""
    X_work = X.copy()
    feat_values = X_work[feature].dropna()

    # Winsorize before computing grid
    q_high = feat_values.quantile(winsorize_pct)
    q_low = feat_values.quantile(1 - winsorize_pct)
    feat_values_trimmed = feat_values.clip(lower=q_low, upper=q_high)

    # Rest of ALE computation...
```

#### 3. Add categorical indicator to features
```python
# In config.py or feature engineering
FEATURE_METADATA = {
    'bransch_sni071_konv': {'type': 'categorical', 'plot_type': 'bar'},
    'ratio_cash_liquidity': {'type': 'continuous', 'winsorize': 0.99},
    'dpo_days_yoy_diff': {'type': 'continuous', 'winsorize': 0.995},
    # ...
}
```

---

## Immediate Action Items

### Priority 1 (High Impact):
1. **Fix categorical handling**:
   - Add `categorical_feature=['bransch_sni071_konv']` to LightGBM fit
   - Add target encoding for Logistic Regression
   - Exclude from ALE plots

2. **Winsorize extreme features** in feature engineering:
   - `ratio_cash_liquidity`: clip at 99th percentile
   - `rr01_ntoms_yoy_abs`: clip at 1st/99th percentile
   - `dpo_days_yoy_diff`: clip at 1st/99th percentile (or investigate data errors)

### Priority 2 (Medium Impact):
3. **Update ALE plot selection**:
   - Exclude categorical features
   - Add feature type checking
   - Add winsorizing to ALE computation

4. **Investigate data quality**:
   - `dpo_days_yoy_diff` values >600k days are impossible → data errors?
   - Check original data source for these extreme values

### Priority 3 (Nice to Have):
5. **Add feature metadata system** to config
6. **Create separate visualizations** for categorical features (e.g., SHAP bar plots by category)

---

## Long-term Recommendations

1. **Feature engineering improvements**:
   - Replace absolute changes (`_yoy_abs`) with percentage changes or scaled versions
   - Add robust transformations for ratio features
   - Implement systematic outlier detection in data pipeline

2. **Model training improvements**:
   - Properly declare all categorical features
   - Consider separate preprocessing pipelines for LightGBM vs Logit
   - Add data validation checks before training

3. **XAI improvements**:
   - Create feature-type-aware plotting functions
   - Add automatic outlier detection for ALE plots
   - Implement categorical-specific XAI methods (e.g., category-wise SHAP)

---

## Recommended Immediate Fix for Notebook

For the XAI notebook, add this before ALE plots:

```python
# Filter features suitable for ALE plots
continuous_features = [
    f for f in top_15_features
    if f not in ['bransch_sni071_konv']  # Categorical
]

# Winsorize extreme features for ALE
X_val_ale = X_val.copy()
winsorize_features = {
    'ratio_cash_liquidity': (0.01, 0.99),
    'rr01_ntoms_yoy_abs': (0.01, 0.99),
    'dpo_days_yoy_diff': (0.01, 0.99),
}

for feature, (lower, upper) in winsorize_features.items():
    if feature in X_val_ale.columns:
        q_low = X_val_ale[feature].quantile(lower)
        q_high = X_val_ale[feature].quantile(upper)
        X_val_ale[feature] = X_val_ale[feature].clip(lower=q_low, upper=q_high)

# Use X_val_ale for ALE computation instead of X_val
```

This will immediately improve plot quality without requiring feature engineering changes.
