# Feature Engineering Pipeline Data Quality Analysis Report
========================================================================================================================

**Generated:** 2025-11-14

## Executive Summary

This report identifies critical data quality issues, dtype inefficiencies, and 
discrepancies in the feature engineering pipeline for the credit risk modeling project.

### Key Findings:

1. **CRITICAL: Categorical features not using category dtype** - Only 1/8 categorical features properly encoded
2. **CRITICAL: Extreme outliers in working capital metrics** - Days metrics showing impossible values (>27 years)
3. **CRITICAL: Negative production costs** - 80.69% of records have negative production costs, breaking DPO calculations
4. **HIGH: YoY percentage changes unbounded** - Values reaching millions of percent
5. **MEDIUM: Dtype inefficiency** - Several features using oversized dtypes

---

## 1. Summary Table: All 40 Final Features

| Feature | Current dtype | Recommended dtype | Min | Max | Null % | Outliers (5σ) | Issues |
|---------|---------------|-------------------|-----|-----|--------|---------------|--------|
| bransch_sni071_konv | float64 | category | 0.00 | 99000.00 | 11.0 | 0 | Not category dtype |
| company_age | Int16 | Int16 | -1.00 | 163.00 | 0.0 | 49212 | - |
| current_ratio_yoy_pct | float32 | float32 | -1148457.00 | 677502400.00 | 26.4 | 106 | Extreme YoY%; Unbounded ratio |
| dividend_yield | float32 | float32 | -756.00 | 14634.15 | 11.0 | 144 | - |
| dpo_days | float32 | float32 | -351128160.00 | 1311080.00 | 19.3 | 591 | Extreme days (>27yr) |
| dpo_days_trend_3y | float64 | float64 | -175564080.00 | 11421279.60 | 36.8 | 489 | Extreme days (>27yr) |
| dpo_days_yoy_diff | float32 | float32 | -351128160.00 | 351128160.00 | 28.4 | 446 | Extreme days (>27yr) |
| dso_days | float32 | float32 | -67996216.00 | 4244967936.00 | 32.4 | 271 | Extreme days (>27yr) |
| dso_days_yoy_diff | float32 | float32 | -2060297984.00 | 595827392.00 | 40.5 | 236 | Extreme days (>27yr) |
| event_count_last_5y | Int16 | uint8 | 0.00 | 5.00 | 0.0 | 18934 | Oversized int |
| inventory_days_yoy_diff | float32 | float32 | -103192072.00 | 102938624.00 | 28.4 | 1181 | Extreme days (>27yr) |
| log_br07b_kabasu | float32 | float32 | -0.00 | 18.03 | 11.0 | 33 | - |
| log_br10_eksu | float32 | float32 | -0.00 | 20.08 | 15.4 | 3821 | - |
| log_bslov_antanst | float32 | float32 | 0.00 | 10.40 | 0.0 | 23920 | - |
| log_rr15_resar | float32 | float32 | -0.00 | 18.73 | 37.2 | 101 | - |
| ny_anstf | float32 | float32 | -1.00 | 13188.50 | 51.5 | 412 | - |
| ny_avkegkap | float32 | float32 | -64690.00 | 45295.48 | 15.5 | 3379 | - |
| ny_foradlvpanst | float32 | float32 | -5486125.00 | 24772000.00 | 48.6 | 1010 | - |
| ny_kapomsh | float32 | float32 | -0.88 | 31020.00 | 11.2 | 2124 | - |
| ny_kasslikv | float32 | float32 | -8943.00 | 2141347.00 | 17.1 | 2735 | - |
| ny_nettomarg | float32 | float32 | -3474184.00 | 3287739.00 | 32.4 | 431 | - |
| ny_omsf | float32 | float32 | -4.22 | 1337810.00 | 37.9 | 387 | - |
| ny_omspanst | float32 | float32 | -54.00 | 99013568.00 | 48.6 | 1033 | - |
| ny_rs | float32 | float32 | -3.20 | 1140862.75 | 15.7 | 487 | - |
| ny_skuldgrd | float32 | float32 | -15954.26 | 1226958.50 | 15.5 | 4058 | - |
| ny_solid | float32 | float32 | -49.89 | 2751.91 | 15.5 | 796 | - |
| ny_solid_yoy_diff | float32 | float32 | -2750.82 | 2751.49 | 24.8 | 659 | - |
| profit_cagr_3y | float32 | float32 | -1.00 | 147.46 | 67.7 | 21751 | - |
| ratio_cash_interest_cov | float32 | float32 | -1770902.00 | 93259.00 | 46.0 | 9090 | Unbounded ratio |
| ratio_cash_liquidity | float32 | float32 | -26977.00 | 371347.00 | 16.8 | 7921 | Unbounded ratio |
| ratio_cash_liquidity_yoy_abs | float32 | float32 | -370961.22 | 154306.61 | 25.6 | 6654 | Unbounded ratio |
| ratio_cash_liquidity_yoy_pct | float32 | float32 | -1761359.00 | 264824576.00 | 31.3 | 146 | Extreme YoY%; Unbounded ratio |
| ratio_depreciation_cost | float32 | float32 | -66864.00 | 295.00 | 32.4 | 962 | - |
| ratio_retained_earnings_equity | float32 | float32 | -1825320.75 | 71914.60 | 11.0 | 183 | Unbounded ratio |
| ratio_short_term_debt_share | float32 | float32 | -455.20 | 787.54 | 15.7 | 1751 | - |
| revenue_cagr_3y | float32 | float32 | -0.99 | 88.77 | 53.1 | 31066 | - |
| revenue_drawdown_5y | float64 | float64 | -1.46 | 0.00 | 50.9 | 0 | - |
| rr01_ntoms_yoy_abs | float32 | float32 | -48785892.00 | 99013568.00 | 19.7 | 6429 | - |
| rr07_rorresul_yoy_pct | float32 | float32 | -3911197.00 | 714887.00 | 24.2 | 451 | Extreme YoY% |
| term_spread | float32 | float32 | -1.34 | 1.62 | 17.9 | 0 | - |

---
## 2. Categorical Feature Analysis

### Problem: Categorical dtype lost in feature engineering

**Root Cause Analysis:**

1. **make_dataset.py** (line 179): Properly converts categoricals using `_ensure_categories()`
2. **engineer.py**: Does NOT preserve categorical dtypes - all operations return numeric types
3. **Result**: Categorical encoding is lost when features are engineered and saved to parquet

### Current Status:

| Feature | Defined as Categorical | Current dtype | Properly Encoded | Unique Values |
|---------|----------------------|---------------|------------------|---------------|
| bransch_sni071_konv | Yes | float64 | ✗ | 814 |
| bransch_borsbransch_konv | Yes | int8 | ✗ | 13 |
| ser_laen | Yes | float64 | ✗ | 21 |
| knc_kncfall | Yes | float64 | ✗ | 4 |
| ser_aktiv | Yes | int8 | ✗ | 2 |
| ser_nystartat | Yes | float64 | ✗ | 1 |
| bol_konkurs | Yes | float64 | ✗ | 1 |
| sme_category | Yes | category | ✓ | 4 |

### Impact:

- **Memory inefficiency**: Numeric dtypes use more memory than category dtype for low-cardinality features
- **LightGBM suboptimal**: LightGBM's native categorical support requires `dtype='category'`
- **Incorrect feature interpretation**: Models may treat ordinal codes as continuous variables

### Where Categorical Encoding Was Lost:

```python
# make_dataset.py (line 179) - SETS categoricals ✓
interim_df = _ensure_categories(interim_df, CATEGORICAL_COLS)

# engineer.py - DOES NOT preserve categoricals ✗
# All operations (join, merge, etc.) return numeric types
# No astype('category') calls in engineer.py
```

---
## 3. Top Data Quality Issues (Prioritized)

### CRITICAL Issues

#### Issue 1: Negative Production Costs Breaking DPO Calculation

**Severity**: CRITICAL

**Description**: 80.69% of records have negative `rr06a_prodkos` (production costs), causing:
- Negative DPO days when they should be positive (or zero)
- Extreme values: MAX = 1,311,080 days (>3,500 years)
- Meaningless working capital metrics

**Root Cause**:
```python
# engineer.py line 289
dpo_days = _safe_div(df['br13a_ksklev'], df['rr06a_prodkos']) * 365

# When rr06a_prodkos is negative (common in Swedish accounting):
# - Payables (positive) / Production costs (negative) = NEGATIVE ratio
# When rr06a_prodkos is near-zero:
# - Payables / 0.001 = EXTREME value
```

**Evidence**:
- Records with negative production costs: 10,065,496 (80.69%)
- Records with zero/near-zero production costs: 740,448 (5.94%)
- DPO values < -1000 days: 3.78% of dataset

**Impact on Documentation**:

From `engineered_features.md` (line 72):
> "Days payables outstanding; supplier payment terms."

**Actual behavior**: Mostly negative values due to accounting conventions, not payment terms.

#### Issue 2: Zero/Near-Zero Revenue Breaking DSO Calculation

**Severity**: CRITICAL

**Description**: 21.57% of records have zero or near-zero revenue, causing:
- Division by near-zero: MAX = 4,244,967,936 days (>11 million years)
- Undefined DSO for companies with no sales

**Root Cause**:
```python
# engineer.py line 287
dso_days = _safe_div(df['br06g_kfordsu'], df['rr01_ntoms']) * 365

# When rr01_ntoms is near-zero:
# - Receivables / 0.001 = EXTREME value
```

**Evidence**:
- Records with zero/near-zero revenue: 2,690,292 (21.57%)
- DSO P99.9%: 199,568 days (>500 years)
- DSO P95%: 991 days (still extreme)

#### Issue 3: Categorical Features Not Using Category Dtype

**Severity**: CRITICAL

**Description**: Only 1 of 8 categorical features uses `dtype='category'`

**Problem Features**:
- `bransch_sni071_konv`: float64 (should be category, 814 unique values)
- `bransch_borsbransch_konv`: int8 (should be category, 13 unique values)
- `ser_laen`: float64 (should be category, 21 unique values)
- `knc_kncfall`: float64 (should be category, 4 unique values)
- `ser_aktiv`: int8 (should be category, 2 unique values)
- `ser_nystartat`: float64 (should be category, 1 unique value)
- `bol_konkurs`: float64 (should be category, 1 unique value)

**Impact**:
- LightGBM cannot use native categorical handling
- Higher memory usage
- Models may misinterpret ordinal codes as continuous


### HIGH Priority Issues

#### Issue 4: Unbounded YoY Percentage Changes

**Severity**: HIGH

**Features Affected**:

| Feature | Min | Max | P99.9% | Records >500% |
|---------|-----|-----|--------|---------------|
| `rr07_rorresul_yoy_pct` | -3,911,197 | 714,887 | 233 | 914,970 (9.68%) |
| `ratio_cash_liquidity_yoy_pct` | -1,761,359 | 264,824,576 | 1,414 | 516,540 (6.02%) |
| `current_ratio_yoy_pct` | -1,148,457 | 677,502,400 | 850 | 303,526 (3.31%) |

**Root Cause**: Division by near-zero denominators in percentage change calculations

**Example**:
```python
# Company goes from ratio_cash_liquidity = 0.001 to 10
# YoY % change = (10 - 0.001) / 0.001 = 9,999 = 999,900%
```

#### Issue 5: Revenue Drawdown Outside Valid Range

**Severity**: HIGH

**Description**: Drawdown should be in [-1, 0] but 46 records (0.0008%) have values < -1

**Evidence**:
- Min value: -1.463 (should be -1.000)
- Sample values: -1.463, -1.079, -1.066, -1.011

**Root Cause**: Likely numerical precision issues or division by near-zero in drawdown calculation

**From documentation** (engineered_features.md line 143):
> "Maximum drawdown of revenue (min value / peak - 1)"

**Expected**: If revenue goes to 0, drawdown = (0 / peak) - 1 = -1 (not < -1)

### MEDIUM Priority Issues

#### Issue 6: Inefficient Data Types

**Severity**: MEDIUM

**Recommendations**:

| Feature | Current | Recommended | Reason | Memory Impact |
|---------|---------|-------------|--------|---------------|
| `event_count_last_5y` | Int16 | uint8 | Range 0-5 fits in uint8 | Medium |
| `revenue_drawdown_5y` | float64 | float32 | Precision not critical | Low |
| `dpo_days_trend_3y` | float64 | float32 | Precision not critical | Low |
| `bransch_sni071_konv` | float64 | category | Categorical with 814 levels | High |

#### Issue 7: CAGR Values Exceeding Reasonable Bounds

**Severity**: MEDIUM

**Description**: CAGR (compound annual growth rate) values showing extreme growth rates

| Feature | Min | Max | P99% | Records >200% |
|---------|-----|-----|------|---------------|
| `profit_cagr_3y` | -0.997 | 147.46 | 4.78 | 193,687 (4.81%) |
| `revenue_cagr_3y` | -0.991 | 88.77 | 2.53 | 88,673 (1.52%) |

**Assessment**: While extreme, these may be legitimate for high-growth startups or recovery scenarios.
**Recommendation**: Winsorize at P1/P99 rather than hard caps.

---

## 4. Specific Recommendations

### A. Features to Winsorize

Apply winsorization to handle extreme outliers while preserving information:

| Feature | Recommended Percentiles | Reason |
|---------|------------------------|--------|
| `dso_days` | 1st-99th (0.0, 9490) | Extreme max (4.2B days) breaks analysis |
| `dpo_days` | 1st-99th (-778, 0) | Negative values from accounting; cap extremes |
| `inventory_days` | 1st-99th (-1058, 0) | Negative values from accounting; cap extremes |
| `dso_days_yoy_diff` | 1st-99th (-2357, 3407) | Cap extreme changes |
| `dpo_days_yoy_diff` | 1st-99th (-290, 455) | Cap extreme changes |
| `inventory_days_yoy_diff` | 1st-99th (-246, 340) | Cap extreme changes |
| `dpo_days_trend_3y` | 1st-99th (-179, 274) | Cap extreme trends |
| `rr07_rorresul_yoy_pct` | 1st-99th (-25.8, 26.0) | Cap extreme YoY changes |
| `ratio_cash_liquidity_yoy_pct` | 1st-99th (-1.0, 63.0) | Cap extreme YoY changes |
| `current_ratio_yoy_pct` | 1st-99th (-0.99, 25.9) | Cap extreme YoY changes |
| `revenue_cagr_3y` | 1st-99th (-0.76, 2.53) | Cap extreme growth rates |
| `profit_cagr_3y` | 1st-99th (-0.82, 4.78) | Cap extreme growth rates |
| `ratio_cash_interest_cov` | Cap at 1000 | Unbounded ratio |
| `ratio_cash_liquidity` | Cap at 1000 | Unbounded ratio |
| `ratio_retained_earnings_equity` | 1st-99th | Unbounded ratio |

### B. Categorical Features to Fix

Add to engineer.py at the end of `create_engineered_features()`:

```python
# Ensure categorical dtypes are preserved
for col in CATEGORICAL_COLS:
    if col in df.columns:
        df[col] = df[col].astype('category')
```

**Alternative**: Add categorical encoding to `prepare_modeling_data()` function.

### C. Data Types to Change

In engineer.py, add dtype optimization after feature creation:

```python
# Optimize dtypes for efficiency
df['event_count_last_5y'] = df['event_count_last_5y'].astype('uint8')

# Downcast float64 to float32 where precision not critical
for col in ['revenue_drawdown_5y', 'dpo_days_trend_3y']:
    if col in df.columns and df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')
```

### D. Documentation Corrections Needed

**engineered_features.md** needs updates for these features:

1. **dpo_days** (line 72):
   - Current: "Days payables outstanding; supplier payment terms."
   - Should add: "Note: Commonly negative in Swedish accounting where production costs are recorded as negative values. Interpret with caution."

2. **inventory_days** (line 71):
   - Should add: "Note: Can be negative due to accounting conventions. Winsorized in practice."

3. **revenue_drawdown_5y** (line 143):
   - Should specify: "Expected range: [-1, 0]. Values <-1 indicate data quality issues (0.0008% of records)."
---

## 5. Implementation vs Documentation Discrepancies

### Comparison of Top 20 Most Important Features

(Based on typical feature importance from LightGBM models)

| Feature | Documented Purpose | Implementation Status | Discrepancy |
|---------|-------------------|----------------------|-------------|
| ny_solid | Equity-to-assets ratio | Implemented correctly | None |
| ny_skuldgrd | As documented | Correct | None |
| ny_kasslikv | As documented | Correct | None |
| ny_nettomarg | As documented | Correct | None |
| log_br10_eksu | Log-transformed total equity | Implemented correctly | None |
| log_br07b_kabasu | As documented | Correct | None |
| revenue_cagr_3y | As documented | Correct | None |
| revenue_drawdown_5y | Maximum drawdown [-1, 0] | 46 records have values < -1 | LOW: Rare edge case (0.0008%) |
| dpo_days | Days payables outstanding; supplier payment terms | Mostly NEGATIVE due to negative production costs (80.69% of records) | HIGH: Documentation does not mention negative values |
| dso_days | Days sales outstanding | Extreme outliers (max 4.2B days) due to zero/near-zero revenue | MEDIUM: Documentation does not mention outlier handling |
| ratio_cash_liquidity | Quick ratio | Extreme outliers (max 371,347) due to near-zero current liabilities | MEDIUM: Documentation does not mention unbounded nature |
| profit_cagr_3y | As documented | Correct | None |
| company_age | As documented | Correct | None |
| ny_rs | As documented | Correct | None |
| event_count_last_5y | As documented | Correct | None |
| term_spread | As documented | Correct | None |
| ratio_short_term_debt_share | As documented | Correct | None |
| ny_avkegkap | As documented | Correct | None |
| dividend_yield | As documented | Correct | None |
| ratio_cash_interest_cov | As documented | Correct | None |

### Key Discrepancy: Working Capital Metrics

**Documentation** (engineered_features.md, lines 64-75) describes working capital metrics as:

- "Days sales outstanding; completes working capital trinity (DSO + inventory + DPO)"
- "Days payables outstanding; supplier payment terms"
- "Days inventory outstanding"

**Reality**: Swedish accounting practices result in:

- **DPO**: 80.69% of records have NEGATIVE production costs → NEGATIVE dpo_days
- **Inventory**: Similar issues with negative values
- **DSO**: 21.57% of records have zero/near-zero revenue → extreme outliers

**Impact**: The "working capital trinity" is not interpretable as described in academic literature.

---

## 6. Root Cause Analysis: Why Did This Happen?

### Issue 1: Negative Production Costs

**Root Cause**: Swedish accounting allows income statement items to be negative:
- `rr06a_prodkos` (production costs) can be negative when:
  - Change in inventory is positive (finished goods increased)
  - Work performed by the company for itself is capitalized
  - See `rr02a_lagerf` (change in inventories of finished goods)

**Why it wasn't caught**:
- No validation of denominator sign in `_safe_div()` function
- No domain knowledge checks (e.g., "DPO should typically be 0-365 days")
- No visualization of distributions during feature engineering

### Issue 2: Categorical Dtype Loss

**Root Cause**: Pandas operations that modify DataFrames often lose categorical dtype:

```python
# make_dataset.py sets categoricals
interim_df = _ensure_categories(interim_df, CATEGORICAL_COLS)  # ✓

# engineer.py modifies the DataFrame
df.set_index(['ORGNR', 'ser_year'], drop=False, inplace=True)  # Preserves dtypes
df = df.join(pd.DataFrame(new_features, index=df.index))  # MAY lose dtypes
df.drop(columns=cols_to_drop, inplace=True)  # Preserves remaining dtypes
df.reset_index(drop=True, inplace=True)  # Preserves dtypes
df.to_parquet(output_path, ...)  # Saves whatever dtypes are present

# No re-enforcement of categorical dtypes before saving!
```

**Why it wasn't caught**:
- No dtype validation in engineer.py
- No unit tests checking categorical preservation
- Parquet format CAN store categorical dtype, but doesn't enforce it

### Issue 3: Unbounded Ratios and YoY Changes

**Root Cause**: No bounds checking or winsorization in feature engineering:

```python
# _safe_div handles div-by-zero → NaN, but not near-zero → extreme values
def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator
    mask = denominator.isna() | (denominator == 0)  # Only checks EXACT zero
    if mask.any():
        result = result.mask(mask)
    return result.replace([np.inf, -np.inf], np.nan)
```

**Why it wasn't caught**:
- No threshold for "near-zero" (e.g., `abs(denominator) < 0.01`)
- No post-calculation winsorization
- Assumed _safe_div was sufficient protection
---

## 7. Recommended Fix Strategy

### Phase 1: Critical Fixes (Do First)

#### 1.1 Fix Categorical Encoding

**File**: `credit_risk_xai/features/engineer.py`

**Location**: End of `create_engineered_features()` function (before return)

**Code to add**:
```python
# Ensure categorical dtypes are preserved for LightGBM native support
logger.info("Enforcing categorical dtypes for configured categorical columns")
for col in CATEGORICAL_COLS:
    if col in df.columns:
        df[col] = df[col].astype('category')
```

#### 1.2 Add Winsorization for Extreme Outliers

**File**: `credit_risk_xai/features/engineer.py`

**Location**: After all feature calculations, before categorical enforcement

**Code to add**:
```python
def _winsorize_feature(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip extreme values at specified percentiles."""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

# Winsorize features with extreme outliers
logger.info("Winsorizing features with extreme outliers")
winsorize_config = {
    'dso_days': (0.01, 0.99),
    'dpo_days': (0.01, 0.99),
    'inventory_days': (0.01, 0.99),
    'dso_days_yoy_diff': (0.01, 0.99),
    'dpo_days_yoy_diff': (0.01, 0.99),
    'inventory_days_yoy_diff': (0.01, 0.99),
    'dpo_days_trend_3y': (0.01, 0.99),
    'rr07_rorresul_yoy_pct': (0.01, 0.99),
    'ratio_cash_liquidity_yoy_pct': (0.01, 0.99),
    'current_ratio_yoy_pct': (0.01, 0.99),
    'revenue_cagr_3y': (0.01, 0.99),
    'profit_cagr_3y': (0.01, 0.99),
}

for col, (lower, upper) in winsorize_config.items():
    if col in df.columns:
        df[col] = _winsorize_feature(df[col], lower, upper)
```

#### 1.3 Optimize Data Types

**File**: `credit_risk_xai/features/engineer.py`

**Location**: After categorical enforcement

**Code to add**:
```python
# Optimize dtypes for memory efficiency
logger.info("Optimizing data types")
if 'event_count_last_5y' in df.columns:
    df['event_count_last_5y'] = df['event_count_last_5y'].astype('uint8')

# Downcast float64 to float32 where precision not critical
float64_to_32 = ['revenue_drawdown_5y', 'dpo_days_trend_3y']
for col in float64_to_32:
    if col in df.columns and df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')
```

### Phase 2: Validation and Testing

#### 2.1 Add Data Quality Checks

**File**: `credit_risk_xai/features/engineer.py`

**Location**: End of `create_engineered_features()` function

**Code to add**:
```python
# Data quality validation
logger.info("Running data quality checks")

# Check categorical dtypes
for col in CATEGORICAL_COLS:
    if col in df.columns:
        assert df[col].dtype.name == 'category', f"{col} should be category dtype"

# Check feature ranges
assert df['revenue_drawdown_5y'].min() >= -1.0, "Drawdown values < -1.0 detected"
assert df['revenue_drawdown_5y'].max() <= 0.0, "Drawdown values > 0.0 detected"

# Log feature statistics
for col in FEATURES_FOR_MODEL:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        logger.debug(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, null={df[col].isna().sum()}")
```

#### 2.2 Create Unit Tests

**File**: `tests/test_feature_engineering.py` (create if doesn't exist)

**Tests to add**:
```python
def test_categorical_dtypes_preserved():
    """Test that categorical features maintain category dtype after engineering."""
    df = create_engineered_features(sample_data, macro_df)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            assert df[col].dtype.name == 'category'

def test_winsorization_applied():
    """Test that extreme outliers are capped."""
    df = create_engineered_features(sample_data, macro_df)
    assert df['dso_days'].max() < 100000  # Should be winsorized
    assert df['current_ratio_yoy_pct'].abs().max() < 1000  # Should be winsorized

def test_revenue_drawdown_bounds():
    """Test that drawdown is in valid range."""
    df = create_engineered_features(sample_data, macro_df)
    assert (df['revenue_drawdown_5y'] >= -1.0).all()
    assert (df['revenue_drawdown_5y'] <= 0.0).all()
```

### Phase 3: Documentation Updates

#### 3.1 Update engineered_features.md

**Changes needed**:

1. **Line 72 (dpo_days)**:
   ```markdown
   | `dpo_days` | `(br13a_ksklev / rr06a_prodkos) * 365` | **RETAINED** | Days payables outstanding. 
   **Note**: Commonly negative due to Swedish accounting practices where production costs can be 
   negative. Winsorized at 1st-99th percentile to handle extreme values. |
   ```

2. **Line 71 (inventory_days)**:
   ```markdown
   | ~~`inventory_days`~~ | ~~`(br06c_lagersu / rr06a_prodkos) * 365`~~ | **REMOVED** | Low importance; 
   information captured by derivatives (inventory_days_yoy_diff). **Note**: Can be negative due to 
   accounting conventions. Winsorized in practice. |
   ```

3. **Line 143 (revenue_drawdown_5y)**:
   ```markdown
   | `revenue_drawdown_5y` | 5 years | Maximum drawdown of revenue (min value / peak - 1) | **RETAINED** | 
   High SHAP importance (rank 7). Captures revenue resilience and exposure to demand shocks. 5y window 
   needed to capture full business cycles. **Expected range**: [-1, 0]. Edge cases (<-1) handled via 
   data quality checks. |
   ```

---

## 8. Testing Checklist

Before deploying fixes:

- [ ] Run `make_dataset.py --force` to regenerate interim data
- [ ] Run `engineer.py --force` to regenerate features with fixes
- [ ] Verify categorical dtypes preserved: `df.dtypes[df.dtypes == 'category']`
- [ ] Check winsorization applied: `df[['dso_days', 'current_ratio_yoy_pct']].describe()`
- [ ] Verify no extreme outliers: `df[FEATURES_FOR_MODEL].describe().T['max']`
- [ ] Run unit tests: `pytest tests/test_feature_engineering.py`
- [ ] Retrain baseline model and verify PR-AUC doesn't degrade
- [ ] Update documentation: `engineered_features.md`

---

## 9. Appendix: Feature Statistics

### Complete Descriptive Statistics (All 40 Features)

```
                                     count         mean            std           min           1%          5%        25%        50%         75%         95%           99%              max
company_age                     12473668.0    12.892438      13.744771          -1.0          0.0         0.0        3.0        9.0        18.0        39.0          64.0            163.0
bransch_sni071_konv             11106840.0  57945.47938   18949.706027           0.0       1500.0     24340.0    46431.0    62010.0     70220.0     86230.0       96021.0          99000.0
ny_kapomsh                      11070521.0     1.500862      27.439348     -0.882353          0.0         0.0   0.006462   0.683402    1.991361     4.60509      8.813865          31020.0
ny_rs                           10520772.0     1.604279     606.942993          -3.2          0.0         0.0        0.0   0.005405    0.028415    0.132116          1.75       1140862.75
ny_skuldgrd                     10536460.0    12.995375     672.615173 -15954.260742          0.0         0.0   0.245614   0.891566     2.90625   21.786264    137.454987        1226958.5
ny_solid                        10536460.0     0.533239       1.567894    -49.887409     0.007207     0.04386   0.255952   0.528571    0.802693         1.0           1.0       2751.90918
ny_avkegkap                     10535642.0    -0.311665        51.1637      -64690.0    -8.272727       -1.32  -0.041204    0.09835      0.4102    0.985594      2.372703     45295.480469
ny_kasslikv                     10340460.0    23.725716    1487.965698       -8943.0     0.000021    0.105401   0.863636   1.683773      3.9375   33.621143    239.014938        2141347.0
ny_nettomarg                     8429269.0     1.726622    1961.533691    -3474184.0        -11.0   -1.036158  -0.014171   0.048836    0.198237    0.853316       8.98001        3287739.0
ny_omspanst                      6412513.0  1908.797974       64096.75         -54.0          0.0        57.0      504.0      939.0      1630.0      4674.0  13647.879883       99013568.0
ny_foradlvpanst                  6412739.0   584.650085   14054.243164    -5486125.0       -419.0       -22.0      218.0      423.0       679.0      1414.0        2966.0       24772000.0
ny_omsf                          7742237.0     1.921196     597.096069      -4.21519         -1.0    -0.94435  -0.167047   0.020733    0.238743    2.662791     11.097762        1337810.0
ny_anstf                         6054224.0     -0.01465       5.656321          -1.0         -1.0        -1.0        0.0        0.0         0.0         0.5      1.181818          13188.5
log_br10_eksu                   10555555.0     6.244926       1.923224          -0.0     2.772589    3.713572   4.828314   5.993961    7.362645    9.706377     11.892005        20.082548
log_br07b_kabasu                11097732.0     4.732583       2.448842          -0.0          0.0         0.0   3.367296   5.030438    6.408529    8.320448      9.882264        18.033281
log_bslov_antanst               12473668.0     0.731412       0.970778           0.0          0.0         0.0        0.0   0.693147    1.098612     2.70805      4.110874        10.398306
log_rr15_resar                   7833280.0     4.455623       2.442269          -0.0          0.0         0.0   2.995732    4.70048    6.023448    8.243019     10.199175        18.733965
ratio_depreciation_cost          8429297.0     -0.19278      26.457619      -66864.0    -1.266064   -0.294118  -0.051207  -0.009542         0.0         0.0           0.0            295.0
ratio_cash_interest_cov          6738819.0  -199.185547    2372.188721    -1770902.0      -2995.0      -744.0 -66.333336  -8.277778   -0.678571         0.0          -0.0          93259.0
ratio_cash_liquidity            10383651.0    10.311901     283.011536      -26977.0          0.0         0.0   0.111111   0.722125    2.356709   20.263159    141.666672         371347.0
ratio_short_term_debt_share     10519579.0     0.857631       0.636199   -455.200012          0.0    0.064815   0.962085        1.0         1.0         1.0           1.0       787.538452
ratio_retained_earnings_equity  11102190.0     0.238191     552.477295   -1825320.75    -4.396567   -0.507246        0.0   0.391509    0.772556    1.249652      5.035714     71914.601562
dividend_yield                  11102223.0     0.086149       5.455712        -756.0          0.0         0.0        0.0        0.0     0.03662    0.543478      0.871658     14634.146484
dso_days                         8428206.0  5249.406738     2522581.25   -67996216.0          0.0    2.844156  29.871956  61.949543  126.346161  990.842468        9490.0     4244967936.0
dpo_days                        10064630.0  -324.395752    124672.6875  -351128160.0  -777.608704 -145.526733 -37.758621 -12.561313        -0.0        -0.0          -0.0        1311080.0
rr01_ntoms_yoy_abs              10011112.0   939.137329   110158.84375   -48785892.0     -12714.0     -1732.0      -73.0        0.0       268.0      3790.0       23661.0       99013568.0
rr07_rorresul_yoy_pct            9448807.0    -1.131404    1492.768433    -3911197.0   -25.833334        -4.1     -0.875  -0.145137         0.5         5.5          26.0         714887.0
ny_solid_yoy_diff                9375868.0     0.011634       1.492033  -2750.818115    -0.533025   -0.195467  -0.024067   0.001355    0.052027    0.226731        0.5289      2751.488037
ratio_cash_liquidity_yoy_pct     8575282.0   121.896095     104132.625    -1761359.0         -1.0   -0.956321  -0.383735        0.0     0.54949    6.363993     63.023239      264824576.0
ratio_cash_liquidity_yoy_abs     9274777.0     0.180016     281.710754 -370961.21875   -44.422012   -3.936294  -0.198029        0.0    0.279352    4.955592     51.865005    154306.609375
dso_days_yoy_diff                7420916.0  -333.823334    1493204.125 -2060297984.0 -2357.244873 -265.564148 -19.819376        0.0    18.75761  258.047363   3406.666992      595827392.0
inventory_days_yoy_diff          8925630.0    71.562485   78059.351562  -103192072.0  -245.567184  -23.122553        0.0        0.0         0.0   31.304049    339.806396      102938624.0
dpo_days_yoy_diff                8926087.0    78.869553  184762.296875  -351128160.0  -290.128204  -50.955662  -4.158493        0.0    7.137549   72.401588    454.508362      351128160.0
current_ratio_yoy_pct            9177100.0   270.436829  253848.515625    -1148457.0    -0.989416    -0.73915   -0.18309        0.0    0.284193    2.757202     25.877018      677502400.0
revenue_cagr_3y                  5848437.0     0.114251       0.688921     -0.990543    -0.758879   -0.479555  -0.086157   0.025211    0.150185    0.922618      2.530349        88.765907
profit_cagr_3y                   4030885.0     0.310422       1.188957     -0.996498    -0.816181   -0.626414   -0.19996   0.074174    0.446347    1.950662      4.783983       147.459473
revenue_drawdown_5y              6130236.0    -0.389484       0.369273     -1.462992         -1.0        -1.0  -0.727965  -0.261089   -0.049923         0.0           0.0              0.0
dpo_days_trend_3y                7885578.0    15.297088   69970.269574  -175564080.0  -179.288778  -31.812879  -3.216968        0.0    5.360905   45.272036    274.106028  11421279.603729
event_count_last_5y             12473668.0      0.03357       0.227027           0.0          0.0         0.0        0.0        0.0         0.0         0.0           1.0              5.0
term_spread                     10245078.0     0.210625       0.714608     -1.335367    -1.335367   -1.335367    -0.2754   0.363442    0.480642    1.534667      1.616875         1.616875
```

---

## End of Report

**Report generated**: 2025-11-14
**Dataset**: /data/processed/serrano_features.parquet
**Total records analyzed**: 12,473,668
**Features analyzed**: 40 (from FEATURES_FOR_MODEL)