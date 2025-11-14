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
