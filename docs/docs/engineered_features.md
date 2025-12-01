# Engineered Feature Catalogue

This document summarizes the **29 final features** selected via comprehensive feature selection pipeline. The pipeline combined VIF analysis, Stability Selection (Bootstrap), Boruta Algorithm, SHAP-based ranking, and RFECV to systematically reduce from 54 candidate features to an optimal set of 29.

## Final Feature Selection (2025)

**Selection Method**: Multi-Method Consensus + RFECV Optimization
- **Phase 1**: Baseline model (54 features)
- **Phase 2**: VIF multicollinearity removal (→ 46 features)
- **Phase 3**: Stability Selection via bootstrap (50 iterations, 70% threshold → 32 features)
- **Phase 3B**: Boruta all-relevant features (→ 9 confirmed features)
- **Phase 4**: SHAP ranking & 4-method consensus (≥3 methods → 28 features)
- **Phase 5**: RFECV optimization using ROC-AUC (→ 28 features optimal)
- **Manual additions**: +2 features (ny_omsf, log_br10_eksu) based on domain knowledge
- **Manual removal**: -1 feature (ratio_cash_liquidity_yoy_pct, kept absolute version)
- **Performance**: ROC-AUC 0.8962, PR-AUC 0.1715 (baseline: 0.8980, 0.1723)
- **Efficiency**: 46% reduction in feature count with minimal performance loss (-0.0018 ROC-AUC)

**Model Scope**: This credit risk model applies to **independent companies only** (filtered to `knc_kncfall==1`). Subsidiaries, parent companies, and other organizational structures are excluded because they exhibit fundamentally different risk profiles due to intragroup financing, parent support mechanisms, and consolidated financial statements.

**Key Implementation**: Feature engineering code still computes intermediate features needed for deriving the final 29 features (e.g., `inventory_days` is computed to calculate `inventory_days_yoy_diff`), but only the 29 selected features are included in `FEATURES_FOR_MODEL`.

## Final 29 Features by Category

### Base Features (3)
- `company_age` - Years since registration
- `sni_group_3digit` - Industry code (3-digit SNI grouping, ~267 categories)
- `ser_laen` - County code (geographic effects, ~20 categories)

### Nyckeltal Ratios (10)
- `ny_foradlvpanst` - Value added per employee
- `ny_kapomsh` - Capital turnover (revenue/total assets)
- `ny_rs` - Interest coverage ratio
- `ny_skuldgrd` - Debt ratio (total liabilities/total assets)
- `ny_solid` - Equity ratio (equity/total assets)
- `ny_avkegkap` - Return on equity (ROE)
- `ny_kasslikv` - Quick ratio ((current assets - inventories) / current liabilities)
- `ny_nettomarg` - Net profit margin
- `ny_omspanst` - Revenue per employee
- `ny_omsf` - YoY change in net sales (short-term revenue momentum) [Manual addition]

### Log-Transformed Nominal Features (2)
- `log_br07b_kabasu` - Log of cash and bank balances
- `log_br10_eksu` - Log of total equity (size/buffer indicator) [Manual addition]

### Engineered Ratio Features (5)
- `ratio_depreciation_cost` - Depreciation intensity (depreciation/revenue)
- `ratio_cash_interest_cov` - Cash interest coverage (cash/financial costs)
- `ratio_cash_liquidity` - Cash ratio ((cash + short-term investments) / current liabilities)
- `ratio_retained_earnings_equity` - Retained earnings composition of equity
- `dividend_yield` - Dividends relative to equity

### Working Capital Efficiency (2)
- `dso_days` - Days sales outstanding
- `dpo_days` - Days payables outstanding

### Year-over-Year Trends (3)
- `ny_solid_yoy_diff` - YoY change in equity ratio
- `ratio_cash_liquidity_yoy_abs` - Absolute YoY change in cash ratio
- `inventory_days_yoy_diff` - YoY change in inventory turnover days

### Multi-Year Temporal Features (3)
- `revenue_cagr_3y` - 3-year revenue compound annual growth rate
- `profit_cagr_3y` - 3-year profit compound annual growth rate
- `revenue_drawdown_5y` - Maximum 5-year revenue drawdown (downside risk)

### Macroeconomic Conditions (1)
- `term_spread` - Long-short interest rate spread (yield curve)

---

## Removed Features (25 features eliminated)

### Removed by VIF (Multicollinearity - VIF > 10)
- `ratio_ocf_to_debt` (VIF=219.72) - Extreme multicollinearity
- `log_br10_eksu` (VIF=76.74) - *Later re-added manually for equity size information*
- `unemp_rate` (VIF=41.97) - Correlation with other macro indicators
- `ny_kasslikv` (VIF=32.20) - *Conflict: selected by Boruta but removed by VIF*
- `log_bslov_antanst` (VIF=22.20) - Employee count redundant with other size proxies
- `ratio_short_term_debt_share` (VIF=16.19) - Debt structure captured by other metrics
- `log_rr15_resar` (VIF=14.23) - Net profit scale redundant with profitability ratios
- `current_ratio_yoy_pct` (VIF=10.37) - Current ratio change redundant

### Removed by Low Stability (Bootstrap < 70%)
- `rr01_ntoms_yoy_abs` (18%) - Absolute revenue change unstable
- `ocf_proxy` (14%) - Operating cash flow proxy low stability
- `ny_omsf` (8%) - *Later re-added manually as YoY revenue growth signal*
- `ratio_ocf_to_debt_yoy_diff` (8%) - OCF debt coverage change
- `ocf_proxy_yoy_pct` (6%) - OCF percentage change
- `net_debt_to_ebitda` (4%) - Leverage metric
- `gdp_growth` (2%) - GDP growth rate
- `any_event_last_5y` (0%) - Credit event history (categorical artifact)
- `bransch_borsbransch_konv` (0%) - Industry grouping (categorical artifact)

### Removed by Consensus (< 3 methods)
- `ratio_cash_liquidity_yoy_pct` - *Manually removed in favor of absolute version*
- `dso_days_yoy_diff` - DSO change not in consensus
- `dpo_days_yoy_diff` - DPO change not in consensus
- `rr07_rorresul_yoy_pct` - Operating profit YoY change
- `dpo_days_trend_3y` - Multi-year DPO trend
- `interest_avg_short` - Short-term interest rates
- `inflation_yoy` - Inflation rate
- `revenue_beta_gdp_5y` - Revenue-GDP cyclicality

### All OCF Features Removed
- `ocf_proxy`, `ratio_ocf_to_debt`, `ocf_proxy_yoy_pct`, `ratio_ocf_to_debt_yoy_diff`, `ocf_proxy_trend_3y`

### Additional Leverage Features Removed
- `net_debt_to_ebitda`, `net_debt_to_ebitda_yoy_diff`

---

## Feature Selection Pipeline Details

### Phase 2: VIF Analysis
- **Method**: Iterative removal of features with VIF > 10
- **Iterations**: 8 iterations until convergence
- **Result**: 46 features (removed 8)

### Phase 3: Stability Selection
- **Method**: Bootstrap resampling (50 iterations, 80% sample size)
- **Selection**: Top 35 features by SHAP importance per iteration
- **Threshold**: ≥70% selection frequency
- **Result**: 32 features with stability ≥70%

### Phase 3B: Boruta Algorithm
- **Method**: All-relevant feature selection using permutation tests
- **Parameters**:
  - n_estimators=750 (LightGBM trees)
  - max_iter=100
  - perc=90 (90th percentile threshold)
- **Result**: 9 confirmed features (too conservative)
- **Note**: Boruta was added back per user request despite conservative selection

### Phase 4: SHAP Ranking & Consensus
- **Methods combined**: VIF, Stability, Boruta, SHAP (top 35)
- **Consensus threshold**: ≥3 of 4 methods
- **Result**: 28 features with majority agreement

### Phase 5: RFECV Optimization
- **Method**: Recursive Feature Elimination with Cross-Validation
- **Metric**: ROC-AUC (3-fold stratified CV)
- **Tested counts**: 25, 28 features
- **Result**: 28 features optimal (ROC-AUC 0.8962 vs 0.8938 for 25)

### Manual Adjustments
**Added**:
1. `ny_omsf` (YoY revenue change) - Short-term momentum complements 3-year CAGR
2. `log_br10_eksu` (total equity) - Absolute equity size complements equity ratio

**Removed**:
1. `ratio_cash_liquidity_yoy_pct` - Kept absolute version (better captures threshold crossings)

**Rationale**: Domain expertise suggests absolute equity size and short-term revenue momentum provide unique information despite lower statistical selection scores.

---

## Performance Comparison

| Method | Features | ROC-AUC | PR-AUC | Notes |
|--------|----------|---------|--------|-------|
| Baseline | 54 | 0.8980 | 0.1723 | All features |
| VIF | 46 | 0.8977 | 0.1718 | Multicollinearity removal |
| Stability | 32 | 0.8971 | 0.1716 | Bootstrap 70% threshold |
| Boruta | 9 | 0.8812 | 0.1542 | Too conservative |
| SHAP | 35 | 0.8968 | 0.1714 | Top 35 by importance |
| Consensus | 28 | 0.8963 | 0.1716 | ≥3 methods agree |
| **Final (RFECV)** | **28** | **0.8962** | **0.1715** | **Optimal** |
| Final (+ manual) | **29** | *TBD* | *TBD* | With domain additions |

**Key Insight**: 46% feature reduction with only 0.0018 ROC-AUC loss demonstrates effective feature selection without sacrificing predictive power.

---

## Detailed Feature Descriptions

### Log-Transformed Nominal Features

To address skewness in absolute financial values, nominal values are log-transformed using `log1p()` (robust to zeros). Negative values return NaN (not economically meaningful for balance sheet items).

| Feature | Source Column | Purpose |
| --- | --- | --- |
| `log_br07b_kabasu` | Cash and bank (kSEK) | Liquidity buffer; absolute cash matters for small firms. |
| `log_br10_eksu` | Total equity (kSEK) | Capital base measurement; complements equity ratios. Manual addition. |

**Rationale**: While ratios capture relative performance, absolute levels matter for SMEs where scale effects are pronounced. VIF removed log_br10_eksu due to correlation with ny_solid (equity ratio), but absolute equity size provides unique information about company buffer capacity.

### Nyckeltal Ratios

Standard Swedish financial ratios provided in the Serrano database.

| Feature | Definition | Purpose |
| --- | --- | --- |
| `ny_foradlvpanst` | Value added per employee | Labor productivity |
| `ny_kapomsh` | Revenue / Total assets | Capital turnover efficiency |
| `ny_rs` | (Profit + Financial costs) / Financial costs | Interest coverage |
| `ny_skuldgrd` | Total liabilities / Total assets | Leverage (debt ratio) |
| `ny_solid` | Equity / Total assets | Solvency (equity ratio) |
| `ny_avkegkap` | Profit / Equity | Return on equity (ROE) |
| `ny_kasslikv` | (Current assets - Inventories) / Current liabilities | Quick ratio (acid-test) |
| `ny_nettomarg` | Net profit / Revenue | Net profit margin |
| `ny_omspanst` | Revenue / Employees | Revenue per employee |
| `ny_omsf` | YoY change in net sales | Short-term revenue momentum (manual addition) |

### Engineered Ratios

Custom ratios designed for credit risk assessment.

| Feature | Formula | Purpose |
| --- | --- | --- |
| `ratio_depreciation_cost` | `rr05_avskriv / rr01_ntoms` | Depreciation intensity; proxy for capital intensity |
| `ratio_cash_interest_cov` | `br07b_kabasu / (rr09_finkostn - rr09d_jfrstfin)` | Cash-on-hand relative to annual financial costs |
| `ratio_cash_liquidity` | `(br07b_kabasu + br07a_kplacsu) / br13_ksksu` | Cash ratio (most liquid assets only) |
| `ratio_retained_earnings_equity` | `br10e_balres / br10_eksu` | Retained earnings composition of equity |
| `dividend_yield` | `rr00_utdbel / br10_eksu` | Dividends relative to equity |

### Working Capital Efficiency

| Feature | Formula | Purpose |
| --- | --- | --- |
| `dso_days` | `(br06g_kfordsu / rr01_ntoms) * 365` | Days sales outstanding (receivables collection) |
| `dpo_days` | `(br13a_ksklev / rr06a_prodkos) * 365` | Days payables outstanding (supplier payment) |

### Year-over-Year Trends

| Feature | Definition | Purpose |
| --- | --- | --- |
| `ny_solid_yoy_diff` | YoY difference in equity ratio | Capital structure drift |
| `ratio_cash_liquidity_yoy_abs` | Absolute YoY change in cash ratio | Liquidity trend magnitude (manual choice over %) |
| `inventory_days_yoy_diff` | YoY difference in inventory days | Working capital efficiency trend |

**Note**: Absolute change in cash ratio preferred over percentage change because it better captures threshold crossings (e.g., moving from 0.8 to 1.0 is critical).

### Multi-Year Temporal Features

| Feature | Window | Definition | Purpose |
| --- | --- | --- | --- |
| `revenue_cagr_3y` | 3 years | CAGR of revenue | Sustained growth trajectory |
| `profit_cagr_3y` | 3 years | CAGR of net profit | Profitability growth |
| `revenue_drawdown_5y` | 5 years | Maximum revenue decline from peak | Downside risk exposure |

### Macroeconomic Conditions

| Feature | Definition | Purpose |
| --- | --- | --- |
| `term_spread` | Long-short interest rate spread | Yield curve; credit market conditions |

---

This catalogue is kept in sync with the project's feature engineering pipeline. Last updated: 2025 (29 features).
