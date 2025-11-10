# Engineered Feature Catalogue

This document summarizes the **40 final features** selected via comprehensive feature selection pipeline (Strategy 4: Hybrid). The pipeline combined VIF analysis, Stability Selection, Boruta Algorithm, and SHAP-based ranking to systematically reduce from 65 candidate features to an optimal set of 40.

## Final Feature Selection (2025)

**Selection Method**: Strategy 4 (Hybrid Consensus + SHAP Top-Up)
- **Core**: 32 features with consensus from ≥3 selection methods (VIF, Stability, Boruta, SHAP)
- **Top-up**: 8 additional features ranked by SHAP importance
- **Performance**: 0.6500 PR-AUC (exceeds 65-feature baseline of 0.6487)
- **Efficiency**: 38% reduction in feature count with improved performance

**Model Scope**: This credit risk model applies to **independent companies only** (filtered to `knc_kncfall==1`). Subsidiaries, parent companies, and other organizational structures are excluded because they exhibit fundamentally different risk profiles due to intragroup financing, parent support mechanisms, and consolidated financial statements.

**Removed Features** (25 features eliminated via systematic selection):
- Log nominal: `log_rr01_ntoms`, `log_br09_tillgsu`, `log_rr07_rorresul`
- Ratios: `ratio_ebitda_margin`, `ratio_nwc_sales`, `ratio_secured_debt_assets`
- Liquidity: `current_ratio` (kept `current_ratio_yoy_pct` only), `inventory_days`
- Trends: `ny_skuldgrd_yoy_diff`, `net_debt_to_ebitda_yoy_diff`
- Temporal: `equity_cagr_3y`, `equity_drawdown_5y`, `inventory_days_trend_3y`
- OCF metrics: All 5 OCF features (`ocf_proxy`, `ratio_ocf_to_debt`, etc.)
- Altman: `working_capital_to_assets`, `retained_earnings_to_assets`
- Leverage: `financial_mismatch`, `net_debt_to_ebitda`
- Macro: `gdp_growth`, `interest_avg_short`, `revenue_beta_gdp_5y`
- Base: `bslov_antanst`, `bransch_borsbransch_konv`, `ny_avktokap`

**Retained Features** (40 high-value features):
- See sections below for complete feature definitions and rationale

**Key Implementation**: Feature engineering code still computes intermediate features needed for deriving the final 40 features (e.g., `inventory_days` is computed to calculate `inventory_days_yoy_diff`), but only the 40 selected features are included in `FEATURES_FOR_MODEL`.

## Log-Transformed Nominal Features

To address skewness in absolute financial values and align with academic literature (Italian bankruptcy models commonly use log(total assets)), nominal values are log-transformed using `log1p()` (robust to zeros). Negative values return NaN (not economically meaningful for balance sheet items).

| Feature | Source Column | Status | Purpose |
| --- | --- | --- | --- |
| ~~`log_rr01_ntoms`~~ | Net revenue (kSEK) | **REMOVED** | Size proxy eliminated via feature selection. |
| ~~`log_br09_tillgsu`~~ | Total assets (kSEK) | **REMOVED** | Redundant with equity; eliminated via feature selection. |
| `log_br10_eksu` | Total equity (kSEK) | **RETAINED** | Capital base measurement; complements equity ratios. |
| `log_br07b_kabasu` | Cash and bank (kSEK) | **RETAINED** | Liquidity buffer; absolute cash matters for small firms. |
| `log_bslov_antanst` | Number of employees | **RETAINED** | Alternative size proxy; less volatile than revenue. |
| ~~`log_rr07_rorresul`~~ | Operating profit (kSEK) | **REMOVED** | Profitability scale eliminated via feature selection. |
| `log_rr15_resar` | Net profit (kSEK) | **RETAINED** | Bottom-line profitability scale (NaN for negative values). |

**Rationale**: While ratios capture relative performance, absolute levels matter for SMEs where scale effects are pronounced. Log transformation reduces skewness and provides scale-invariant features that complement ratio-based metrics. Strategy 4 retained 4 of 7 log-transformed features.

## Cost Structure & Profitability Ratios

| Feature | Definition / Formula | Status | Purpose |
| --- | --- | --- | --- |
| ~~`ratio_personnel_cost`~~ | ~~`rr04_perskos / rr01_ntoms`~~ | **REMOVED** | Redundant (multicollinearity with ny_nettomarg, r=0.92; low unique contribution after pruning). |
| `ratio_depreciation_cost` | `rr05_avskriv / rr01_ntoms` | **RETAINED** | Depreciation intensity relative to sales; proxy for capital intensity. Bridges EBITDA→EBIT. |
| ~~`ratio_other_operating_cost`~~ | ~~`rr06_rorkoov / rr01_ntoms`~~ | **REMOVED** | Lowest ablation impact (-0.000446), 3 red flags, SHAP=0.010. Captured by other profitability metrics. |
| ~~`ratio_financial_cost`~~ | ~~`rr09_finkostn / rr01_ntoms`~~ | **REMOVED** | Redundant (multicollinearity & low unique contribution after pruning). |
| ~~`ratio_ebitda_margin`~~ | ~~`(rr07_rorresul + rr05_avskriv) / rr01_ntoms`~~ | **REMOVED** | Near-perfect correlation with `ny_rormarg` (r=0.998). |
| ~~`ratio_ebit_interest_cov`~~ | ~~`rr07_rorresul / (rr09_finkostn - rr09d_jfrstfin)`~~ | **REMOVED** | Low SHAP (0.021), signal captured by `ny_rs` (interest coverage already in Nyckeltal). |
| `ratio_cash_interest_cov` | `br07b_kabasu / (rr09_finkostn - rr09d_jfrstfin)` | **RETAINED** | Cash-on-hand relative to annual financial costs. |
| ~~`ratio_dividend_payout`~~ | ~~`rr00_utdbel / rr15_resar`~~ | **REMOVED** | Unstable denominator (profit can be ≤ 0); replaced with `dividend_yield`. |
| `dividend_yield` | `rr00_utdbel / br10_eksu` | **RETAINED** | Stable alternative to dividend payout using equity as denominator. |
| ~~`ratio_group_support`~~ | ~~`(br10f_kncbdrel + br10g_agtskel) / rr01_ntoms`~~ | **REMOVED** | Only relevant for subsidiaries/group companies. Model applies to independent companies (knc_kncfall==1) only. |
| ~~`ratio_intragroup_financing_share`~~ | ~~`(rr08a_rteinknc + rr09a_rtekoknc) / (rr08_finintk + rr09_finkostn)`~~ | **REMOVED** | Only relevant for subsidiaries/group companies. Model applies to independent companies (knc_kncfall==1) only. |

## Liquidity & Working Capital Efficiency

| Feature | Definition / Formula | Status | Purpose |
| --- | --- | --- | --- |
| `ratio_cash_liquidity` | `(br07b_kabasu + br07a_kplacsu) / br13_ksksu` | **RETAINED** | Quick ratio (cash & near cash vs. current liabilities). |
| `dso_days` | `(br06g_kfordsu / rr01_ntoms) * 365` | **RETAINED** | Days sales outstanding; completes working capital trinity (DSO + inventory + DPO) per academic literature. |
| ~~`inventory_days`~~ | ~~`(br06c_lagersu / rr06a_prodkos) * 365`~~ | **REMOVED** | Low importance; information captured by derivatives (inventory_days_yoy_diff). Note: Still computed for YoY derivative. |
| `dpo_days` | `(br13a_ksklev / rr06a_prodkos) * 365` | **RETAINED** | Days payables outstanding; supplier payment terms. |
| ~~`current_ratio`~~ | ~~`br08_omstgsu / br13_ksksu`~~ | **REMOVED** | Eliminated via feature selection; YoY variant retained. Note: Still computed temporarily for YoY derivative. |
| ~~`cash_conversion_cycle`~~ | ~~`dso_days + inventory_days - dpo_days`~~ | **REMOVED** | High correlation with `dso_days` (r=0.971). |
| ~~`ratio_nwc_sales`~~ | ~~`(br06_lagerkford + br07_kplackaba - br13_ksksu) / rr01_ntoms`~~ | **REMOVED** | Net working capital relative to sales eliminated via feature selection. |

## Capital Structure Detail

| Feature | Definition / Formula | Status | Purpose |
| --- | --- | --- | --- |
| `ratio_short_term_debt_share` | `br13_ksksu / (br13_ksksu + br15_lsksu)` | **RETAINED** | Share of debt maturing within 12 months. |
| ~~`ratio_secured_debt_assets`~~ | ~~`(br14_kskkrin + br16_lskkrin) / br09_tillgsu`~~ | **REMOVED** | Secured debt relative to total assets eliminated via feature selection. |
| `ratio_retained_earnings_equity` | `br10e_balres / br10_eksu` | **RETAINED** | Retained earnings composition of equity. |
| ~~`equity_to_sales`~~ | ~~`br10_eksu / rr01_ntoms`~~ | **REMOVED** | Redundant (multicollinearity & low unique contribution after pruning). |
| ~~`equity_to_profit`~~ | ~~`br10_eksu / rr15_resar`~~ | **REMOVED** | Redundant with ROE (`ny_avkegkap`); unstable denominator (profit can be ≤ 0). |
| ~~`assets_to_profit`~~ | ~~`br09_tillgsu / rr15_resar`~~ | **REMOVED** | Redundant with ROA (`ny_avktokap`); unstable denominator (profit can be ≤ 0). |

## Raw Financial Statement Values: Log-Transformed Approach

**Previous Approach** (Now Revised): Raw nominal values were excluded, assuming size was sufficiently captured by employee count and equity.

**Current Approach**: Based on model analysis showing that absolute levels DO matter (e.g., `br07b_kabasu` ranked #5, `br10_eksu` ranked #21), and aligning with academic literature (Altman Z-Score, Italian models using log(assets)), **all nominal financial statement values are now included as log-transformed features**.

**Rationale for log transformation**:
1. **Reduces skewness**: Financial data is heavily right-skewed; log transformation normalizes distributions
2. **Scale invariance**: log(1M kr) - log(100k kr) ≈ log(10M kr) - log(1M kr) → equal percentage changes
3. **Literature alignment**: Italian bankruptcy models and Altman Z-Score use log(total assets) as standard practice
4. **SME relevance**: Absolute scale matters for small firms where fixed costs and indivisibilities create threshold effects

All log transformations use `log1p()` (robust to zeros). Negative values return NaN (not economically meaningful for balance sheet items; acceptable for profit measures where losses indicate distress).

## YoY Change & Trend Features

| Feature | Definition | Status | Purpose |
| --- | --- | --- | --- |
| `rr01_ntoms_yoy_abs` | YoY absolute change in revenue. | **RETAINED** | Revenue momentum (absolute change preferred over pct). |
| ~~`rr01_ntoms_yoy_pct`~~ | ~~YoY % change in revenue.~~ | **REMOVED** | Perfect correlation with `ny_omsf` (r=1.0). |
| `rr07_rorresul_yoy_pct` | YoY % change in operating profit. | **RETAINED** | Profit momentum. |
| ~~`br09_tillgsu_yoy_pct`~~, ~~`br09_tillgsu_yoy_abs`~~ | ~~YoY asset changes.~~ | **REMOVED** | br09_tillgsu (total assets) redundant with br10_eksu (equity), r=0.904. |
| `ny_solid_yoy_diff` | YoY difference in equity ratio. | **RETAINED** | Capital structure drift. |
| ~~`ny_skuldgrd_yoy_diff`~~ | ~~YoY difference in leverage.~~ | **REMOVED** | Eliminated via feature selection. |
| `ratio_cash_liquidity_yoy_pct` | YoY % change in quick ratio. | **RETAINED** | Liquidity trend. |
| `ratio_cash_liquidity_yoy_abs` | YoY absolute change in quick ratio. | **RETAINED** | Liquidity trend magnitude. |
| ~~`ratio_ebit_interest_cov_yoy_pct`~~ | ~~YoY change in EBIT coverage.~~ | **REMOVED** | `ratio_ebit_interest_cov` removed (low SHAP). |
| `dso_days_yoy_diff` | YoY difference in days sales outstanding. | **RETAINED** | Complements DSO level; working capital trinity. |
| `inventory_days_yoy_diff` | YoY difference in inventory days. | **RETAINED** | Working capital efficiency trend. |
| `dpo_days_yoy_diff` | YoY difference in days payables outstanding. | **RETAINED** | Supplier payment trend. |
| `current_ratio_yoy_pct` | YoY % change in current ratio. | **RETAINED** | Liquidity trend for current ratio. |
| ~~`net_debt_to_ebitda_yoy_diff`~~ | ~~YoY change in net debt to EBITDA.~~ | **REMOVED** | Leverage trajectory eliminated via feature selection. |

## Temporal Features (Multi-Year Lookback)

**Selection methodology**: The following 4 temporal features were selected via Strategy 4 from an initial set of 34 candidates using rigorous nested cross-validation, with additional pruning based on multicollinearity analysis.

### Growth Metrics (CAGR)

Capture fundamental business momentum and growth trajectory.

| Feature | Window | Formula | Status | Selection Rationale |
| --- | --- | --- | --- | --- |
| `revenue_cagr_3y` | 3 years | CAGR of `rr01_ntoms` | **RETAINED** | Revenue growth is a primary credit risk indicator. 3y window balances recent trends with stability. |
| ~~`assets_cagr_3y`~~ | ~~3 years~~ | ~~CAGR of `br09_tillgsu`~~ | **REMOVED** | br09_tillgsu (total assets) redundant with br10_eksu (equity), r=0.904. |
| ~~`equity_cagr_3y`~~ | ~~3 years~~ | ~~CAGR of `br10_eksu`~~ | **REMOVED** | Eliminated via feature selection (Strategy 4). |
| `profit_cagr_3y` | 3 years | CAGR of `rr15_resar` | **RETAINED** | Profit growth trajectory is a direct indicator of business health. |

**Note**: 5y CAGR variants were tested but 3y windows provided optimal signal without overfitting. Growth metrics (CAGR) were found to be more informative than volatility or average-based features for these metrics.

### Risk Metrics (Drawdown)

Capture downside risk exposure and vulnerability to adverse conditions.

| Feature | Window | Definition | Status | Selection Rationale |
| --- | --- | --- | --- | --- |
| `revenue_drawdown_5y` | 5 years | Maximum drawdown of revenue (min value / peak - 1) | **RETAINED** | High SHAP importance (rank 7). Captures revenue resilience and exposure to demand shocks. 5y window needed to capture full business cycles. |
| ~~`equity_drawdown_5y`~~ | ~~5 years~~ | ~~Maximum drawdown of equity~~ | **REMOVED** | Eliminated via feature selection (Strategy 4). |

**Note**: Drawdown features were only kept for revenue, where it showed statistically significant incremental value beyond CAGR alone.

### Working Capital Trends

Early warning signals for operational deterioration and cash flow stress.

| Feature | Window | Definition | Status | Selection Rationale |
| --- | --- | --- | --- | --- |
| ~~`dso_days_trend_3y`~~ | ~~3 years~~ | ~~Linear slope of days sales outstanding~~ | **REMOVED** | dso_days redundant with ratio_nwc_sales, r=-0.944. |
| ~~`inventory_days_trend_3y`~~ | ~~3 years~~ | ~~Linear slope of inventory days~~ | **REMOVED** | Eliminated via feature selection (Strategy 4). |
| `dpo_days_trend_3y` | 3 years | Linear slope of days payables outstanding | **RETAINED** | Lengthening payment cycles can signal liquidity pressure. |

**Note**: Trend features were more informative than volatility or average-based features for working capital metrics.

### Excluded Temporal Feature Categories

The following temporal feature types were systematically excluded after testing:

| Category | Reason for Exclusion | Test Result |
| --- | --- | --- |
| Operating margin temporal features | Static `ny_rormarg` captures most signal; temporal derivatives add minimal value | AUC drop: +0.000062 (statistically zero) |
| Net margin temporal features | Static `ny_nettomarg` and YoY changes sufficient; trends/volatility/averages redundant | AUC drop: -0.000223 (negative = overfitting) |
| Leverage temporal features | Static `ny_skuldgrd` and YoY changes sufficient | AUC drop: -0.000215 (negative = overfitting) |
| Cash liquidity temporal features | Static ratio and YoY changes capture signal; trends/volatility/averages redundant | AUC drop: -0.000277 (negative = overfitting) |

**Key insight**: For margin and liquidity metrics, **static values + YoY changes** are more robust than multi-year trends/volatility/averages, which tend to overfit to validation-specific patterns.

## Operating Cash Flow (OCF) Features

**Status**: All OCF features were eliminated via Strategy 4 feature selection.

| Feature | Definition | Status |
| --- | --- | --- |
| ~~`ocf_proxy`~~ | ~~`(rr07_rorresul + rr05_avskriv) - ΔWorking Capital`~~ | **REMOVED** |
| ~~`ratio_ocf_to_debt`~~ | ~~`ocf_proxy / (br13_ksksu + br15_lsksu)`~~ | **REMOVED** |
| ~~`ocf_proxy_yoy_pct`~~ | ~~YoY % change in `ocf_proxy`~~ | **REMOVED** |
| ~~`ratio_ocf_to_debt_yoy_diff`~~ | ~~YoY change in `ratio_ocf_to_debt`~~ | **REMOVED** |
| ~~`ocf_proxy_trend_3y`~~ | ~~3-year linear slope of `ocf_proxy`~~ | **REMOVED** |

**Rationale**: While OCF features were initially added based on academic literature emphasizing cash flow, the comprehensive feature selection pipeline (Strategy 4) determined that these features did not provide sufficient incremental predictive value relative to their complexity.

## Altman Z-Score Components

**Status**: Altman-specific components were eliminated via Strategy 4 feature selection.

| Feature | Altman Component | Definition | Status |
| --- | --- | --- | --- |
| ~~`working_capital_to_assets`~~ | ~~X₁~~ | ~~`(br08_omstgsu - br13_ksksu) / br09_tillgsu`~~ | **REMOVED** |
| ~~`retained_earnings_to_assets`~~ | ~~X₂~~ | ~~`br10e_balres / br09_tillgsu`~~ | **REMOVED** |

**Note**: While Altman-specific ratios were removed, similar information is captured by retained features:
- X₃ (EBIT/Assets) via profitability metrics
- X₄ (Equity/Liabilities) via `ny_skuldgrd` and `ny_solid`
- X₅ (Sales/Assets) via `ny_kapomsh` (asset turnover)

## Leverage & Financial Mismatch Features

**Status**: Advanced leverage features were eliminated via Strategy 4 feature selection.

| Feature | Definition | Status |
| --- | --- | --- |
| ~~`financial_mismatch`~~ | ~~`(br13_ksksu - br08_omstgsu) / br09_tillgsu`~~ | **REMOVED** |
| ~~`net_debt_to_ebitda`~~ | ~~`(br13_ksksu + br15_lsksu - br07_kplackaba) / (rr07_rorresul + rr05_avskriv)`~~ | **REMOVED** |
| ~~`net_debt_to_ebitda_yoy_diff`~~ | ~~YoY change in `net_debt_to_ebitda`~~ | **REMOVED** |

**Rationale**: While these advanced leverage metrics are standard in credit rating agencies, the feature selection process determined that simpler leverage metrics (like `ny_skuldgrd` and `ratio_short_term_debt_share`) captured the essential information more efficiently.

## Credit Event History

| Feature | Definition / Purpose | Status |
| --- | --- | --- |
| ~~`years_since_last_credit_event`~~ | Years since last credit event. | **REMOVED** - Potential data leakage - backward-looking feature that may reflect information not available at prediction time for independent companies. |
| ~~`event_count_total`~~ | Total credit events in history. | **REMOVED** - Replaced with `event_count_last_5y` to prevent overfitting to rare historical events (only 0.16% of companies have events older than 5 years). |
| `event_count_last_5y` | Credit events within the past 5 years. | **RETAINED** - Preferred over total count to avoid data leakage from sparse historical events. |

## Macro Features & Firm-Macro Comparisons

| Feature | Definition | Status |
| --- | --- | --- |
| ~~`gdp_growth`~~ | ~~Annual GDP growth (market prices).~~ | **REMOVED** - Eliminated via feature selection (Strategy 4) despite theoretical relevance. |
| ~~`interest_avg_short`~~ | ~~Annual average of short-term corporate borrowing rates (≤3m).~~ | **REMOVED** - Eliminated via feature selection (Strategy 4) despite theoretical relevance. |
| `term_spread` | Long – short rate spread. | **RETAINED** - Captures yield curve information relevant to credit conditions. |
| ~~`inflation_yoy`~~ | ~~YoY CPI change (based on annual average KPIF).~~ | **REMOVED** - Near-zero variance (0.0002) and low predictive value after pruning. |
| ~~`unemp_rate`~~ | ~~National unemployment level.~~ | **REMOVED** - Low importance; macroeconomic conditions sufficiently captured by term_spread. |
| ~~`revenue_beta_gdp_5y`~~ | ~~Rolling 5-year beta (cyclicality) of revenue growth vs. GDP growth.~~ | **REMOVED** - Eliminated via feature selection (Strategy 4). |

---

This catalogue is kept in sync with the project's feature engineering pipeline. Update this file whenever you modify the feature set.
