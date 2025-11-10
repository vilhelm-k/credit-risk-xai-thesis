# Engineered Feature Catalogue

This document summarises every engineered feature produced by `credit_risk_xai.features.engineer`. The tables are grouped by theme and note the intent and computation used in the pipeline. Unless otherwise stated, ratios use raw financial statement values in kSEK and are computed within each company-year panel.

**Model Scope**: This credit risk model applies to **independent companies only** (filtered to `knc_kncfall==1`). Subsidiaries, parent companies, and other organizational structures are excluded because they exhibit fundamentally different risk profiles due to intragroup financing, parent support mechanisms, and consolidated financial statements. This filtering ensures that financial ratios have consistent interpretation across all companies in the dataset.

**Base Features**: The model includes firm characteristics (`bslov_antanst`, `company_age`), industry controls (`bransch_sni071_konv`, `bransch_borsbransch_konv`), and macroeconomic conditions (`gdp_growth`, `interest_avg_short`, `term_spread`). Geographic controls (`ser_laen` - county code) and organizational structure (`knc_kncfall`) were excluded: the former due to low predictive value, the latter because it's used as a data filter rather than a feature.

**Recent Update (Refactoring 2025)**: Based on comprehensive model analysis and academic literature review (Altman Z-Score, Italian bankruptcy prediction studies), the feature set has been substantially revised:

**Features Removed** (stability & redundancy):
- `ratio_dividend_payout` → Replaced with `dividend_yield` (unstable denominator: profit can be ≤ 0)
- `equity_to_profit`, `assets_to_profit` → Redundant with ROE/ROA; unstable denominators
- `ratio_ebit_interest_cov` → Low SHAP (0.021), captured by `ny_rs`

**Features Added** (literature-based enhancements):
- **Log-transformed nominal values** (7 features): Size proxies using log scale to reduce skewness
- **DSO restored** (2 features): `dso_days` + `dso_days_yoy_diff` - completes working capital trinity per literature
- **OCF metrics** (2 features): Operating cash flow proxy and OCF-to-debt ratio
- **Altman Z-Score components** (2 features): Working capital/assets, retained earnings/assets
- **Leverage metrics** (3 features): Financial mismatch, net debt to EBITDA + YoY trend
- **Current ratio** (2 features): Standard liquidity metric + YoY trend

**Net Change**: +11 features (removed 4, added 15)

**Previous Pruning Context**: Following earlier iterative pruning, 50+ features were removed based on correlations, near-zero variance, and low ablation impact (see git history for details)

## Log-Transformed Nominal Features

To address skewness in absolute financial values and align with academic literature (Italian bankruptcy models commonly use log(total assets)), all nominal values are log-transformed using `log1p()` (robust to zeros). Negative values return NaN (not economically meaningful for balance sheet items).

| Feature | Source Column | Purpose |
| --- | --- | --- |
| `log_rr01_ntoms` | Net revenue (kSEK) | **ADDED**: Size proxy; reduces revenue skewness. Matches Italian paper approach. |
| `log_br09_tillgsu` | Total assets (kSEK) | **ADDED**: Standard in academic literature (Altman, Italian models); scale-invariant size proxy. |
| `log_br10_eksu` | Total equity (kSEK) | **ADDED**: Capital base measurement; complements equity ratios. |
| `log_br07b_kabasu` | Cash and bank (kSEK) | **ADDED**: Liquidity buffer; absolute cash matters for small firms. |
| `log_bslov_antanst` | Number of employees | **ADDED**: Alternative size proxy; less volatile than revenue. |
| `log_rr07_rorresul` | Operating profit (kSEK) | **ADDED**: Profitability scale (NaN for negative values). |
| `log_rr15_resar` | Net profit (kSEK) | **ADDED**: Bottom-line profitability scale (NaN for negative values). |

**Rationale**: While ratios capture relative performance, absolute levels matter for SMEs where scale effects are pronounced. Log transformation reduces skewness and provides scale-invariant features that complement ratio-based metrics.

## Cost Structure & Profitability Ratios

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| ~~`ratio_personnel_cost`~~ | ~~`rr04_perskos / rr01_ntoms`~~ | **REMOVED**: Redundant (multicollinearity with ny_nettomarg, r=0.92; low unique contribution after pruning). |
| `ratio_depreciation_cost` | `rr05_avskriv / rr01_ntoms` | Depreciation intensity relative to sales; proxy for capital intensity. Bridges EBITDA→EBIT. |
| ~~`ratio_other_operating_cost`~~ | ~~`rr06_rorkoov / rr01_ntoms`~~ | **REMOVED**: Lowest ablation impact (-0.000446), 3 red flags, SHAP=0.010. Captured by other profitability metrics. |
| ~~`ratio_financial_cost`~~ | ~~`rr09_finkostn / rr01_ntoms`~~ | **REMOVED**: Redundant (multicollinearity & low unique contribution after pruning). |
| ~~`ratio_ebitda_margin`~~ | ~~`(rr07_rorresul + rr05_avskriv) / rr01_ntoms`~~ | **REMOVED**: Near-perfect correlation with `ny_rormarg` (r=0.998). |
| ~~`ratio_ebit_interest_cov`~~ | ~~`rr07_rorresul / (rr09_finkostn - rr09d_jfrstfin)`~~ | **REMOVED**: Low SHAP (0.021), signal captured by `ny_rs` (interest coverage already in Nyckeltal). |
| `ratio_cash_interest_cov` | `br07b_kabasu / (rr09_finkostn - rr09d_jfrstfin)` | Cash-on-hand relative to annual financial costs. |
| ~~`ratio_dividend_payout`~~ | ~~`rr00_utdbel / rr15_resar`~~ | **REMOVED**: Unstable denominator (profit can be ≤ 0); replaced with `dividend_yield`. |
| `dividend_yield` | `rr00_utdbel / br10_eksu` | **ADDED**: Stable alternative to dividend payout using equity as denominator. |
| ~~`ratio_group_support`~~ | ~~`(br10f_kncbdrel + br10g_agtskel) / rr01_ntoms`~~ | **REMOVED**: Only relevant for subsidiaries/group companies. Model applies to independent companies (knc_kncfall==1) only. |
| ~~`ratio_intragroup_financing_share`~~ | ~~`(rr08a_rteinknc + rr09a_rtekoknc) / (rr08_finintk + rr09_finkostn)`~~ | **REMOVED**: Only relevant for subsidiaries/group companies. Model applies to independent companies (knc_kncfall==1) only. |

## Liquidity & Working Capital Efficiency

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| `ratio_cash_liquidity` | `(br07b_kabasu + br07a_kplacsu) / br13_ksksu` | Quick ratio (cash & near cash vs. current liabilities). |
| `dso_days` | `(br06g_kfordsu / rr01_ntoms) * 365` | **ADDED BACK**: Days sales outstanding; completes working capital trinity (DSO + inventory + DPO) per academic literature. |
| ~~`inventory_days`~~ | ~~`(br06c_lagersu / rr06a_prodkos) * 365`~~ | **REMOVED**: Low importance; information captured by derivatives (inventory_days_yoy_diff, inventory_days_trend_3y). |
| `dpo_days` | `(br13a_ksklev / rr06a_prodkos) * 365` | Days payables outstanding; supplier payment terms. |
| `current_ratio` | `br08_omstgsu / br13_ksksu` | **ADDED**: Standard liquidity metric (current assets / current liabilities). Includes inventory, complements quick ratio. |
| ~~`cash_conversion_cycle`~~ | ~~`dso_days + inventory_days - dpo_days`~~ | **REMOVED**: High correlation with `dso_days` (r=0.971). |
| `ratio_nwc_sales` | `(br06_lagerkford + br07_kplackaba - br13_ksksu) / rr01_ntoms` | Net working capital relative to sales. |

## Capital Structure Detail

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| `ratio_short_term_debt_share` | `br13_ksksu / (br13_ksksu + br15_lsksu)` | Share of debt maturing within 12 months. |
| `ratio_secured_debt_assets` | `(br14_kskkrin + br16_lskkrin) / br09_tillgsu` | Secured debt relative to total assets. |
| `ratio_retained_earnings_equity` | `br10e_balres / br10_eksu` | Retained earnings composition of equity. |
| ~~`equity_to_sales`~~ | ~~`br10_eksu / rr01_ntoms`~~ | **REMOVED**: Redundant (multicollinearity & low unique contribution after pruning). |
| ~~`equity_to_profit`~~ | ~~`br10_eksu / rr15_resar`~~ | **REMOVED**: Redundant with ROE (`ny_avkegkap`); unstable denominator (profit can be ≤ 0). |
| ~~`assets_to_profit`~~ | ~~`br09_tillgsu / rr15_resar`~~ | **REMOVED**: Redundant with ROA (`ny_avktokap`); unstable denominator (profit can be ≤ 0). |

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

| Feature | Definition | Purpose |
| --- | --- | --- |
| ~~`rr01_ntoms_yoy_pct`~~, `rr01_ntoms_yoy_abs` | YoY change in revenue. | **REMOVED pct variant**: Perfect correlation with `ny_omsf` (r=1.0). Absolute change retained. |
| `rr07_rorresul_yoy_pct` | YoY changes in operating profit. | Profit momentum. |
| ~~`br09_tillgsu_yoy_pct`~~, ~~`br09_tillgsu_yoy_abs`~~ | ~~YoY asset changes.~~ | **REMOVED**: br09_tillgsu (total assets) redundant with br10_eksu (equity), r=0.904. |
| `ny_solid_yoy_diff` | YoY differences in equity ratio. | Capital structure drift. |
| `ny_skuldgrd_yoy_diff` | YoY differences in leverage. | Capital structure drift. |
| `ratio_cash_liquidity_yoy_pct` | YoY change in quick ratio. | Liquidity trend. |
| `ratio_cash_liquidity_yoy_abs` | YoY absolute change in quick ratio. | Liquidity trend magnitude. |
| ~~`ratio_ebit_interest_cov_yoy_pct`~~ | ~~YoY change in EBIT coverage.~~ | **REMOVED**: `ratio_ebit_interest_cov` removed (low SHAP). |
| `dso_days_yoy_diff`, `inventory_days_yoy_diff`, `dpo_days_yoy_diff` | YoY differences in working capital efficiency. | **dso_days_yoy_diff ADDED BACK**: Complements DSO level; working capital trinity complete. |
| `current_ratio_yoy_pct` | YoY change in current ratio. | **ADDED**: Liquidity trend for current ratio. |
| `net_debt_to_ebitda_yoy_diff` | YoY change in net debt to EBITDA. | **ADDED**: Leverage trajectory; indicates improving/deteriorating debt service capacity. |

## Temporal Features (Multi-Year Lookback)

**Selection methodology**: The following 6 temporal features were selected from an initial set of 34 candidates using rigorous 5×3 nested cross-validation (see `notebooks/03_feature_selection.ipynb`), with additional pruning based on multicollinearity analysis. The selection process involved:
1. **Window selection**: For each metric/computation type, testing 2y vs 3y vs 5y windows
2. **Computation redundancy analysis**: Determining which computation types (CAGR, trend, volatility, average, drawdown) are necessary per metric
3. **Metric prioritization**: Testing which metrics contribute statistically significant predictive value
4. **Multicollinearity pruning**: Removing features based on highly correlated source metrics (dso_days, br09_tillgsu)

### Growth Metrics (CAGR)

Capture fundamental business momentum and growth trajectory.

| Feature | Window | Formula | Selection Rationale |
| --- | --- | --- | --- |
| `revenue_cagr_3y` | 3 years | CAGR of `rr01_ntoms` | Revenue growth is a primary credit risk indicator. 3y window balances recent trends with stability. |
| ~~`assets_cagr_3y`~~ | ~~3 years~~ | ~~CAGR of `br09_tillgsu`~~ | **REMOVED**: br09_tillgsu (total assets) redundant with br10_eksu (equity), r=0.904. |
| `equity_cagr_3y` | 3 years | CAGR of `br10_eksu` | Equity growth shows capital accumulation and retained earnings reinvestment. |
| `profit_cagr_3y` | 3 years | CAGR of `rr15_resar` | Profit growth trajectory is a direct indicator of business health. |

**Note**: 5y CAGR variants were tested but 3y windows provided optimal signal without overfitting. Growth metrics (CAGR) were found to be more informative than volatility or average-based features for these metrics. `assets_cagr_3y` was removed due to high correlation between its source metric (total assets) and equity.

### Risk Metrics (Drawdown)

Capture downside risk exposure and vulnerability to adverse conditions.

| Feature | Window | Definition | Selection Rationale |
| --- | --- | --- | --- |
| `revenue_drawdown_5y` | 5 years | Maximum drawdown of revenue within rolling window (min value / peak - 1) | High SHAP importance (rank 7). Captures revenue resilience and exposure to demand shocks. 5y window needed to capture full business cycles. |
| `equity_drawdown_5y` | 5 years | Maximum drawdown of equity within rolling window | Signals capital erosion risk. Works synergistically with equity CAGR to distinguish steady growth from volatile patterns. |

**Note**: Drawdown features were only kept for revenue and equity, where they showed statistically significant incremental value beyond CAGR alone.

### Working Capital Trends

Early warning signals for operational deterioration and cash flow stress.

| Feature | Window | Definition | Selection Rationale |
| --- | --- | --- | --- |
| ~~`dso_days_trend_3y`~~ | ~~3 years~~ | ~~Linear slope of days sales outstanding~~ | **REMOVED**: dso_days redundant with ratio_nwc_sales, r=-0.944. |
| `inventory_days_trend_3y` | 3 years | Linear slope of inventory days on hand | Rising inventory days may indicate obsolescence or demand weakness. |
| `dpo_days_trend_3y` | 3 years | Linear slope of days payables outstanding | Lengthening payment cycles can signal liquidity pressure. |

**Note**: Trend features were more informative than volatility or average-based features for working capital metrics. DSO trends were not included (user preference to avoid redundancy with temporal DSO features).

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

**NEW**: Based on academic literature emphasizing cash flow over accounting profit, OCF proxy features have been added.

| Feature | Definition | Purpose |
| --- | --- | --- |
| `ocf_proxy` | `(rr07_rorresul + rr05_avskriv) - ΔWorking Capital` | **ADDED**: Operating cash flow proxy. EBIT + Depreciation minus change in working capital. Captures cash generation vs. accounting profit. |
| `ratio_ocf_to_debt` | `ocf_proxy / (br13_ksksu + br15_lsksu)` | **ADDED**: OCF relative to total debt. Key solvency metric - measures debt service capacity from operations. |

**Rationale**: Accounting profit can be manipulated via accruals; cash flow is harder to game. OCF-to-debt ratio is standard in leveraged finance and credit analysis.

## Altman Z-Score Components

**NEW**: Components from the Altman Z-Score model for private companies (widely cited bankruptcy prediction model).

| Feature | Altman Component | Definition | Purpose |
| --- | --- | --- | --- |
| `working_capital_to_assets` | X₁ | `(br08_omstgsu - br13_ksksu) / br09_tillgsu` | **ADDED**: Liquidity normalized by firm size. Altman's X₁ component. |
| `retained_earnings_to_assets` | X₂ | `br10e_balres / br09_tillgsu` | **ADDED**: Cumulative profitability normalized by size. Altman's X₂. Differs from existing `ratio_retained_earnings_equity` (uses assets vs. equity denominator). |

**Note**: Other Altman components already exist:
- X₃ (EBIT/Assets) ≈ `ny_avktokap` (ROA)
- X₄ (Equity/Liabilities) = inverse of `ny_skuldgrd` (debt-to-equity)
- X₅ (Sales/Assets) = `ny_kapomsh` (asset turnover)

## Leverage & Financial Mismatch Features

**NEW**: Advanced leverage metrics from academic literature (Italian bankruptcy models).

| Feature | Definition | Purpose |
| --- | --- | --- |
| `financial_mismatch` | `(br13_ksksu - br08_omstgsu) / br09_tillgsu` | **ADDED**: Asset-liability maturity mismatch. Measures short-term liabilities exceeding current assets, normalized by total assets. Indicates refinancing risk. |
| `net_debt_to_ebitda` | `(br13_ksksu + br15_lsksu - br07_kplackaba) / (rr07_rorresul + rr05_avskriv)` | **ADDED**: Net debt (total debt minus liquid assets) relative to EBITDA. Classic leveraged finance metric. |
| `net_debt_to_ebitda_yoy_diff` | YoY change in `net_debt_to_ebitda` | **ADDED**: Leverage trajectory. Positive values indicate deteriorating debt service capacity. |

**Rationale**: Standard debt metrics may miss refinancing risk (short-term debt with illiquid assets) and overlook cash buffers (net debt vs. gross debt). These features are standard in credit rating agencies and leveraged finance.

## Credit Event History

| Feature | Definition / Purpose |
| --- | --- |
| ~~`years_since_last_credit_event`~~ | **REMOVED**: Potential data leakage - backward-looking feature that may reflect information not available at prediction time for independent companies. |
| ~~`event_count_total`~~ | **REMOVED**: Replaced with `event_count_last_5y` to prevent overfitting to rare historical events (only 0.16% of companies have events older than 5 years). |
| `event_count_last_5y` | Credit events within the past 5 years. Preferred over total count to avoid data leakage from sparse historical events. |

## Macro Features & Firm-Macro Comparisons

| Feature | Definition |
| --- | --- |
| `gdp_growth` | Annual GDP growth (market prices). **KEPT**: Core macroeconomic control despite low importance - provides theoretical completeness. |
| `interest_avg_short` | Annual average of short-term corporate borrowing rates (≤3m). **KEPT**: Core macroeconomic control despite low importance - provides theoretical completeness. |
| `term_spread` | Long – short rate spread. |
| ~~`inflation_yoy`~~ | ~~YoY CPI change (based on annual average KPIF).~~ **REMOVED**: Near-zero variance (0.0002) and low predictive value after pruning. |
| ~~`unemp_rate`~~ | ~~National unemployment level.~~ **REMOVED**: Low importance; macroeconomic conditions sufficiently captured by gdp_growth and interest_avg_short. |
| `revenue_beta_gdp_5y` | Rolling 5-year beta (cyclicality) of revenue growth vs. GDP growth. |

---

This catalogue is kept in sync with the project's feature engineering pipeline. Update this file whenever you modify the feature set.
