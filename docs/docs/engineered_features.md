# Engineered Feature Catalogue

This document summarises every engineered feature produced by `credit_risk_xai.features.engineer`. The tables are grouped by theme and note the intent and computation used in the pipeline. Unless otherwise stated, ratios use raw financial statement values in kSEK and are computed within each company-year panel.

**Note**: Following an iterative process of correlation analysis and feature importance evaluation, a total of 27 redundant or low-importance features have been removed to improve model efficiency while maintaining predictive performance.

## Cost Structure & Profitability Ratios

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| `ratio_personnel_cost` | `rr04_perskos / rr01_ntoms` | Share of net sales consumed by personnel costs. |
| `ratio_depreciation_cost` | `rr05_avskriv / rr01_ntoms` | Depreciation intensity relative to sales; proxy for capital intensity. |
| `ratio_other_operating_cost` | `rr06_rorkoov / rr01_ntoms` | Non-personnel operating cost burden. |
| `ratio_financial_cost` | `rr09_finkostn / rr01_ntoms` | Financial cost load relative to revenue. |
| `ratio_ebitda_margin` | `(rr07_rorresul + rr05_avskriv) / rr01_ntoms` | EBITDA margin; cushions volatility in EBIT. |
| `ratio_ebit_interest_cov` | `rr07_rorresul / (rr09_finkostn - rr09d_jfrstfin)` | Interest coverage based on EBIT. |
| `ratio_cash_interest_cov` | `br07b_kabasu / (rr09_finkostn - rr09d_jfrstfin)` | Cash-on-hand relative to annual financial costs. |
| `ratio_dividend_payout` | `rr00_utdbel / rr15_resar` | Dividend share of current-year profit (signals cash distribution). |
| `ratio_group_support` | `(br10f_kncbdrel + br10g_agtskel) / rr01_ntoms` | Extent of owner/group support relative to revenue. |
| `ratio_intragroup_financing_share` | `(rr08a_rteinknc + rr09a_rtekoknc) / (rr08_finintk + rr09_finkostn)` | Share of financing activity conducted with group companies. |

## Liquidity & Working Capital Efficiency

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| `ratio_cash_liquidity` | `(br07b_kabasu + br07a_kplacsu) / br13_ksksu` | Quick ratio (cash & near cash vs. current liabilities). |
| `dso_days` | `(br06g_kfordsu / rr01_ntoms) * 365` | Days sales outstanding; receivable collection speed. |
| `inventory_days` | `(br06c_lagersu / rr06a_prodkos) * 365` | Days inventory on hand. |
| `dpo_days` | `(br13a_ksklev / rr06a_prodkos) * 365` | Days payables outstanding; supplier payment terms. |
| `cash_conversion_cycle` | `dso_days + inventory_days - dpo_days` | Overall working capital efficiency. |
| `ratio_nwc_sales` | `(br06_lagerkford + br07_kplackaba - br13_ksksu) / rr01_ntoms` | Net working capital relative to sales. |

## Capital Structure Detail

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| `ratio_short_term_debt_share` | `br13_ksksu / (br13_ksksu + br15_lsksu)` | Share of debt maturing within 12 months. |
| `ratio_secured_debt_assets` | `(br14_kskkrin + br16_lskkrin) / br09_tillgsu` | Secured debt relative to total assets. |
| `ratio_retained_earnings_equity` | `br10e_balres / br10_eksu` | Retained earnings composition of equity. |
| `equity_to_sales` | `br10_eksu / rr01_ntoms` | Equity base relative to sales generation. |
| `equity_to_profit`| `br10_eksu / rr15_resar` | Equity base relative to net profit generation. |
| `assets_to_profit`| `br09_tillgsu / rr15_resar` | Asset base relative to net profit generation. |

## YoY Change & Trend Features

| Feature | Definition | Purpose |
| --- | --- | --- |
| `rr01_ntoms_yoy_pct, rr01_ntoms_yoy_abs` | YoY change in revenue. | Capture growth/direction of sales. |
| `rr07_rorresul_yoy_pct` | YoY changes in operating profit. | Profit momentum. |
| `br09_tillgsu_yoy_pct, br09_tillgsu_yoy_abs` | YoY asset changes. | Balance sheet expansion. |
| `ny_solid_yoy_diff` | YoY differences in equity ratio. | Capital structure drift. |
| `ny_skuldgrd_yoy_diff` | YoY differences in leverage. | Capital structure drift. |
| `ratio_cash_liquidity_yoy_pct` | YoY change in quick ratio. | Liquidity trend. |
| `ratio_cash_liquidity_yoy_abs` | YoY absolute change in quick ratio. | Liquidity trend magnitude. |
| `ratio_ebit_interest_cov_yoy_pct` | YoY change in EBIT coverage. | Debt service trend. |
| `dso_days_yoy_diff`, `inventory_days_yoy_diff`, `dpo_days_yoy_diff` | YoY differences in working capital efficiency. | Operational stress indicators. |

## Temporal Features (Multi-Year Lookback)

**Selection methodology**: The following 9 temporal features were selected from an initial set of 34 candidates using rigorous 5×3 nested cross-validation (see `notebooks/03_feature_selection.ipynb`). The selection process involved:
1. **Window selection**: For each metric/computation type, testing 2y vs 3y vs 5y windows
2. **Computation redundancy analysis**: Determining which computation types (CAGR, trend, volatility, average, drawdown) are necessary per metric
3. **Metric prioritization**: Testing which metrics contribute statistically significant predictive value

**Results**: The selected 9 features achieve 98.4% of the full model's performance (34 temporal features) while using only 26.5% of the features, with statistically significant improvement over baseline (p=0.0096).

### Growth Metrics (CAGR)

Capture fundamental business momentum and growth trajectory.

| Feature | Window | Formula | Selection Rationale |
| --- | --- | --- | --- |
| `revenue_cagr_3y` | 3 years | CAGR of `rr01_ntoms` | Revenue growth is a primary credit risk indicator. 3y window balances recent trends with stability. |
| `assets_cagr_3y` | 3 years | CAGR of `br09_tillgsu` | Asset growth signals expansion or contraction in business scale. |
| `equity_cagr_3y` | 3 years | CAGR of `br10_eksu` | Equity growth shows capital accumulation and retained earnings reinvestment. |
| `profit_cagr_3y` | 3 years | CAGR of `rr15_resar` | Profit growth trajectory is a direct indicator of business health. |

**Note**: 5y CAGR variants were tested but 3y windows provided optimal signal without overfitting. Growth metrics (CAGR) were found to be more informative than volatility or average-based features for these metrics.

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
| `dso_days_trend_3y` | 3 years | Linear slope of days sales outstanding | Increasing DSO signals collection difficulties and working capital strain. |
| `inventory_days_trend_3y` | 3 years | Linear slope of inventory days on hand | Rising inventory days may indicate obsolescence or demand weakness. |
| `dpo_days_trend_3y` | 3 years | Linear slope of days payables outstanding | Lengthening payment cycles can signal liquidity pressure. |

**Note**: Trend features were more informative than volatility or average-based features for working capital metrics. All three components (DSO, inventory, DPO) provide complementary signals about different aspects of working capital management.

### Excluded Temporal Feature Categories

The following temporal feature types were systematically excluded after testing:

| Category | Reason for Exclusion | Test Result |
| --- | --- | --- |
| Operating margin temporal features | Static `ny_rormarg` captures most signal; temporal derivatives add minimal value | AUC drop: +0.000062 (statistically zero) |
| Net margin temporal features | Static `ny_nettomarg` and YoY changes sufficient; trends/volatility/averages redundant | AUC drop: -0.000223 (negative = overfitting) |
| Leverage temporal features | Static `ny_skuldgrd` and YoY changes sufficient | AUC drop: -0.000215 (negative = overfitting) |
| Cash liquidity temporal features | Static ratio and YoY changes capture signal; trends/volatility/averages redundant | AUC drop: -0.000277 (negative = overfitting) |

**Key insight**: For margin and liquidity metrics, **static values + YoY changes** are more robust than multi-year trends/volatility/averages, which tend to overfit to validation-specific patterns.

## Credit Event History

| Feature | Definition / Purpose |
| --- | --- |
| `years_since_last_credit_event` | Years elapsed since the previous bankruptcy/reorganisation (NaN for first-time). |
| `event_count_total` | Cumulative number of credit events per company. |
| `event_count_last_5y` | Credit events within the past 5 years. |

## Macro Features & Firm-Macro Comparisons

| Feature | Definition |
| --- | --- |
| `gdp_growth` | Annual GDP growth (market prices). |
| `interest_avg_short` | Annual average of short-term corporate borrowing rates (≤3m). |
| `term_spread` | Long – short rate spread. |
| `inflation_yoy` | YoY CPI change (based on annual average KPIF). |
| `unemp_rate` | National unemployment level. |
| `revenue_beta_gdp_5y` | Rolling 5-year beta (cyclicality) of revenue growth vs. GDP growth. |

---

This catalogue is kept in sync with the project's feature engineering pipeline. Update this file whenever you modify the feature set.
