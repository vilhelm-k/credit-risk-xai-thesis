# Engineered Feature Catalogue

This document summarises every engineered feature produced by `credit_risk_xai.features.engineer`. The tables are grouped by theme and note the intent and computation used in the pipeline. Unless otherwise stated, ratios use raw financial statement values in kSEK and are computed within each company-year panel.

## Cost Structure & Profitability Ratios

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| `ratio_personnel_cost` | `rr04_perskos / rr01_ntoms` | Share of net sales consumed by personnel costs. |
| `ratio_depreciation_cost` | `rr05_avskriv / rr01_ntoms` | Depreciation intensity relative to sales; proxy for capital intensity. |
| `ratio_other_operating_cost` | `rr06_rorkoov / rr01_ntoms` | Non-personnel operating cost burden. |
| `ratio_financial_cost` | `rr09_finkostn / rr01_ntoms` | Financial cost load relative to revenue. |
| `ratio_ebitda_margin` | `(rr07_rorresul + rr05_avskriv) / rr01_ntoms` | EBITDA margin; cushions volatility in EBIT. |
| `ratio_ebit_interest_cov` | `rr07_rorresul / (rr09_finkostn - rr09d_jfrstfin)` | Interest coverage based on EBIT. |
| `ratio_ebitda_interest_cov` | `(rr07_rorresul + rr05_avskriv) / (rr09_finkostn - rr09d_jfrstfin)` | Interest coverage using EBITDA. |
| `ratio_cash_interest_cov` | `br07b_kabasu / (rr09_finkostn - rr09d_jfrstfin)` | Cash-on-hand relative to annual financial costs. |
| `ratio_dividend_payout` | `rr00_utdbel / rr15_resar` | Dividend share of current-year profit (signals cash distribution). |
| `ratio_group_support` | `(br10f_kncbdrel + br10g_agtskel) / rr01_ntoms` | Extent of owner/group support relative to revenue. |

## Liquidity & Working Capital Efficiency

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| `ratio_cash_liquidity` | `(br07b_kabasu + br07a_kplacsu) / br13_ksksu` | Quick ratio (cash & near cash vs. current liabilities). Computed for consistency with other working capital features. |
| `dso_days` | `(br06g_kfordsu / rr01_ntoms) * 365` | Days sales outstanding; receivable collection speed. |
| `inventory_days` | `(br06c_lagersu / rr06a_prodkos) * 365` | Days inventory on hand. |
| `dpo_days` | `(br13a_ksklev / rr06a_prodkos) * 365` | Days payables outstanding; supplier payment terms. |
| `ratio_nwc_sales` | `(br06_lagerkford + br07_kplackaba - br13_ksksu) / rr01_ntoms` | Net working capital relative to sales. |

## Capital Structure Detail

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| `ratio_short_term_debt_share` | `br13_ksksu / (br13_ksksu + br15_lsksu)` | Share of debt maturing within 12 months. |
| `ratio_secured_debt_assets` | `(br14_kskkrin + br16_lskkrin) / br09_tillgsu` | Secured debt relative to total assets. |
| `ratio_retained_earnings_equity` | `br10e_balres / br10_eksu` | Retained earnings composition of equity. |
| `ratio_share_capital_equity` | `br10a_aktiekap / br10_eksu` | Share capital proportion of equity. |

## YoY Change & Trend Features

| Feature | Definition | Purpose |
| --- | --- | --- |
| `rr01_ntoms_yoy_pct`, `rr01_ntoms_yoy_abs` | YoY % change and absolute change in revenue. | Capture growth/direction of sales. |
| `rr07_rorresul_yoy_pct`, `rr07_rorresul_yoy_abs` | YoY changes in operating profit. | Profit momentum. |
| `br09_tillgsu_yoy_pct`, `br09_tillgsu_yoy_abs` | YoY asset changes. | Balance sheet expansion. |
| `ny_solid_yoy_diff`, `ny_skuldgrd_yoy_diff` | YoY differences in equity ratio and leverage. | Capital structure drift. |
| `ratio_cash_liquidity_yoy_pct`, `ratio_cash_liquidity_yoy_abs` | YoY change in quick ratio. | Liquidity trend. |
| `ratio_ebit_interest_cov_yoy_pct` | YoY change in EBIT coverage. | Debt service trend. |
| `dso_days_yoy_diff`, `inventory_days_yoy_diff`, `dpo_days_yoy_diff` | YoY differences in working capital efficiency. | Operational stress indicators. |

### Rolling CAGR (per company)

| Feature | Window | Formula |
| --- | --- | --- |
| `revenue_cagr_3y`, `revenue_cagr_5y` | 3 / 5 years | CAGR of `rr01_ntoms`. |
| `assets_cagr_3y`, `assets_cagr_5y` | 3 / 5 years | CAGR of `br09_tillgsu`. |
| `equity_cagr_3y`, `equity_cagr_5y` | 3 / 5 years | CAGR of `br10_eksu`. |
| `profit_cagr_3y`, `profit_cagr_5y` | 3 / 5 years | CAGR of `rr15_resar`. |

### Rolling Slopes & Trends

| Feature | Window | Definition |
| --- | --- | --- |
| `ny_rormarg_trend_3y`, `ny_rormarg_trend_5y` | 3 / 5 years | Linear slope of operating margin (ny_rormarg). |
| `ny_skuldgrd_trend_3y`, `ny_skuldgrd_trend_5y` | 3 / 5 years | Slope of debt/equity ratio. |
| `ratio_cash_liquidity_trend_3y`, `ratio_cash_liquidity_trend_5y` | 3 / 5 years | Liquidity slope. |
| `dso_days_trend_3y`, `inventory_days_trend_3y`, `dpo_days_trend_3y` | 3 years | Slopes for working capital efficiency metrics. |

### Rolling Volatility, Averages & Drawdowns

| Feature | Window | Definition |
| --- | --- | --- |
| `ny_rormarg_vol_3y`, `ny_rormarg_vol_5y` | 3 / 5 years | Rolling standard deviation of operating margin. |
| `ny_skuldgrd_vol_3y`, `ny_skuldgrd_vol_5y` | 3 / 5 years | Rolling std of leverage ratio. |
| `ratio_cash_liquidity_vol_3y` | 3 years | Rolling std of quick ratio. |
| `ny_rormarg_avg_2y`, `ny_rormarg_avg_5y` | 2 / 5 years | Rolling mean of operating margin. |
| `ratio_cash_liquidity_avg_2y`, `ratio_cash_liquidity_avg_5y` | 2 / 5 years | Rolling mean quick ratio. |
| `revenue_drawdown_5y`, `equity_drawdown_5y` | 5 years | Max drawdown of revenue / equity inside rolling window. |

## Credit Event History

| Feature | Definition / Purpose |
| --- | --- |
| `years_since_last_credit_event` | Years elapsed since the previous bankruptcy/reorganisation (NaN for first-time). |
| `last_event_within_1y`, `last_event_within_2y`, `last_event_within_3y`, `last_event_within_5y` | Binary flags indicating recency of prior event. |
| `event_count_total` | Cumulative number of credit events per company. |
| `event_count_last_5y` | Credit events within the past 5 years. |
| `ever_failed` | Indicator the company experienced at least one event. |

## Macro Features & Firm-Macro Comparisons

| Feature | Definition |
| --- | --- |
| `gdp_growth`, `gdp_growth_3y_avg` | Annual GDP growth and 3-year average (market prices). |
| `interest_avg_short`, `interest_avg_medium`, `interest_avg_long` | Annual averages of corporate borrowing rates (≤3m, 1–5y, >5y). |
| `interest_delta_short` | YoY change in short-term borrowing rate. |
| `term_spread`, `term_spread_delta` | Long – short rate spread and its YoY change. |
| `inflation_yoy`, `inflation_trailing_3y` | YoY CPI change (based on annual average KPIF) and 3-year mean. |
| `unemp_rate`, `unemp_delta` | National unemployment level and YoY change. |
| `real_revenue_growth` | `rr01_ntoms_yoy_pct - inflation_yoy`. |
| `revenue_vs_gdp` | `rr01_ntoms_yoy_pct - gdp_growth`. |
| `profit_vs_gdp` | `rr07_rorresul_yoy_pct - gdp_growth`. |
| `correlation_revenue_gdp_5y` | Rolling 5-year correlation between revenue YoY growth and GDP growth (requires ≥4 overlapping points). |

---

Maintaining this catalogue ensures new features added to `credit_risk_xai.features.engineer` are documented and can be evaluated for relevance before inclusion in modeling experiments. Update this file whenever you modify the feature set.***
