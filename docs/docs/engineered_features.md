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
| `rr01_ntoms_yoy_pct` | YoY % change in revenue. | Capture growth/direction of sales. |
| `rr07_rorresul_yoy_pct` | YoY changes in operating profit. | Profit momentum. |
| `br09_tillgsu_yoy_pct` | YoY asset changes. | Balance sheet expansion. |
| `ny_solid_yoy_diff` | YoY differences in equity ratio. | Capital structure drift. |
| `ny_skuldgrd_yoy_diff` | YoY differences in leverage. | Capital structure drift. |
| `ratio_cash_liquidity_yoy_pct` | YoY change in quick ratio. | Liquidity trend. |
| `ratio_cash_liquidity_yoy_abs` | YoY absolute change in quick ratio. | Liquidity trend magnitude. |
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
| `ny_nettomarg_trend_3y`, `ny_nettomarg_trend_5y` | 3 / 5 years | Linear slope of net margin (ny_nettomarg). |
| `ny_skuldgrd_trend_3y`, `ny_skuldgrd_trend_5y` | 3 / 5 years | Slope of debt/equity ratio. |
| `ratio_cash_liquidity_trend_3y`, `ratio_cash_liquidity_trend_5y` | 3 / 5 years | Liquidity slope. |
| `dso_days_trend_3y`, `inventory_days_trend_3y`, `dpo_days_trend_3y` | 3 years | Slopes for working capital efficiency metrics. |

### Rolling Volatility, Averages & Drawdowns

| Feature | Window | Definition |
| --- | --- | --- |
| `ny_rormarg_vol_3y`, `ny_rormarg_vol_5y` | 3 / 5 years | Rolling standard deviation of operating margin. |
| `ny_nettomarg_vol_3y`, `ny_nettomarg_vol_5y` | 3 / 5 years | Rolling standard deviation of net margin. |
| `ny_skuldgrd_vol_3y`, `ny_skuldgrd_vol_5y` | 3 / 5 years | Rolling std of leverage ratio. |
| `ratio_cash_liquidity_vol_3y` | 3 years | Rolling std of quick ratio. |
| `ny_rormarg_avg_2y`, `ny_rormarg_avg_5y` | 2 / 5 years | Rolling mean of operating margin. |
| `ny_nettomarg_avg_2y`, `ny_nettomarg_avg_5y` | 2 / 5 years | Rolling mean of net margin. |
| `ratio_cash_liquidity_avg_2y`, `ratio_cash_liquidity_avg_5y` | 2 / 5 years | Rolling mean quick ratio. |
| `revenue_drawdown_5y`, `equity_drawdown_5y` | 5 years | Max drawdown of revenue / equity inside rolling window. |

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
