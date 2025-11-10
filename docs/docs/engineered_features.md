# Engineered Feature Catalogue

This document summarises every engineered feature produced by `credit_risk_xai.features.engineer`. The tables are grouped by theme and note the intent and computation used in the pipeline. Unless otherwise stated, ratios use raw financial statement values in kSEK and are computed within each company-year panel.

**Model Scope**: This credit risk model applies to **independent companies only** (filtered to `knc_kncfall==1`). Subsidiaries, parent companies, and other organizational structures are excluded because they exhibit fundamentally different risk profiles due to intragroup financing, parent support mechanisms, and consolidated financial statements. This filtering ensures that financial ratios have consistent interpretation across all companies in the dataset.

**Base Features**: The model includes firm characteristics (`bslov_antanst`, `company_age`), industry controls (`bransch_sni071_konv`, `bransch_borsbransch_konv`), and macroeconomic conditions (`gdp_growth`, `interest_avg_short`, `term_spread`). Geographic controls (`ser_laen` - county code) and organizational structure (`knc_kncfall`) were excluded: the former due to low predictive value, the latter because it's used as a data filter rather than a feature.

**Note**: Following an iterative process of correlation analysis and feature importance evaluation, a total of 54 redundant or low-importance features have been removed to improve model efficiency while maintaining predictive performance. This includes:
- **Phase 1 pruning**: 36 features removed based on perfect correlations, near-zero variance, and low ablation impact
- **Phase 2 pruning** (validated via 5-fold CV): 7 additional features removed based on multicollinearity analysis (correlation threshold |r| > 0.85) including: `ratio_personnel_cost`, `ratio_financial_cost`, `equity_to_sales`, `dso_days` (and derivatives), `br09_tillgsu` (and derivatives), `ny_rormarg`, and `inflation_yoy`
- **Group-structure features**: 2 features (`ratio_group_support`, `ratio_intragroup_financing_share`) removed as they are only relevant for subsidiaries/parent companies
- **Organizational filter**: 1 feature (`knc_kncfall`) moved from model feature to data filter
- **Data leakage prevention**: 1 feature (`years_since_last_credit_event`) removed as it may not be available at prediction time for independent companies
- **Low-importance features**: 5 features removed based on empirical importance analysis: `ser_laen` (geographic control), `unemp_rate` (redundant macro control), `rr07_rorresul` (redundant with profitability ratios), `inventory_days` (captured by derivatives), `rr01_ntoms` (size captured by employee count and equity)

## Cost Structure & Profitability Ratios

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| ~~`ratio_personnel_cost`~~ | ~~`rr04_perskos / rr01_ntoms`~~ | **REMOVED**: Redundant (multicollinearity with ny_nettomarg, r=0.92; low unique contribution after pruning). |
| `ratio_depreciation_cost` | `rr05_avskriv / rr01_ntoms` | Depreciation intensity relative to sales; proxy for capital intensity. Bridges EBITDA→EBIT. |
| ~~`ratio_other_operating_cost`~~ | ~~`rr06_rorkoov / rr01_ntoms`~~ | **REMOVED**: Lowest ablation impact (-0.000446), 3 red flags, SHAP=0.010. Captured by other profitability metrics. |
| ~~`ratio_financial_cost`~~ | ~~`rr09_finkostn / rr01_ntoms`~~ | **REMOVED**: Redundant (multicollinearity & low unique contribution after pruning). |
| ~~`ratio_ebitda_margin`~~ | ~~`(rr07_rorresul + rr05_avskriv) / rr01_ntoms`~~ | **REMOVED**: Near-perfect correlation with `ny_rormarg` (r=0.998). |
| `ratio_ebit_interest_cov` | `rr07_rorresul / (rr09_finkostn - rr09d_jfrstfin)` | Interest coverage based on EBIT. |
| `ratio_cash_interest_cov` | `br07b_kabasu / (rr09_finkostn - rr09d_jfrstfin)` | Cash-on-hand relative to annual financial costs. |
| `ratio_dividend_payout` | `rr00_utdbel / rr15_resar` | Dividend share of current-year profit (signals cash distribution). |
| ~~`ratio_group_support`~~ | ~~`(br10f_kncbdrel + br10g_agtskel) / rr01_ntoms`~~ | **REMOVED**: Only relevant for subsidiaries/group companies. Model applies to independent companies (knc_kncfall==1) only. |
| ~~`ratio_intragroup_financing_share`~~ | ~~`(rr08a_rteinknc + rr09a_rtekoknc) / (rr08_finintk + rr09_finkostn)`~~ | **REMOVED**: Only relevant for subsidiaries/group companies. Model applies to independent companies (knc_kncfall==1) only. |

## Liquidity & Working Capital Efficiency

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| `ratio_cash_liquidity` | `(br07b_kabasu + br07a_kplacsu) / br13_ksksu` | Quick ratio (cash & near cash vs. current liabilities). |
| ~~`dso_days`~~ | ~~`(br06g_kfordsu / rr01_ntoms) * 365`~~ | **REMOVED**: Redundant (multicollinearity with ratio_nwc_sales, r=-0.944). |
| ~~`inventory_days`~~ | ~~`(br06c_lagersu / rr06a_prodkos) * 365`~~ | **REMOVED**: Low importance; information captured by derivatives (inventory_days_yoy_diff, inventory_days_trend_3y). |
| `dpo_days` | `(br13a_ksklev / rr06a_prodkos) * 365` | Days payables outstanding; supplier payment terms. |
| ~~`cash_conversion_cycle`~~ | ~~`dso_days + inventory_days - dpo_days`~~ | **REMOVED**: High correlation with `dso_days` (r=0.971). |
| `ratio_nwc_sales` | `(br06_lagerkford + br07_kplackaba - br13_ksksu) / rr01_ntoms` | Net working capital relative to sales. |

## Capital Structure Detail

| Feature | Definition / Formula | Purpose |
| --- | --- | --- |
| `ratio_short_term_debt_share` | `br13_ksksu / (br13_ksksu + br15_lsksu)` | Share of debt maturing within 12 months. |
| `ratio_secured_debt_assets` | `(br14_kskkrin + br16_lskkrin) / br09_tillgsu` | Secured debt relative to total assets. |
| `ratio_retained_earnings_equity` | `br10e_balres / br10_eksu` | Retained earnings composition of equity. |
| ~~`equity_to_sales`~~ | ~~`br10_eksu / rr01_ntoms`~~ | **REMOVED**: Redundant (multicollinearity & low unique contribution after pruning). |
| `equity_to_profit`| `br10_eksu / rr15_resar` | Equity base relative to net profit generation. |
| `assets_to_profit`| `br09_tillgsu / rr15_resar` | Asset base relative to net profit generation. |

## Removed Raw Financial Statement Values

The following raw financial statement values are excluded from the model despite being used in feature engineering:

| Feature | Definition | Removal Rationale |
| --- | --- | --- |
| ~~`rr01_ntoms`~~ | Net revenue (total sales) in kSEK | **REMOVED**: Firm size controlled by `bslov_antanst` (employees) and `br10_eksu` (equity); revenue scale information captured by derivatives (`revenue_cagr_3y`, `revenue_drawdown_5y`, `rr01_ntoms_yoy_abs`). Raw revenue magnitude adds no incremental predictive value after controlling for ratios and growth. |
| ~~`br09_tillgsu`~~ | Total assets in kSEK | **REMOVED**: High correlation with `br10_eksu` (r=0.904); balance sheet size sufficiently represented by equity. Avoiding multicollinearity. |
| ~~`rr07_rorresul`~~ | Operating profit/loss in kSEK | **REMOVED**: Redundant with profitability ratios (`ny_nettomarg`, `ny_avkegkap`); profit dynamics captured by `rr07_rorresul_yoy_pct`. Raw absolute profit less informative than margins and growth rates. |

**Rationale for excluding raw values**: After controlling for firm size (employees, equity), efficiency ratios (margins, turnover), and temporal dynamics (growth, trends), the absolute magnitude of financial statement line items provides minimal incremental information. Modern credit risk models prioritize **relative performance** (ratios) and **change dynamics** (trends, growth) over absolute scale.

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
| `ratio_ebit_interest_cov_yoy_pct` | YoY change in EBIT coverage. | Debt service trend. |
| ~~`dso_days_yoy_diff`~~, `inventory_days_yoy_diff`, `dpo_days_yoy_diff` | YoY differences in working capital efficiency. | **dso_days_yoy_diff REMOVED**: dso_days redundant with ratio_nwc_sales. |

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

**Note**: Trend features were more informative than volatility or average-based features for working capital metrics. Two components (inventory, DPO) provide complementary signals about working capital management; DSO removed due to redundancy with net working capital ratio.

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
