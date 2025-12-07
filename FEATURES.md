# Feature Definitions (V2 Model)

This document describes the 22 features used in the V2 credit risk model.

## Summary

| Category | Count | Features |
|----------|-------|----------|
| Base Features | 2 | company_age, sni_group_3digit |
| Nyckeltal Ratios | 7 | ny_kapomsh, ny_skuldgrd, ny_solid, ny_avkegkap, ny_kasslikv, ny_nettomarg, ny_omsf |
| Log-Transformed Size | 1 | log_total_assets |
| Altman/Ohlson Ratios | 6 | working_capital_ta, retained_earnings_ta, interest_coverage, ratio_cash_liquidity, gross_margin, dividend_yield |
| Working Capital Efficiency | 2 | dso_days, dpo_days |
| Temporal Features | 3 | revenue_cagr_3y, revenue_drawdown_5y, ebitda_volatility |
| Macroeconomic | 1 | term_spread |
| **Total** | **22** | |

---

## Base Features (2)

| Feature | Display Name | Definition | Units |
|---------|--------------|------------|-------|
| `company_age` | Company Age | Years since company registration date | years |
| `sni_group_3digit` | Industry (SNI) | 3-digit SNI (Swedish Standard Industrial Classification) code | categorical |

---

## Nyckeltal Ratios (7)

These are standard Swedish financial ratios from the Serrano database nyckeltal (key figures) table.

| Feature | Display Name | Formula | Source Columns | Units |
|---------|--------------|---------|----------------|-------|
| `ny_kapomsh` | Total Asset Turnover | Revenue / Total Assets | `rr01_ntoms / br09_tillgsu` | ratio |
| `ny_skuldgrd` | Debt Ratio | Total Liabilities / Total Assets | `(br11_obessksu + br13_ksksu) / br09_tillgsu` | ratio |
| `ny_solid` | Equity Ratio | Equity / Total Assets | `br10_eksu / br09_tillgsu` | ratio |
| `ny_avkegkap` | Return on Equity | Net Profit / Equity | `rr15_resar / br10_eksu` | ratio |
| `ny_kasslikv` | Quick Ratio | (Current Assets - Inventory) / Current Liabilities | `(br06_omssu - br05_lagersu) / br13_ksksu` | ratio |
| `ny_nettomarg` | Net Profit Margin | Net Profit / Revenue | `rr15_resar / rr01_ntoms` | ratio |
| `ny_omsf` | Revenue Growth (YoY) | Year-over-year change in net sales | Pre-computed in Serrano | ratio |

---

## Log-Transformed Size (1)

| Feature | Display Name | Formula | Source Column | Units |
|---------|--------------|---------|---------------|-------|
| `log_total_assets` | Log Total Assets | log(1 + Total Assets) | `log1p(br09_tillgsu)` | log(kSEK) |

**Note**: Total assets are in thousands of SEK (kSEK). The log transformation addresses right-skewness typical of firm size distributions. This corresponds to Ohlson's Size measure (W).

---

## Altman/Ohlson Aligned Ratios (6)

These features are designed to align with components of the Altman Z-score and Ohlson O-score models.

| Feature | Display Name | Formula | Alignment | Units |
|---------|--------------|---------|-----------|-------|
| `working_capital_ta` | Working Capital / TA | (Current Assets - Current Liabilities) / Total Assets | Altman X1 | ratio |
| `retained_earnings_ta` | Retained Earnings / TA | Retained Earnings / Total Assets | Altman X2 | ratio |
| `interest_coverage` | Interest Coverage | EBIT / Interest Expense, clipped to [-5, 20] | Solvency indicator | ratio |
| `ratio_cash_liquidity` | Cash Ratio | (Cash + Short-term Investments) / Current Liabilities | Liquidity indicator | ratio |
| `gross_margin` | Gross Margin | (Revenue - Cost of Goods Sold) / Revenue | Profitability | ratio |
| `dividend_yield` | Dividend Payer | Binary: 1 if dividends > 0, else 0 | Behavioral signal | binary |

### Source Column Mappings

| Feature | Source Columns |
|---------|----------------|
| `working_capital_ta` | `(br06_omssu - br13_ksksu) / br09_tillgsu` |
| `retained_earnings_ta` | `br10e_balres / br09_tillgsu` |
| `interest_coverage` | `rr07_rorresul / (rr09_finkostn - rr09d_jfrstfin)` |
| `ratio_cash_liquidity` | `(br07b_kabasu + br07a_kplacsu) / br13_ksksu` |
| `gross_margin` | `(rr01_ntoms - rr06a_prodkos) / rr01_ntoms` |
| `dividend_yield` | `1 if rr00_utdbel > 0 else 0` |

---

## Working Capital Efficiency (2)

| Feature | Display Name | Formula | Units |
|---------|--------------|---------|-------|
| `dso_days` | Days Sales Outstanding | (Accounts Receivable / Revenue) * 365 | days |
| `dpo_days` | Days Payables Outstanding | (Accounts Payable / COGS) * 365 | days |

### Source Column Mappings

| Feature | Source Columns |
|---------|----------------|
| `dso_days` | `(br06g_kfordsu / rr01_ntoms) * 365` |
| `dpo_days` | `(br13a_ksklev / rr06a_prodkos) * 365` |

---

## Temporal Features (3)

Multi-year features capturing growth, volatility, and downside risk.

| Feature | Display Name | Formula | Window | Units |
|---------|--------------|---------|--------|-------|
| `revenue_cagr_3y` | Revenue CAGR (3Y) | (Revenue_t / Revenue_{t-3})^(1/3) - 1 | 3 years | ratio |
| `revenue_drawdown_5y` | Revenue Drawdown (5Y) | Maximum decline from peak revenue over 5 years | 5 years | ratio (negative) |
| `ebitda_volatility` | EBITDA Volatility (3Y) | StdDev(EBITDA over 3 years) / Total Assets | 3 years | ratio |

### Notes

- **revenue_cagr_3y**: Compound annual growth rate of revenue. Requires 3 years of history.
- **revenue_drawdown_5y**: Measures worst revenue decline from historical peak. Always negative or zero. Higher magnitude indicates greater historical distress.
- **ebitda_volatility**: Operational risk proxy. EBITDA = Operating Profit + Depreciation (`rr07_rorresul + rr05_avskriv`).

---

## Macroeconomic (1)

| Feature | Display Name | Formula | Source | Units |
|---------|--------------|---------|--------|-------|
| `term_spread` | Term Spread | Long-term interest rate - Short-term interest rate | SCB Lending Rates | percentage points |

### Data Source

**Statistics Sweden (SCB)**: Lending Rates to Households and Non-Financial Corporations, Breakdown by Fixation Periods.

- **Long-term rate**: Mean annual rate for loans with fixation period "Over 5 years"
- **Short-term rate**: Mean annual rate for loans with fixation period "Up to 3 months (floating rate)"
- **Filter**: Non-financial corporations, new and renegotiated agreements

**URL**: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__FM__FM5001__FM5001C/RantaT01N/

---

## Serrano Database Column Reference

The features are computed from the following Serrano database tables:

| Prefix | Table | Description |
|--------|-------|-------------|
| `br*` | Balansräkning | Balance sheet items |
| `rr*` | Resultaträkning | Income statement items |
| `ny_*` | Nyckeltal | Pre-computed financial ratios |
| `ser_*` | Serrano metadata | Company identifiers, dates, status |

### Key Balance Sheet Columns (br*)

| Column | Description |
|--------|-------------|
| `br05_lagersu` | Inventory |
| `br06_omssu` | Current assets (total) |
| `br06g_kfordsu` | Accounts receivable |
| `br07a_kplacsu` | Short-term investments |
| `br07b_kabasu` | Cash and bank balances |
| `br09_tillgsu` | Total assets |
| `br10_eksu` | Total equity |
| `br10e_balres` | Retained earnings |
| `br11_obessksu` | Long-term liabilities |
| `br13_ksksu` | Current liabilities |
| `br13a_ksklev` | Accounts payable |

### Key Income Statement Columns (rr*)

| Column | Description |
|--------|-------------|
| `rr00_utdbel` | Dividends paid |
| `rr01_ntoms` | Net sales (revenue) |
| `rr05_avskriv` | Depreciation and amortization |
| `rr06a_prodkos` | Cost of goods sold |
| `rr07_rorresul` | Operating profit (EBIT) |
| `rr09_finkostn` | Financial costs |
| `rr09d_jfrstfin` | Interest income |
| `rr15_resar` | Net profit |

---

## Data Filters Applied

The model is trained on a filtered subset of the Serrano database:

| Filter | Column | Condition | Rationale |
|--------|--------|-----------|-----------|
| Active companies | `ser_aktiv` | = 1 | Exclude dormant/inactive firms |
| SME size | `sme_category` | IN ('Small', 'Medium') | Focus on SME segment |
| Credit reporting | `knc_kncfall` | = 1 | Independent companies only |
| Non-financial | `bransch_borsbransch_konv` | != '40.0' | Exclude financial services |

**Final dataset**: ~304,000 firm-year observations (1998-2023), 1.76% default rate.
