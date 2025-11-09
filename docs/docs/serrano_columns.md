# Serrano Dataset Column Catalogue

Reference guide to the raw columns loaded from the Serrano database (Version 3, October 2023). Monetary values are reported in **thousands of SEK (kSEK)** unless otherwise noted.

---

## Base Columns (`base_cols`)

| Column | Description | Units / Notes |
| --- | --- | --- |
| `ORGNR` | Corporate identity number (Organisationsnummer). | Identifier. |
| `ser_namn` | Business name as of 31 Dec for the reporting year. | Text. |
| `ser_year` | Calendar year the observation refers to. | Year (Int). |
| `bol_konkurs` | Bankruptcy indicator. | `1` = bankrupt during year, else `0`. |
| `bol_q80dat` | Reorganisation start date. | Date; non-null indicates event. |
| `ser_jurform` | Legal form code. | Lookup (JURFMT): `22` Partrederi, `23` Fund, `31` Partnership, `41` Bank AB, `42` Insurance AB, `49` Limited company, `51` Cooperative, `81` State unit, `82` Municipality, `83` Local federation, `84` County council, `85` Social insurance office, `87` Public institution, `88` Mortgage association, `89` Regional agency, `92` Mutual insurance, `93` Savings bank, `96` Foreign entity. |
| `ser_stklf` | Employment size class. | Lookup (STKL1FMT): `0` = 0, `1` = 1–4, `2` = 5–9, `3` = 10–19, `4` = 20–49, `5` = 50–99, `6` = 100–199, `7` = 200+, `9` = missing. |
| `bslov_antanst` | Employees per financial statements. | Headcount. |
| `ser_aktiv` | Active company indicator. | `1` = active, `0` = inactive. |
| `ser_nystartat` | Newly started flag. | `1` = start-up year (ABs from 1998 onward). |
| `ser_regdat` | Registration date (Bolagsverket). | Date. |
| `bransch_sni071_konv` | Primary SNI07 industry code (converted from SNI02 where needed). | Integer category. |
| `bransch_borsbransch_konv` | Aggregated sector grouping (11 categories). | Integer category, e.g. Industrial goods, Finance & Real estate. |
| `ser_laen` | County (län) code of registration. | Integer category. |
| `knc_kncfall` | Group situation indicator. | Category (e.g. independent, Swedish subsidiary). |

---

## Financial Ratios (`ny_cols`)

| Column | Description | Units / Notes |
| --- | --- | --- |
| `ny_kapomsh` | Asset turnover = Net sales ÷ Total assets. | Ratio. |
| `ny_avktokap` | Return on assets = Adjusted operating result ÷ Total assets. | Ratio / %. |
| `ny_rs` | Interest rate on debt = (Financial costs – adjustments) ÷ Adjusted liabilities. | Ratio / %. |
| `ny_skuldgrd` | Debt-to-equity ratio. | Ratio. |
| `ny_solid` | Equity-to-assets ratio. | Ratio / %. |
| `ny_avkegkap` | Return on equity = Adjusted net result ÷ Adjusted equity. | Ratio / %. |
| ~~`ny_rorkapo`~~ | ~~Working capital ÷ Net sales.~~ | **REMOVED**: Perfect correlation with engineered `ratio_nwc_sales` (r≈1.0), more NaNs (3665 vs 3525), slightly lower AUC. Replaced by `ratio_nwc_sales`. |
| `ny_kasslikv` | Quick ratio = (Current assets – inventories) ÷ Current liabilities. | Ratio. |
| ~~`ny_rormarg`~~ | ~~Operating margin = Adjusted operating profit ÷ Net sales.~~ | **REMOVED from model**: Redundant (multicollinearity with ny_nettomarg, r=0.979). |
| `ny_nettomarg` | Net margin. | Ratio / %. |
| `ny_vinstprc` | Profitability percentage (adjusted operating profit after financial income ÷ Net sales). | Ratio / %. |
| `ny_omspanst` | Net sales per employee. | kSEK per employee. |
| `ny_foradlvpanst` | Value added per employee. | kSEK per employee. |
| `ny_omsf` | Net sales growth vs. prior year. | % change. |
| `ny_anstf` | Employee count growth vs. prior year. | % change. |

---

## Income Statement Columns (`rr_cols`)

| Column | Description | Units / Notes |
| --- | --- | --- |
| `rr07_rorresul` | Operating profit/loss (recognised). | kSEK. |
| `rr08_finintk` | Financial income. | kSEK. |
| `rr09_finkostn` | Financial costs. | kSEK. |
| `rr10_finres_int` | Result after financial items. | kSEK. |
| `rr12_resefin` | Earnings after financial items. | kSEK. |
| `rr14_skatter` | Taxes. | kSEK. |
| `rr15_resar` | Net profit/loss for the year. | kSEK. |
| `rr01_ntoms` | Net sales. | kSEK. |
| `rr02_rointov` | Other operating income. | kSEK. |
| `rr05_avskriv` | Depreciation/amortisation. | kSEK. |
| `rr04_perskos` | Personnel costs (total). | kSEK. |
| `rr03_jfrst` | Items affecting comparability. | kSEK. |
| `rr06_rorkoov` | Other operating costs. | kSEK. |
| `rr13_bsldisp` | Appropriations. | kSEK. |
| `rr09d_jfrstfin` | Financial items affecting comparability. | kSEK. |
| `rr04a_loner` | Salaries and remuneration. | kSEK. |
| `rr04c_foradlv` | Value added (personnel-related). | kSEK. |
| `rr04b_sockostn` | Social security expenses. | kSEK. |
| `rr00_utdbel` | Dividends distributed. | kSEK. |
| `rr02a_lagerf` | Change in inventories of finished goods and WIP. | kSEK. |
| `rr02b_aktarb` | Capitalised work for own account. | kSEK. |
| `rr06a_prodkos` | Production costs. | kSEK. |
| `rr08d_resand` | Income from securities and receivables (other). | kSEK. |
| `rr08a_rteinknc` | Financial income – group companies. | kSEK. |
| `rr08b_rteinext` | Financial income – external. | kSEK. |
| `rr08c_rteinov` | Financial income – other. | kSEK. |
| `rr09a_rtekoknc` | Interest expenses – group companies. | kSEK. |
| `rr09b_rtekoext` | Interest expenses – external. | kSEK. |
| `rr09c_rtekoov` | Other financial expenses. | kSEK. |
| `rr13a_extraint` | Extraordinary income. | kSEK. |
| `rr13b_extrakos` | Extraordinary expenses. | kSEK. |
| `rr13c_kncbdr` | Group contributions (income). | kSEK. |
| `rr13d_agtsk` | Shareholder contributions recognised. | kSEK. |
| `rr13e_bsldisp` | Appropriations (detail). | kSEK. |

---

## Balance Sheet Columns (`br_cols`)

| Column | Description | Units / Notes |
| --- | --- | --- |
| `br01_imanlsu` | Intangible fixed assets (total). | kSEK. |
| `br03_maskiner` | Machinery and equipment. | kSEK. |
| `br02_matanlsu` | Tangible fixed assets (total). | kSEK. |
| `br04_fianltsu` | Financial fixed assets (total). | kSEK. |
| `br05_anltsu` | Fixed assets (aggregate). | kSEK. |
| `br08_omstgsu` | Current assets (total). | kSEK. |
| `br09_tillgsu` | Total assets (balance-sheet total). | kSEK. |
| `br10_eksu` | Total equity. | kSEK. |
| `br11_obeskres` | Untaxed reserves. | kSEK. |
| `br12_avssu` | Provisions. | kSEK. |
| `br14_kskkrin` | Current liabilities to credit institutions. | kSEK. |
| `br13_ksksu` | Current liabilities (total). | kSEK. |
| `br16_lskkrin` | Non-current liabilities to credit institutions. | kSEK. |
| `br15_lsksu` | Non-current liabilities (total). | kSEK. |
| `br17_eksksu` | Total equity and liabilities (should equal `br09_tillgsu`). | kSEK. |
| `br06_lagerkford` | Inventories and receivables (aggregate). | kSEK. |
| `br07_kplackaba` | Liquid assets (cash + near-cash). | kSEK. |
| `br06c_lagersu` | Inventories (goods). | kSEK. |
| `br06g_kfordsu` | Current receivables (total). | kSEK. |
| `br07a_kplacsu` | Short-term investments / securities. | kSEK. |
| `br07b_kabasu` | Cash and bank balances. | kSEK. |
| `br02a_byggmark` | Buildings and land. | kSEK. |
| `br02b_matanlov` | Other tangible fixed assets. | kSEK. |
| `br01a_foubautg` | Capitalised R&D expenditure. | kSEK. |
| `br01b_patlic` | Patents, licences, concessions. | kSEK. |
| `br01c_goodwill` | Goodwill. | kSEK. |
| `br01d_imanlov` | Other intangible assets. | kSEK. |
| `br04a_andknc` | Participations in group/associated companies. | kSEK. |
| `br04b_lfordknc` | Long-term receivables from group/associated companies. | kSEK. |
| `br04c_landelag` | Loans to partners and related parties. | kSEK. |
| `br04d_fianltov` | Other financial assets (long term). | kSEK. |
| `br06a_pagarb` | Work in progress. | kSEK. |
| `br06b_lagerov` | Other inventories. | kSEK. |
| `br06d_kundford` | Accounts receivable (trade). | kSEK. |
| `br06e_kfordknc` | Receivables from group/associated companies (current). | kSEK. |
| `br06f_kfordov` | Other current receivables (e.g. tax, prepayments). | kSEK. |
| `br10a_aktiekap` | Share capital. | kSEK. |
| `br10b_overkurs` | Share premium reserve. | kSEK. |
| `br10c_uppskr` | Revaluation reserve. | kSEK. |
| `br10d_ovrgbkap` | Other restricted equity (statutory reserve). | kSEK. |
| `br10e_balres` | Retained earnings (accumulated). | kSEK. |
| `br10f_kncbdrel` | Group contributions (capital). | kSEK. |
| `br10g_agtskel` | Shareholder contributions. | kSEK. |
| `br10h_resarb` | Profit/loss for the year (part of equity). | kSEK. |
| `br13a_ksklev` | Accounts payable (trade). | kSEK. |
| `br13b_kskknc` | Current liabilities to group/associated companies. | kSEK. |
| `br13c_kskov` | Other current liabilities (accruals, VAT, wages). | kSEK. |
| `br15a_lskknc` | Non-current liabilities to group/associated companies. | kSEK. |
| `br15b_lskov` | Other non-current liabilities. | kSEK. |
| `br15c_obllan` | Bond loans / other long-term debt. | kSEK. |

---

This catalogue mirrors the column definitions used by the processing pipeline (`cols_to_load`) so analysts can quickly review raw data fields before feature engineering. Update it if the underlying documentation or column set changes.***
