"""
Central configuration for the credit-risk XAI project.

This module stores path definitions, column registries, feature lists, and
modeling constants so that notebooks, scripts, and command-line utilities all
share a single source of truth.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MACRO_CACHE_PATH = INTERIM_DATA_DIR / "macro_annual.parquet"
BASE_CACHE_PATH = INTERIM_DATA_DIR / "serrano_base.parquet"
FEATURE_CACHE_PATH = PROCESSED_DATA_DIR / "serrano_features.parquet"

# -----------------------------------------------------------------------------
# Column selections
# -----------------------------------------------------------------------------

BASE_COLS = [
    "ORGNR",
    "ser_namn",
    "ser_year",
    "bol_konkurs",
    "bol_q80dat",
    "ser_jurform",
    "ser_stklf",
    "bslov_antanst",
    "ser_aktiv",
    "ser_nystartat",
    "ser_regdat",
    "bransch_sni071_konv",
    "bransch_borsbransch_konv",
    "ser_laen",
    "knc_kncfall",
]

NY_COLS = [
    "ny_kapomsh",
    "ny_avktokap",
    "ny_rs",
    "ny_skuldgrd",
    "ny_solid",
    "ny_avkegkap",
    # "ny_rorkapo",  # REMOVED: Perfect correlation with ratio_nwc_sales (r≈1.0), more NaNs, slightly lower AUC
    "ny_kasslikv",
    # "ny_rormarg",  # REMOVED: Redundant (multicollinearity with ny_nettomarg, r=0.979)
    "ny_nettomarg",
    # "ny_vinstprc", # REMOVED: Redundant with ny_nettomarg
    "ny_omspanst",
    "ny_foradlvpanst",
    "ny_omsf",
    "ny_anstf",
]

KEPT_RAW_COLS = [
    "rr01_ntoms",  # Kept for feature engineering but excluded from model features (see exclusion list below)
    "br09_tillgsu",  # Kept for feature engineering but excluded from model features (see exclusion list below)
    "br10_eksu",
    "bslov_antanst",
    "br07b_kabasu",
    # "br13_ksksu",  # REMOVED: Bottom 20% in both SHAP (0.0096) and tree importance (1465)
    # "br15_lsksu",  # REMOVED: Bottom 20% in both SHAP (0.0136) and tree importance (1297)
    "rr07_rorresul",  # Kept for feature engineering but excluded from model features (see exclusion list below)
    "rr15_resar",
]

# Raw columns to exclude from modeling (kept for feature engineering but not used as model features)
EXCLUDED_RAW_COLS = [
    "br09_tillgsu",  # Redundant (multicollinearity with br10_eksu, r=0.904); br10_eksu provides similar signal
    "rr01_ntoms",  # Size controlled by bslov_antanst + br10_eksu; scale information captured by derivatives (revenue_cagr_3y, revenue_drawdown_5y, rr01_ntoms_yoy_abs)
    "rr07_rorresul",  # Redundant with profitability ratios (ny_nettomarg, ny_avkegkap); information captured by rr07_rorresul_yoy_pct
]

RR_SOURCE_COLS = [
    "rr02_rointov",
    "rr05_avskriv",
    "rr04_perskos",
    "rr03_jfrst",
    "rr06_rorkoov",
    "rr09_finkostn",
    "rr09d_jfrstfin",
    "rr04a_loner",
    "rr04c_foradlv",
    "rr04b_sockostn",
    "rr00_utdbel",
    "rr02a_lagerf",
    "rr02b_aktarb",
    "rr06a_prodkos",
    "rr08d_resand",
    "rr08a_rteinknc",
    "rr08b_rteinext",
    "rr08c_rteinov",
    "rr09a_rtekoknc",
    "rr09b_rtekoext",
    "rr09c_rtekoov",
    "rr08_finintk",
    "rr10_finres_int",
    "rr12_resefin",
    "rr14_skatter",
    "rr13_bsldisp",
    "rr13a_extraint",
    "rr13b_extrakos",
    "rr13c_kncbdr",
    "rr13d_agtsk",
    "rr13e_bsldisp",
]

BR_SOURCE_COLS = [
    "br01_imanlsu",
    "br03_maskiner",
    "br02_matanlsu",
    "br04_fianltsu",
    "br05_anltsu",
    "br08_omstgsu",
    "br11_obeskres",
    "br12_avssu",
    "br14_kskkrin",
    "br16_lskkrin",
    "br17_eksksu",
    "br06_lagerkford",
    "br07_kplackaba",
    "br06c_lagersu",
    "br06g_kfordsu",
    "br07a_kplacsu",
    "br02a_byggmark",
    "br02b_matanlov",
    "br01a_foubautg",
    "br01b_patlic",
    "br01c_goodwill",
    "br01d_imanlov",
    "br04a_andknc",
    "br04b_lfordknc",
    "br04c_landelag",
    "br04d_fianltov",
    "br06a_pagarb",
    "br06b_lagerov",
    "br06d_kundford",
    "br06e_kfordknc",
    "br06f_kfordov",
    "br10a_aktiekap",
    "br10b_overkurs",
    "br10c_uppskr",
    "br10d_ovrgbkap",
    "br10e_balres",
    "br10f_kncbdrel",
    "br10g_agtskel",
    "br10h_resarb",
    "br13a_ksklev",
    "br13b_kskknc",
    "br13c_kskov",
    "br15a_lskknc",
    "br15b_lskov",
    "br15c_obllan",
]

COLS_TO_LOAD = list(
    dict.fromkeys(BASE_COLS + NY_COLS + KEPT_RAW_COLS + RR_SOURCE_COLS + BR_SOURCE_COLS)
)

# -----------------------------------------------------------------------------
# Engineered feature registries
# -----------------------------------------------------------------------------

RATIO_FEATURE_NAMES = [
    # "ratio_personnel_cost",  # REMOVED: Redundant (multicollinearity & low unique contribution)
    "ratio_depreciation_cost",
    # "ratio_other_operating_cost",  # REMOVED: Lowest individual impact (-0.000446 AUC), 3 red flags, SHAP=0.010
    # "ratio_financial_cost",  # REMOVED: Redundant (multicollinearity & low unique contribution)
    # "ratio_ebitda_margin",  # REMOVED: Near-perfect correlation with ny_rormarg (r=0.998)
    "ratio_ebit_interest_cov",
    # "ratio_ebitda_interest_cov",  # REMOVED: Highly correlated with ratio_ebit_interest_cov (r=0.99)
    "ratio_cash_interest_cov",
    "ratio_cash_liquidity",
    "ratio_nwc_sales",
    "ratio_short_term_debt_share",
    "ratio_secured_debt_assets",
    "ratio_retained_earnings_equity",
    # "ratio_share_capital_equity", # REMOVED: Low importance
    "ratio_dividend_payout",
    # "ratio_group_support",  # REMOVED: Only relevant for subsidiaries/group companies; filtered to independent companies only
    # "equity_to_sales",  # REMOVED: Redundant (multicollinearity & low unique contribution)
    "equity_to_profit",
    "assets_to_profit",
    # "ratio_intragroup_financing_share",  # REMOVED: Only relevant for subsidiaries/group companies; filtered to independent companies only
]

LIQUIDITY_EFFICIENCY_FEATURES = [
    # "dso_days",  # REMOVED: Redundant (multicollinearity with ratio_nwc_sales, r=-0.944)
    # "inventory_days",  # REMOVED: Low importance; information captured by derivatives (inventory_days_yoy_diff, inventory_days_trend_3y)
    "dpo_days",
    # "cash_conversion_cycle",  # REMOVED: High correlation with dso_days (r=0.971)
]

TREND_FEATURE_NAMES = [
    # "rr01_ntoms_yoy_pct",  # REMOVED: Perfect correlation with ny_omsf (r=1.0)
    "rr01_ntoms_yoy_abs",
    "rr07_rorresul_yoy_pct",
    # "rr07_rorresul_yoy_abs", # REMOVED: Redundant
    # "br09_tillgsu_yoy_pct",  # REMOVED: br09_tillgsu redundant with br10_eksu
    # "br09_tillgsu_yoy_abs",  # REMOVED: br09_tillgsu redundant with br10_eksu
    "ny_solid_yoy_diff",
    "ny_skuldgrd_yoy_diff",
    "ratio_cash_liquidity_yoy_pct",
    "ratio_cash_liquidity_yoy_abs", # KEPT: As per user request
    "ratio_ebit_interest_cov_yoy_pct",
    # "dso_days_yoy_diff",  # REMOVED: dso_days redundant with ratio_nwc_sales
    "inventory_days_yoy_diff",
    "dpo_days_yoy_diff",
]

# Temporal features selected via 5×3 nested CV feature selection (Experiment 1-3)
# See notebooks/03_feature_selection.ipynb for detailed analysis
# 6 features selected (dso_days and br09_tillgsu features removed due to redundancy)
TEMPORAL_FEATURE_NAMES = [
    # Growth metrics (CAGR) - capture fundamental business momentum
    "revenue_cagr_3y",
    # "assets_cagr_3y",  # REMOVED: br09_tillgsu (total assets) redundant with br10_eksu (equity)
    "equity_cagr_3y",
    "profit_cagr_3y",
    # Risk metrics (drawdown) - capture downside exposure
    "revenue_drawdown_5y",
    "equity_drawdown_5y",
    # Working capital trends - early warning signals for operational deterioration
    # "dso_days_trend_3y",  # REMOVED: dso_days redundant with ratio_nwc_sales
    "inventory_days_trend_3y",
    "dpo_days_trend_3y",
]

CRISIS_FEATURE_NAMES = [
    # "years_since_last_credit_event",  # REMOVED: Potential data leakage - backward-looking feature that may reflect information not available at prediction time
    # "last_event_within_1y",  # REMOVED: Redundant with years_since_last_credit_event
    # "last_event_within_2y",  # REMOVED: Redundant with years_since_last_credit_event
    # "last_event_within_3y",  # REMOVED: Redundant with years_since_last_credit_event
    # "last_event_within_5y",  # REMOVED: Redundant with years_since_last_credit_event
    # "event_count_total",  # REMOVED: Replaced with event_count_last_5y to prevent overfitting to rare historical events
    "event_count_last_5y",
    # "ever_failed",  # REMOVED: Zero importance, redundant with event_count_total
]

MACRO_FEATURE_NAMES = [
    "gdp_growth",  # KEPT: Core macroeconomic control despite low importance
    # "gdp_growth_3y_avg", # REMOVED: Redundant
    "interest_avg_short",  # KEPT: Core macroeconomic control despite low importance
    # "interest_avg_medium",  # REMOVED: Highly correlated with short and long rates (r>0.95)
    # "interest_avg_long",  # REMOVED: Highly correlated with short rate (r=0.88)
    # "interest_delta_short", # REMOVED: Redundant
    "term_spread",
    # "term_spread_delta", # REMOVED: Redundant
    # "inflation_yoy",  # REMOVED: Near-zero variance (0.0002), low importance
    # "inflation_trailing_3y",  # REMOVED: Highly correlated with inflation_yoy (r=0.86)
    # "unemp_rate",  # REMOVED: Low importance; macro conditions captured by gdp_growth and interest_avg_short
    # "unemp_delta", # REMOVED: Redundant
    # "real_revenue_growth", # REMOVED: Redundant
    # "revenue_vs_gdp",  # REMOVED: Nearly identical to real_revenue_growth (r=0.999996)
    # "profit_vs_gdp", # REMOVED: Redundant
    "revenue_beta_gdp_5y",
]

ENGINEERED_FEATURE_NAMES = (
    RATIO_FEATURE_NAMES
    + LIQUIDITY_EFFICIENCY_FEATURES
    + TREND_FEATURE_NAMES
    + TEMPORAL_FEATURE_NAMES
    + CRISIS_FEATURE_NAMES
    + MACRO_FEATURE_NAMES
)

# -----------------------------------------------------------------------------
# Feature groupings by financial statement source (for correlation analysis)
# -----------------------------------------------------------------------------

# Balance sheet features (br prefix + balance sheet items from NY_COLS and KEPT_RAW_COLS)
BALANCE_SHEET_FEATURES = (
    BR_SOURCE_COLS
    + [
        # From KEPT_RAW_COLS
        "br09_tillgsu",  # Total assets
        "br10_eksu",     # Equity
        "br07b_kabasu",  # Cash & bank
        "br13_ksksu",    # Current liabilities
        "br15_lsksu",    # Long-term liabilities
        # From NY_COLS (balance sheet ratios)
        "ny_kapomsh",    # Capital turnover
        "ny_skuldgrd",   # Debt ratio
        "ny_solid",      # Equity ratio
        "ny_kasslikv",   # Cash liquidity
    ]
)

# Income statement features (rr prefix + income statement items from NY_COLS and KEPT_RAW_COLS)
INCOME_STATEMENT_FEATURES = (
    RR_SOURCE_COLS
    + [
        # From KEPT_RAW_COLS
        "rr01_ntoms",    # Net revenue
        "rr07_rorresul", # Operating profit
        "rr15_resar",    # Profit after financial items
        # From NY_COLS (income statement ratios)
        "ny_avktokap",   # Return on total capital
        "ny_avkegkap",   # Return on equity
        "ny_rorkapo",    # Return on invested capital
        "ny_rormarg",    # Operating margin
        "ny_nettomarg",  # Net margin
        "ny_vinstprc",   # Profit percentage
    ]
)

# Derived ratios combining balance sheet and income statement (for separate analysis)
DERIVED_RATIO_FEATURES = RATIO_FEATURE_NAMES + LIQUIDITY_EFFICIENCY_FEATURES

# Working capital efficiency metrics (subset of derived ratios, spans BS + IS)
WORKING_CAPITAL_FEATURES = [
    "dso_days",
    "inventory_days",
    "dpo_days",
    "dso_days_yoy_diff",
    "inventory_days_yoy_diff",
    "dpo_days_yoy_diff",
    "ratio_nwc_sales",
]

# Working capital temporal features (trends selected via nested CV)
WORKING_CAPITAL_TEMPORAL_FEATURES = [
    "dso_days_trend_3y",
    "inventory_days_trend_3y",
    "dpo_days_trend_3y",
]

# Operational ratios (subset related to operating performance)
OPERATIONAL_FEATURES = [
    "ny_omspanst",    # Revenue per employee
    "ny_foradlvpanst",  # Value added per employee
    "ny_omsf",        # Asset turnover
    "ny_anstf",       # Asset per employee
    "bslov_antanst",  # Number of employees
]

# Standard feature groupings by source for correlation analysis
FEATURE_GROUPS_BY_SOURCE = {
    "BALANCE_SHEET": BALANCE_SHEET_FEATURES,
    "INCOME_STATEMENT": INCOME_STATEMENT_FEATURES,
    "DERIVED_RATIOS": DERIVED_RATIO_FEATURES,
    "WORKING_CAPITAL": WORKING_CAPITAL_FEATURES,
    "TRENDS": TREND_FEATURE_NAMES,
    "TEMPORAL": TEMPORAL_FEATURE_NAMES,
    "CRISIS_HISTORY": CRISIS_FEATURE_NAMES,
    "MACRO": MACRO_FEATURE_NAMES,
    "OPERATIONAL": OPERATIONAL_FEATURES,
}

# -----------------------------------------------------------------------------
# Modeling configuration
# -----------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "bransch_sni071_konv",
    "bransch_borsbransch_konv",
    "ser_laen",
    "knc_kncfall",
    "ser_aktiv",
    "ser_nystartat",
    "bol_konkurs",
    "sme_category",
]

SME_CATEGORIES = ["Micro", "Small", "Medium", "Large"]

BASE_MODEL_FEATURES = [
    "bslov_antanst",
    # "ser_aktiv",  # REMOVED: Zero importance (all companies are active in filtered dataset)
    # "ser_nystartat",  # REMOVED: Zero variance (SHAP=0.0004)
    "company_age",
    # "ser_stklf", # REMOVED: Duplicative with bslov_antanst
    "bransch_sni071_konv",
    "bransch_borsbransch_konv",
    # "ser_laen",  # REMOVED: Geographic control with low predictive value
    # "knc_kncfall",  # REMOVED: Used as filter (knc_kncfall==1) rather than feature; model applies to independent companies only
]

FEATURES_FOR_MODEL = [
    f for f in list(dict.fromkeys(BASE_MODEL_FEATURES + NY_COLS + KEPT_RAW_COLS + ENGINEERED_FEATURE_NAMES))
    if f not in EXCLUDED_RAW_COLS
]