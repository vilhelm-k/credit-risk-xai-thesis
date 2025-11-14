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
    # Selected via comprehensive feature selection pipeline (Strategy 4: Hybrid)
    "ny_kapomsh",
    "ny_rs",
    "ny_skuldgrd",
    "ny_solid",
    "ny_avkegkap",
    "ny_kasslikv",
    "ny_nettomarg",
    "ny_omspanst",
    "ny_foradlvpanst",
    "ny_omsf",
    "ny_anstf",
]

KEPT_RAW_COLS = [
    "rr01_ntoms",      # For feature engineering only (excluded from model)
    "br09_tillgsu",    # For feature engineering only (excluded from model)
    "br10_eksu",       # For feature engineering only (excluded from model)
    "bslov_antanst",   # For feature engineering only (excluded from model)
    "br07b_kabasu",    # For feature engineering only (excluded from model)
    "rr07_rorresul",   # For feature engineering only (excluded from model)
    "rr15_resar",      # For feature engineering only (excluded from model)
]

# Raw columns to exclude from modeling (kept for feature engineering but not used as model features)
# All nominal values should be log-transformed; raw versions are excluded
EXCLUDED_RAW_COLS = [
    "rr01_ntoms",      # Replaced by log_rr01_ntoms
    "br09_tillgsu",    # Replaced by log_br09_tillgsu
    "br10_eksu",       # Replaced by log_br10_eksu
    "bslov_antanst",   # Replaced by log_bslov_antanst
    "br07b_kabasu",    # Replaced by log_br07b_kabasu
    "rr07_rorresul",   # Replaced by log_rr07_rorresul
    "rr15_resar",      # Replaced by log_rr15_resar
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

# Log-transformed nominal values (reduce skewness, align with literature)
# Selected via comprehensive feature selection pipeline (Strategy 4: Hybrid)
LOG_NOMINAL_FEATURES = [
    "log_br10_eksu",        # Total equity
    "log_br07b_kabasu",     # Cash and bank
    "log_bslov_antanst",    # Number of employees
    "log_rr15_resar",       # Net profit (NaN for negatives)
]

# Cost structure & profitability ratios
# Selected via comprehensive feature selection pipeline (Strategy 4: Hybrid)
RATIO_FEATURE_NAMES = [
    "ratio_depreciation_cost",
    "ratio_cash_interest_cov",
    "ratio_cash_liquidity",
    "ratio_short_term_debt_share",
    "ratio_retained_earnings_equity",
    "dividend_yield",
]

# Working capital efficiency & liquidity
# Selected via comprehensive feature selection pipeline (Strategy 4: Hybrid)
LIQUIDITY_EFFICIENCY_FEATURES = [
    "dso_days",         # Days sales outstanding
    "dpo_days",         # Days payables outstanding
]

# Year-over-year changes and trends
# Selected via comprehensive feature selection pipeline (Strategy 4: Hybrid)
TREND_FEATURE_NAMES = [
    "rr01_ntoms_yoy_abs",
    "rr07_rorresul_yoy_pct",
    "ny_solid_yoy_diff",
    "ratio_cash_liquidity_yoy_pct",
    "ratio_cash_liquidity_yoy_abs",
    "dso_days_yoy_diff",
    "inventory_days_yoy_diff",
    "dpo_days_yoy_diff",
    "current_ratio_yoy_pct",
]

# Multi-year temporal features
# Selected via comprehensive feature selection pipeline (Strategy 4: Hybrid)
TEMPORAL_FEATURE_NAMES = [
    # Growth metrics (CAGR) - fundamental business momentum
    "revenue_cagr_3y",
    "profit_cagr_3y",
    # Risk metrics (drawdown) - downside exposure
    "revenue_drawdown_5y",
    # Working capital trends - early warning signals
    "dpo_days_trend_3y",
]

# Note: OCF, Altman, and Leverage features removed via feature selection
# These were not selected in the final 40-feature set (Strategy 4: Hybrid)

# Credit event history
CRISIS_FEATURE_NAMES = [
    "event_count_last_5y",   # Credit events in past 5 years
]

# Macroeconomic conditions
# Selected via comprehensive feature selection pipeline (Strategy 4: Hybrid)
MACRO_FEATURE_NAMES = [
    "term_spread",           # Long-short rate spread
]

ENGINEERED_FEATURE_NAMES = (
    LOG_NOMINAL_FEATURES
    + RATIO_FEATURE_NAMES
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

# Categorical columns for LightGBM native categorical support
# Only includes true categoricals that benefit from categorical encoding
# Binary/ordinal columns kept as numeric for filtering (knc_kncfall, ser_aktiv, etc.)
CATEGORICAL_COLS = [
    "bransch_sni071_konv",       # Industry code (SNI): ~800 categories
    "bransch_borsbransch_konv",  # Stock exchange industry: 13 categories
    "ser_laen",                  # County code: ~20 categories
    "sme_category",              # SME size category: 4 categories (Micro/Small/Medium/Large)
]

SME_CATEGORIES = ["Micro", "Small", "Medium", "Large"]

BASE_MODEL_FEATURES = [
    # Selected via comprehensive feature selection pipeline (Strategy 4: Hybrid)
    "company_age",
    "bransch_sni071_konv",
]

FEATURES_FOR_MODEL = [
    f for f in list(dict.fromkeys(BASE_MODEL_FEATURES + NY_COLS + KEPT_RAW_COLS + ENGINEERED_FEATURE_NAMES))
    if f not in EXCLUDED_RAW_COLS
]