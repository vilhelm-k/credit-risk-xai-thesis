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
    # Selected via comprehensive feature selection pipeline (29 features from 54)
    # Optimized using VIF, Stability Selection, Boruta, SHAP, and RFECV (ROC-AUC)
    # Manual additions: ny_omsf (YoY revenue growth - short-term momentum)
    "ny_kapomsh",
    "ny_rs",
    "ny_skuldgrd",
    "ny_solid",
    "ny_avkegkap",
    "ny_kasslikv",
    "ny_nettomarg",
    "ny_omspanst",
    "ny_foradlvpanst",
    "ny_omsf",  # YoY change in net sales (short-term revenue momentum)
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
# Engineered feature registries (ORIGINAL feature set)
# -----------------------------------------------------------------------------

# Log-transformed nominal values (reduce skewness, align with literature)
# Selected via comprehensive feature selection pipeline (29 features from 54)
# Manual addition: log_br10_eksu (captures absolute equity size/buffer, complements equity ratio)
LOG_NOMINAL_FEATURES = [
    "log_br07b_kabasu",     # Cash and bank
    "log_br10_eksu",        # Total equity (size/buffer indicator)
]

# Cost structure & profitability ratios
# Selected via comprehensive feature selection pipeline (28 features from 54)
RATIO_FEATURE_NAMES = [
    "ratio_depreciation_cost",
    "ratio_cash_interest_cov",
    "ratio_cash_liquidity",
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
# Selected via comprehensive feature selection pipeline (29 features from 54)
# Removed ratio_cash_liquidity_yoy_pct (kept absolute - better captures threshold crossings)
TREND_FEATURE_NAMES = [
    "ny_solid_yoy_diff",
    "ratio_cash_liquidity_yoy_abs",  # Absolute change in quick ratio (captures magnitude)
    "inventory_days_yoy_diff",
]

# Multi-year temporal features
# Selected via comprehensive feature selection pipeline (29 features from 54)
TEMPORAL_FEATURE_NAMES = [
    # Growth metrics (CAGR) - fundamental business momentum
    "revenue_cagr_3y",
    "profit_cagr_3y",
    # Risk metrics (drawdown) - downside exposure
    "revenue_drawdown_5y",
]

# Note: OCF, Leverage, and additional macro features removed via feature selection
# Final selection: 29 features (28 from automated pipeline + 2 manual additions - 1 removal)
# Manual additions: ny_omsf (YoY revenue), log_br10_eksu (equity size)
# Manual removal: ratio_cash_liquidity_yoy_pct (kept absolute version)

# Macroeconomic conditions
# Selected via comprehensive feature selection pipeline (29 features from 54)
MACRO_FEATURE_NAMES = [
    "term_spread",           # Long-short rate spread
]

# All engineered features for modeling (29 features total after selection)
ENGINEERED_FEATURE_NAMES = (
    LOG_NOMINAL_FEATURES
    + RATIO_FEATURE_NAMES
    + LIQUIDITY_EFFICIENCY_FEATURES
    + TREND_FEATURE_NAMES
    + TEMPORAL_FEATURE_NAMES
    + MACRO_FEATURE_NAMES
)

# -----------------------------------------------------------------------------
# MIGRATED feature set (Altman/Ohlson aligned)
# -----------------------------------------------------------------------------
# Based on literature alignment with Altman Z-score and Ohlson O-score models
# See migration table for detailed rationale

# Log-transformed size measure (Ohlson W - firm size)
LOG_NOMINAL_FEATURES_V2 = [
    "log_total_assets",     # Ohlson's Size (W) - replaces log_br07b_kabasu
]

# Core financial ratios aligned with Altman/Ohlson
RATIO_FEATURE_NAMES_V2 = [
    "working_capital_ta",           # Altman X1: (CA - CL) / TA
    "retained_earnings_ta",         # Altman X2: RE / TA (migrated from RE/Equity)
    "interest_coverage",            # EBIT / Interest Expense (migrated from cash coverage)
    "ratio_cash_liquidity",         # Cash / Current Liabilities (kept - pure liquidity)
    "gross_margin",                 # (Sales - COGS) / Sales (new)
    "dividend_yield",               # Binary dividend payer (kept - top predictor)
]

# Working capital efficiency (kept)
LIQUIDITY_EFFICIENCY_FEATURES_V2 = [
    "dso_days",         # Days sales outstanding
    "dpo_days",         # Days payables outstanding
]

# Temporal/volatility features
TEMPORAL_FEATURE_NAMES_V2 = [
    "revenue_cagr_3y",              # Revenue CAGR (3Y) - kept
    "revenue_drawdown_5y",          # Revenue Drawdown (5Y) - kept (better than StdDev)
    "ebitda_volatility",            # StdDev(EBITDA_3y) / TA (new - operational risk)
]

# Macroeconomic conditions (kept)
MACRO_FEATURE_NAMES_V2 = [
    "term_spread",           # Long-short rate spread
]

# All migrated engineered features
ENGINEERED_FEATURE_NAMES_V2 = (
    LOG_NOMINAL_FEATURES_V2
    + RATIO_FEATURE_NAMES_V2
    + LIQUIDITY_EFFICIENCY_FEATURES_V2
    + TEMPORAL_FEATURE_NAMES_V2
    + MACRO_FEATURE_NAMES_V2
)

# Nyckeltal ratios for V2 (kept features only, renamed where needed)
NY_COLS_V2 = [
    "ny_kapomsh",       # Asset Turnover (Altman X5) - renamed from "Capital Turnover"
    "ny_skuldgrd",      # Debt Ratio (Ohlson TL/TA)
    "ny_solid",         # Equity Ratio
    "ny_avkegkap",      # Return on Equity
    "ny_kasslikv",      # Quick Ratio
    "ny_nettomarg",     # Net Profit Margin
    "ny_omsf",          # Revenue Growth (YoY)
]

# Dropped from V2 (with rationale):
# - ny_rs: Interest Rate on Debt - noisy, replaced by Interest Coverage
# - ny_foradlvpanst: Value Added per Employee - redundant with margins
# - ny_omspanst: Revenue per Employee - proxy for industry (already have SNI)
# - log_br10_eksu: Log Total Equity - undefined for negative equity, redundant
# - profit_cagr_3y: Invalid when profits cross zero
# - YoY delta ratios: Levels matter more than changes per analysis

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
    "sni_group_3digit",          # Industry code (SNI 3-digit Group): ~267 categories (better generalization than 5-digit)
    "bransch_borsbransch_konv",  # Stock exchange industry: 13 categories
    "ser_laen",                  # County code: ~20 categories
    "sme_category",              # SME size category: 4 categories (Micro/Small/Medium/Large)
]

SME_CATEGORIES = ["Micro", "Small", "Medium", "Large"]

BASE_MODEL_FEATURES = [
    # Selected via comprehensive feature selection pipeline (29 features from 54)
    "company_age",
    "sni_group_3digit",  # Aggregated 3-digit SNI for better generalization
    "ser_laen",          # County code (geographic effects)
]

# V2 base features (same as V1)
BASE_MODEL_FEATURES_V2 = [
    "company_age",
    "sni_group_3digit",
]

FEATURES_FOR_MODEL = [
    f for f in list(dict.fromkeys(BASE_MODEL_FEATURES + NY_COLS + KEPT_RAW_COLS + ENGINEERED_FEATURE_NAMES))
    if f not in EXCLUDED_RAW_COLS
]

# V2 feature list (Altman/Ohlson aligned)
FEATURES_FOR_MODEL_V2 = list(dict.fromkeys(
    BASE_MODEL_FEATURES_V2 + NY_COLS_V2 + ENGINEERED_FEATURE_NAMES_V2
))

# -----------------------------------------------------------------------------
# Feature Name Maps (human-readable names for tables/figures)
# -----------------------------------------------------------------------------

# Original feature set name mapping
FEATURE_NAME_MAP = {
    # Base Features (3)
    'company_age': 'Company Age',
    'sni_group_3digit': 'Industry (SNI)',
    'ser_laen': 'County',
    # Nyckeltal Ratios (10)
    'ny_foradlvpanst': 'Value Added per Employee',
    'ny_kapomsh': 'Capital Turnover',
    'ny_rs': 'Interest Rate on Debt',
    'ny_skuldgrd': 'Debt Ratio',
    'ny_solid': 'Equity Ratio',
    'ny_avkegkap': 'Return on Equity',
    'ny_kasslikv': 'Quick Ratio',
    'ny_nettomarg': 'Net Profit Margin',
    'ny_omspanst': 'Revenue per Employee',
    'ny_omsf': 'Revenue Growth (YoY)',
    # Log-Transformed Nominal Features (2)
    'log_br07b_kabasu': 'Log Cash & Bank',
    'log_br10_eksu': 'Log Total Equity',
    # Engineered Ratio Features (5)
    'ratio_depreciation_cost': 'Depreciation Intensity',
    'ratio_cash_interest_cov': 'Cash Interest Coverage',
    'ratio_cash_liquidity': 'Cash Ratio',
    'ratio_retained_earnings_equity': 'Retained Earnings / Equity',
    'dividend_yield': 'Dividend Payer',
    # Working Capital Efficiency (2)
    'dso_days': 'Days Sales Outstanding',
    'dpo_days': 'Days Payables Outstanding',
    # Year-over-Year Trends (3)
    'ny_solid_yoy_diff': 'Equity Ratio Δ (YoY)',
    'ratio_cash_liquidity_yoy_abs': 'Cash Ratio Δ (YoY)',
    'inventory_days_yoy_diff': 'Inventory Days Δ (YoY)',
    # Multi-Year Temporal Features (3)
    'revenue_cagr_3y': 'Revenue CAGR (3Y)',
    'profit_cagr_3y': 'Profit CAGR (3Y)',
    'revenue_drawdown_5y': 'Revenue Drawdown (5Y)',
    # Macroeconomic Conditions (1)
    'term_spread': 'Term Spread',
}

# V2 (Altman/Ohlson aligned) feature set name mapping
FEATURE_NAME_MAP_V2 = {
    # Base Features (3)
    'company_age': 'Company Age',
    'sni_group_3digit': 'Industry (SNI)',
    # Nyckeltal Ratios - Kept (7)
    'ny_kapomsh': 'Total Asset Turnover',  # Renamed: Altman X5
    'ny_skuldgrd': 'Debt Ratio',           # Ohlson TL/TA
    'ny_solid': 'Equity Ratio',
    'ny_avkegkap': 'Return on Equity',
    'ny_kasslikv': 'Quick Ratio',
    'ny_nettomarg': 'Net Profit Margin',
    'ny_omsf': 'Revenue Growth (YoY)',
    # Log-Transformed Size (1)
    'log_total_assets': 'Log Total Assets',  # Ohlson Size (W)
    # Altman/Ohlson Aligned Ratios (5)
    'working_capital_ta': 'Working Capital / TA',      # Altman X1
    'retained_earnings_ta': 'Retained Earnings / TA',  # Altman X2
    'interest_coverage': 'Interest Coverage',          # EBIT / Interest
    'ratio_cash_liquidity': 'Cash Ratio',
    'gross_margin': 'Gross Margin',
    'dividend_yield': 'Dividend Payer',
    # Working Capital Efficiency (2)
    'dso_days': 'Days Sales Outstanding',
    'dpo_days': 'Days Payables Outstanding',
    # Temporal Features (3)
    'revenue_cagr_3y': 'Revenue CAGR (3Y)',
    'revenue_drawdown_5y': 'Revenue Drawdown (5Y)',
    'ebitda_volatility': 'EBITDA Volatility (3Y)',
    # Macroeconomic Conditions (1)
    'term_spread': 'Term Spread',
}

# -----------------------------------------------------------------------------
# Model Version Selection (plug-and-play configuration)
# -----------------------------------------------------------------------------
# Set ACTIVE_MODEL_VERSION to switch between feature sets across all notebooks
# Valid values: "v1" (original), "v2" (Altman/Ohlson aligned)

ACTIVE_MODEL_VERSION = "v2"  # Change this to "v2" to use migrated features

def get_active_features():
    """Get the feature list for the active model version."""
    if ACTIVE_MODEL_VERSION == "v2":
        return FEATURES_FOR_MODEL_V2
    return FEATURES_FOR_MODEL

def get_active_feature_name_map():
    """Get the feature name map for the active model version."""
    if ACTIVE_MODEL_VERSION == "v2":
        return FEATURE_NAME_MAP_V2
    return FEATURE_NAME_MAP

def get_display_name(feature: str) -> str:
    """Get human-readable display name for a feature."""
    name_map = get_active_feature_name_map()
    return name_map.get(feature, feature)

def get_ale_filename(feature: str) -> str:
    """Auto-generate ALE plot filename from feature name."""
    display_name = get_display_name(feature)
    # Convert to lowercase, replace spaces/special chars with underscores
    clean_name = display_name.lower()
    clean_name = clean_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    clean_name = clean_name.replace('δ', 'delta').replace('%', 'pct')
    # Remove consecutive underscores
    while '__' in clean_name:
        clean_name = clean_name.replace('__', '_')
    return f"ale_{clean_name}.pdf"

# Convenience exports for active model
ACTIVE_FEATURES = get_active_features()
ACTIVE_FEATURE_NAME_MAP = get_active_feature_name_map()

# -----------------------------------------------------------------------------
# Feature Bounds for ALE Plots
# -----------------------------------------------------------------------------
# Domain-appropriate bounds for ALE visualization
# Based on descriptive statistics (P1-P99) and domain knowledge
# None = binary feature (handled separately in ALE plotting)

FEATURE_BOUNDS = {
    # =========================================================================
    # BASE FEATURES
    # =========================================================================
    'company_age': (0, 50),         # 0-50 years covers vast majority; 95th percentile is 44

    # =========================================================================
    # NYCKELTAL RATIOS (used by both V1 and V2)
    # =========================================================================
    'ny_kapomsh': (0.5, 8),         # Capital/Asset turnover: 0.5-8x is typical operating range
    'ny_rs': (0, 0.1),              # Interest rate on debt: 0-10% is typical
    'ny_skuldgrd': (0, 20),         # Debt ratio: 0-20 covers P95; higher values are extreme leverage
    'ny_solid': (0, 0.8),           # Equity ratio: 0-80% (P95=0.74)
    'ny_avkegkap': (-1, 1.5),       # ROE: -100% to +150% is economically meaningful
    'ny_kasslikv': (0.2, 4),        # Quick ratio: 0.2-4 covers normal operating range
    'ny_nettomarg': (-0.5, 0.35),   # Net margin: -50% to +35% covers P1-P99
    'ny_omspanst': (100, 6000),     # Revenue per employee (kSEK): 100-6000 typical range
    'ny_foradlvpanst': (100, 1500), # Value added per employee: 100-1500 kSEK
    'ny_omsf': (-0.3, 2),           # Revenue growth YoY: -30% to +200%

    # =========================================================================
    # V1 LOG-TRANSFORMED NOMINAL FEATURES
    # =========================================================================
    'log_br07b_kabasu': (0, 10),    # Log cash: 0-10 corresponds to 0 to ~22M kSEK
    'log_br10_eksu': (4, 10),       # Log equity: 4-10 corresponds to ~55 to ~22M kSEK

    # =========================================================================
    # V2 LOG-TRANSFORMED NOMINAL FEATURES
    # =========================================================================
    'log_total_assets': (6, 12),    # Log total assets: P1=6.5, P99=12.4; ~700 kSEK to ~160M kSEK

    # =========================================================================
    # V1 ENGINEERED RATIO FEATURES
    # =========================================================================
    'ratio_depreciation_cost': (0, 0.15),       # Depreciation as % of revenue: 0-15%
    'ratio_cash_interest_cov': (-2000, 0),      # Cash/interest costs (negative = paying more)
    'ratio_cash_liquidity': (0, 3),             # Cash ratio: 0-3
    'ratio_retained_earnings_equity': (-1, 1.5),# Retained earnings/equity: -100% to +150%
    'dividend_yield': None,                     # BINARY: handled separately

    # =========================================================================
    # V2 ALTMAN/OHLSON ALIGNED RATIOS
    # =========================================================================
    'working_capital_ta': (-0.6, 0.8),   # Altman X1: P1=-0.61, P99=0.83; typical -60% to +80%
    'retained_earnings_ta': (-0.5, 0.8), # Altman X2: P1=-0.44, P99=0.79; typical -50% to +80%
    'interest_coverage': (-5, 20),       # Domain-clipped: [-5, 20] for logit stability
    'gross_margin': (0, 1),              # (Sales-COGS)/Sales: P5=0.10, P99=0.96; 0-100%
    'ebitda_volatility': (0, 0.6),       # StdDev(EBITDA)/TA: P1=0.003, P99=0.59; 0-60%

    # =========================================================================
    # WORKING CAPITAL EFFICIENCY (both versions)
    # =========================================================================
    'dso_days': (5, 150),           # Days sales outstanding: 5-150 days
    'dpo_days': (5, 120),           # Days payables outstanding: 5-120 days

    # =========================================================================
    # V1 YEAR-OVER-YEAR TRENDS
    # =========================================================================
    'ny_solid_yoy_diff': (-0.2, 0.2),           # YoY equity ratio change: ±20pp
    'ratio_cash_liquidity_yoy_abs': (-1, 1),    # YoY cash ratio change (absolute)
    'inventory_days_yoy_diff': (-50, 50),       # YoY inventory days change

    # =========================================================================
    # MULTI-YEAR TEMPORAL FEATURES (both versions)
    # =========================================================================
    'revenue_cagr_3y': (-0.2, 1),    # 3-year revenue CAGR: -20% to +100%
    'profit_cagr_3y': (-0.8, 3),     # 3-year profit CAGR: -80% to +300% (V1 only)
    'revenue_drawdown_5y': (-0.6, 0),# Revenue drawdown: -60% to 0%

    # =========================================================================
    # MACROECONOMIC (both versions)
    # =========================================================================
    'term_spread': (-0.7, 1.7),     # Yield curve spread (uses full data range)
}