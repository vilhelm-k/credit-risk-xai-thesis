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
    "ny_rorkapo",
    "ny_kasslikv",
    "ny_rormarg",
    "ny_nettomarg",
    "ny_vinstprc",
    "ny_omspanst",
    "ny_foradlvpanst",
    "ny_omsf",
    "ny_anstf",
]

KEPT_RAW_COLS = [
    "rr01_ntoms",
    "br09_tillgsu",
    "br10_eksu",
    "bslov_antanst",
    "br07b_kabasu",
    "br13_ksksu",
    "br15_lsksu",
    "rr07_rorresul",
    "rr15_resar",
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
    "ratio_personnel_cost",
    "ratio_depreciation_cost",
    "ratio_other_operating_cost",
    "ratio_financial_cost",
    "ratio_ebitda_margin",
    "ratio_ebit_interest_cov",
    "ratio_ebitda_interest_cov",
    "ratio_cash_interest_cov",
    "ratio_cash_liquidity",
    "ratio_nwc_sales",
    "ratio_short_term_debt_share",
    "ratio_secured_debt_assets",
    "ratio_retained_earnings_equity",
    "ratio_share_capital_equity",
    "ratio_dividend_payout",
    "ratio_group_support",
]

LIQUIDITY_EFFICIENCY_FEATURES = [
    "dso_days",
    "inventory_days",
    "dpo_days",
]

TREND_FEATURE_NAMES = [
    "rr01_ntoms_yoy_pct",
    "rr01_ntoms_yoy_abs",
    "rr07_rorresul_yoy_pct",
    "rr07_rorresul_yoy_abs",
    "br09_tillgsu_yoy_pct",
    "br09_tillgsu_yoy_abs",
    "ny_solid_yoy_diff",
    "ny_skuldgrd_yoy_diff",
    "ratio_cash_liquidity_yoy_pct",
    "ratio_cash_liquidity_yoy_abs",
    "ratio_ebit_interest_cov_yoy_pct",
    "revenue_cagr_3y",
    "assets_cagr_3y",
    "equity_cagr_3y",
    "profit_cagr_3y",
    "revenue_cagr_5y",
    "assets_cagr_5y",
    "equity_cagr_5y",
    "profit_cagr_5y",
    "ny_rormarg_trend_3y",
    "ny_skuldgrd_trend_3y",
    "ratio_cash_liquidity_trend_3y",
    "dso_days_trend_3y",
    "inventory_days_trend_3y",
    "dpo_days_trend_3y",
    "ny_rormarg_trend_5y",
    "ny_skuldgrd_trend_5y",
    "ratio_cash_liquidity_trend_5y",
    "dso_days_yoy_diff",
    "inventory_days_yoy_diff",
    "dpo_days_yoy_diff",
    "ny_rormarg_vol_3y",
    "ny_skuldgrd_vol_3y",
    "ratio_cash_liquidity_vol_3y",
    "ny_rormarg_vol_5y",
    "ny_skuldgrd_vol_5y",
    "ny_rormarg_avg_2y",
    "ratio_cash_liquidity_avg_2y",
    "ny_rormarg_avg_5y",
    "ratio_cash_liquidity_avg_5y",
    "revenue_drawdown_5y",
    "equity_drawdown_5y",
]

CRISIS_FEATURE_NAMES = [
    "years_since_last_credit_event",
    "last_event_within_1y",
    "last_event_within_2y",
    "last_event_within_3y",
    "last_event_within_5y",
    "event_count_total",
    "event_count_last_5y",
    "ever_failed",
]

MACRO_FEATURE_NAMES = [
    "gdp_growth",
    "gdp_growth_3y_avg",
    "interest_avg_short",
    "interest_avg_medium",
    "interest_avg_long",
    "interest_delta_short",
    "term_spread",
    "term_spread_delta",
    "inflation_yoy",
    "inflation_trailing_3y",
    "unemp_rate",
    "unemp_delta",
    "real_revenue_growth",
    "revenue_vs_gdp",
    "profit_vs_gdp",
    "correlation_revenue_gdp_5y",
]

MACRO_FEATURE_PRIORITIES = {
    "gdp_growth": "high",
    "gdp_growth_3y_avg": "high",
    "interest_avg_short": "high",
    "interest_avg_medium": "medium",
    "interest_avg_long": "high",
    "interest_delta_short": "high",
    "term_spread": "high",
    "term_spread_delta": "medium",
    "inflation_yoy": "high",
    "inflation_trailing_3y": "medium",
    "unemp_rate": "medium",
    "unemp_delta": "medium",
    "real_revenue_growth": "high",
    "revenue_vs_gdp": "high",
    "profit_vs_gdp": "medium",
    "correlation_revenue_gdp_5y": "medium",
}

ENGINEERED_FEATURE_NAMES = (
    RATIO_FEATURE_NAMES
    + LIQUIDITY_EFFICIENCY_FEATURES
    + TREND_FEATURE_NAMES
    + CRISIS_FEATURE_NAMES
    + MACRO_FEATURE_NAMES
)

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
    "ser_aktiv",
    "ser_nystartat",
    "company_age",
    "ser_stklf",
    "bransch_sni071_konv",
    "bransch_borsbransch_konv",
    "ser_laen",
    "knc_kncfall",
]

FEATURES_FOR_MODEL = list(
    dict.fromkeys(BASE_MODEL_FEATURES + NY_COLS + KEPT_RAW_COLS + ENGINEERED_FEATURE_NAMES)
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MIN_REVENUE_KSEK = 1_000
CORRELATION_WINDOW = 5
