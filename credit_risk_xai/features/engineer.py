from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from numba import njit

from credit_risk_xai.config import (
    BASE_CACHE_PATH,
    BR_SOURCE_COLS,
    FEATURE_CACHE_PATH,
    FEATURES_FOR_MODEL,
    KEPT_RAW_COLS,
    MACRO_CACHE_PATH,
    NY_COLS,
    RR_SOURCE_COLS,
)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Safe division handling zero/missing denominators.

    Returns NaN for division by zero or missing values.
    Keeps extreme values intact for tree-based models.
    """
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _safe_pct_change(series: pd.Series) -> pd.Series:
    """Compute percentage change with proper handling of zero-to-non-zero transitions.

    Zero-to-non-zero transitions are set to NaN because they produce infinite
    or meaningless percentage changes. This commonly occurs when a company goes
    from zero debt/cash to having debt/cash, or vice versa.
    """
    shifted = series.groupby(level=0).shift(1)
    # Mask zero-to-non-zero transitions: previous value is 0 but current is not
    zero_to_nonzero = (shifted == 0) & (series != 0)
    # Standard pct_change calculation
    result = series.groupby(level=0).pct_change(fill_method=None)
    # Set zero-to-non-zero transitions to NaN
    return result.where(~zero_to_nonzero, np.nan)


def _safe_log(series: pd.Series) -> pd.Series:
    """Log transform with robust handling of zeros and negatives.

    Uses np.log1p(x) which computes log(1+x) efficiently and handles zeros.
    Returns NaN for negative values (not economically meaningful for balance sheet items).
    """
    return series.where(series >= 0).apply(np.log1p)


@njit(cache=True)
def _rolling_slope_kernel(values: np.ndarray) -> float:
    """Compute slope of the linear trend for a dense 1D window with possible NaNs."""
    count = 0
    x_sum = 0.0
    x2_sum = 0.0
    y_sum = 0.0
    xy_sum = 0.0

    for idx in range(values.shape[0]):
        y_val = values[idx]
        if np.isnan(y_val):
            continue

        x_val = float(idx)
        count += 1
        x_sum += x_val
        x2_sum += x_val * x_val
        y_sum += y_val
        xy_sum += x_val * y_val

    if count < 2:
        return np.nan

    denominator = x2_sum - (x_sum * x_sum) / count
    if denominator == 0.0:
        return np.nan

    numerator = xy_sum - (x_sum * y_sum) / count
    return numerator / denominator


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling slope using a compiled numba kernel for memory-efficient performance."""
    result = (
        series.groupby(level=0)
        .rolling(window=window, min_periods=window)
        .apply(_rolling_slope_kernel, raw=True, engine="numba")
    )
    return result.reset_index(level=0, drop=True)


def _rolling_avg(series: pd.Series, window: int) -> pd.Series:
    result = (
        series.groupby(level=0)
        .rolling(window=window, min_periods=2)
        .mean()
    )
    return result.reset_index(level=0, drop=True)


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    result = (
        series.groupby(level=0)
        .rolling(window=window, min_periods=window)
        .std()
    )
    return result.reset_index(level=0, drop=True)


@njit(cache=True)
def _rolling_drawdown_kernel(values: np.ndarray) -> float:
    """Return minimum drawdown (value / peak - 1) within a window, ignoring NaNs."""
    has_value = False
    has_drawdown = False
    run_max = 0.0
    min_drawdown = 0.0

    for idx in range(values.shape[0]):
        val = values[idx]
        if np.isnan(val):
            continue

        if not has_value:
            run_max = val
            has_value = True
        else:
            if val > run_max:
                run_max = val

        if run_max == 0.0:
            drawdown = np.nan
        else:
            drawdown = val / run_max - 1.0

        if not np.isnan(drawdown):
            if not has_drawdown or drawdown < min_drawdown:
                min_drawdown = drawdown
                has_drawdown = True

    if not has_drawdown:
        return np.nan

    return min_drawdown


def _rolling_drawdown(series: pd.Series, window: int) -> pd.Series:
    """Optimized rolling drawdown using a compiled kernel to avoid Python-overhead."""
    result = (
        series.groupby(level=0)
        .rolling(window=window, min_periods=window)
        .apply(_rolling_drawdown_kernel, raw=True, engine="numba")
    )
    return result.reset_index(level=0, drop=True)


def _compute_cagr(series: pd.Series, periods: int) -> pd.Series:
    """
    Compute compound annual growth rate (CAGR) over N periods.

    Returns decimal values: 0.05 = 5% annual growth, 1.0 = 100% annual growth
    """
    shifted = series.groupby(level=0).shift(periods)
    ratio = _safe_div(series, shifted)
    mask = (series > 0) & (shifted > 0)
    cagr = ratio.pow(1 / periods) - 1
    return cagr.where(mask, np.nan)


def _rolling_beta(
    dependent: pd.Series,
    benchmark: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Rolling OLS beta of dependent versus benchmark for each entity."""
    if min_periods is None:
        min_periods = window

    valid_mask = (~dependent.isna()) & (~benchmark.isna())
    if not valid_mask.any():
        return pd.Series(np.nan, index=dependent.index)

    dependent_valid = dependent.where(valid_mask)
    benchmark_valid = benchmark.where(valid_mask)

    group = lambda s: s.groupby(level=0, group_keys=False)

    count = group(valid_mask).rolling(window=window, min_periods=1).sum()
    sum_dep = group(dependent_valid).rolling(window=window, min_periods=1).sum()
    sum_bench = group(benchmark_valid).rolling(window=window, min_periods=1).sum()
    sum_cross = group(dependent_valid * benchmark_valid).rolling(window=window, min_periods=1).sum()
    sum_bench_sq = group(benchmark_valid.pow(2)).rolling(window=window, min_periods=1).sum()

    numerator = sum_cross - (sum_dep * sum_bench) / count
    denominator = sum_bench_sq - (sum_bench * sum_bench) / count

    denominator = denominator.mask(denominator.abs() < 1e-12)
    beta = numerator / denominator
    beta = beta.where(count >= min_periods)
    beta = beta.replace([np.inf, -np.inf], np.nan)

    return beta.droplevel(0).reindex(dependent.index)


# -----------------------------------------------------------------------------
# Feature engineering core
# -----------------------------------------------------------------------------


def create_engineered_features(
    df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    drop_raw_sources: bool = True,
) -> pd.DataFrame:
    """Return a dataframe enriched with engineered features."""
    # Work directly on df - already sorted by ORGNR, ser_year in make_dataset.py
    df["ser_stklf"] = df["ser_stklf"].replace(9, pd.NA)

    # Create aggregated SNI code (3-digit = Group level) for better generalization
    # Keep original bransch_sni071_konv intact, create new column sni_group_3digit
    # Analysis shows: 5-digit has 815 cats (54% <100 obs), 3-digit has 267 cats (31% <100 obs)
    # Signal variance nearly identical (0.000517 vs 0.000671) but much better generalization
    logger.info("Creating aggregated SNI code (3-digit Group level)")
    df["sni_group_3digit"] = (
        df["bransch_sni071_konv"]
        .astype(str)
        .str[:3]  # Take first 3 digits (Group level)
        .astype("category")  # Convert to category for LightGBM
    )

    # Set index without dropping columns (we'll drop at the end if needed)
    df.set_index(["ORGNR", "ser_year"], drop=False, inplace=True)

    # Create groupby object once and reuse it throughout
    group = df.groupby(level=0, group_keys=False)

    # Collect ALL new features in a single dictionary to minimize joins
    new_features = {}

    logger.info("Creating log-transformed nominal features")
    # Log-transform absolute financial values (kSEK) - NOT ratios/percentages
    # Selected via comprehensive feature selection pipeline (Strategy 4: Hybrid)
    # Removed: log_rr01_ntoms, log_br09_tillgsu, log_rr07_rorresul
    nominal_cols_to_log = [
        "br10_eksu",       # Total equity
        "br07b_kabasu",    # Cash and bank
        "bslov_antanst",   # Number of employees
        "rr15_resar",      # Net profit (can be negative - will return NaN for negatives)
    ]
    for col in nominal_cols_to_log:
        if col in df.columns:
            new_features[f"log_{col}"] = _safe_log(df[col]).astype("float32")

    # V2 Feature: Log Total Assets (Ohlson's Size W)
    # Replaces log_br07b_kabasu for size measure - more stable than cash
    new_features["log_total_assets"] = _safe_log(df["br09_tillgsu"]).astype("float32")

    logger.info("Computing cost structure and profitability ratios")
    # Compute intermediate values once
    ebitda = df["rr07_rorresul"] + df["rr05_avskriv"]
    financial_cost_net = df["rr09_finkostn"] - df["rr09d_jfrstfin"]

    new_features.update({
        "ratio_depreciation_cost": _safe_div(-df["rr05_avskriv"], df["rr01_ntoms"]),
        "ratio_cash_interest_cov": _safe_div(df["br07b_kabasu"], financial_cost_net),
    })

    # V2 Feature: Gross Margin = (Sales - COGS) / Sales
    # Captures fundamental business model viability before overheads
    # rr06a_prodkos is negative (cost), so we add it
    new_features["gross_margin"] = _safe_div(
        df["rr01_ntoms"] + df["rr06a_prodkos"],  # Sales - COGS (prodkos is negative)
        df["rr01_ntoms"]
    )

    # V2 Feature: Interest Coverage = EBIT / Interest Expense
    # Standard coverage ratio - ability to pay from earnings, not savings
    # rr09_finkostn is financial costs (negative), so negate it
    # EBIT = Operating profit (rr07_rorresul)
    interest_expense = -df["rr09_finkostn"]  # Make positive for denominator
    new_features["interest_coverage"] = _safe_div(df["rr07_rorresul"], interest_expense)

    logger.info("Computing liquidity and working-capital efficiencies")
    # Compute current liabilities total (sum of subcategories)
    current_liabilities = (
        df["br13a_ksklev"].fillna(0) +
        df["br13b_kskknc"].fillna(0) +
        df["br13c_kskov"].fillna(0)
    )
    # Compute long-term liabilities total (sum of subcategories)
    longterm_liabilities = (
        df["br15a_lskknc"].fillna(0) +
        df["br15b_lskov"].fillna(0) +
        df["br15c_obllan"].fillna(0)
    )
    total_debt = current_liabilities + longterm_liabilities

    new_features.update({
        "ratio_cash_liquidity": _safe_div(df["br07b_kabasu"] + df["br07a_kplacsu"], current_liabilities),
        "dso_days": _safe_div(df["br06g_kfordsu"], df["rr01_ntoms"]) * 365,
        "inventory_days": _safe_div(df["br06c_lagersu"], -df["rr06a_prodkos"]) * 365,
        "dpo_days": _safe_div(df["br13a_ksklev"], -df["rr06a_prodkos"]) * 365,
    })

    # V2 Feature: Working Capital / Total Assets (Altman X1)
    # Most famous bankruptcy ratio - measures short-term liquidity relative to size
    working_capital = df["br08_omstgsu"] - current_liabilities
    new_features["working_capital_ta"] = _safe_div(working_capital, df["br09_tillgsu"])

    # V2 Feature: Retained Earnings / Total Assets (Altman X2)
    # Measures cumulative profitability - migrated from RE/Equity (more stable denominator)
    # br10e_balres = Retained earnings (accumulated)
    new_features["retained_earnings_ta"] = _safe_div(df["br10e_balres"], df["br09_tillgsu"])

    logger.info("Computing capital structure and payout ratios")
    new_features.update({
        "ratio_short_term_debt_share": _safe_div(current_liabilities, total_debt),
        "ratio_retained_earnings_equity": _safe_div(df["br10e_balres"], df["br10_eksu"]),
        # Binary: 1 if company pays dividend, 0 otherwise
        # rr00_utdbel is negative when dividends are paid (outflow)
        "dividend_yield": (df["rr00_utdbel"] > 0).astype("Int8"),
    })

    # Compute current_ratio now (needed for YoY calculations later)
    current_ratio_temp = _safe_div(df["br08_omstgsu"], current_liabilities)
    new_features["current_ratio_temp"] = current_ratio_temp

    # Compute OCF and leverage features (needed for YoY/trend calculations even if not in final model)
    working_capital = df["br08_omstgsu"] - current_liabilities
    net_debt = total_debt - df["br07_kplackaba"]
    ocf_proxy = ebitda - working_capital.groupby(level=0).diff()

    new_features.update({
        "ocf_proxy": ocf_proxy,
        "ratio_ocf_to_debt": _safe_div(ocf_proxy, total_debt),
        "net_debt_to_ebitda": _safe_div(net_debt, ebitda),
        # Store EBITDA for later use in volatility calculation (V2 feature)
        "_ebitda_temp": ebitda,
    })

    # Join basic ratios early so they can be used in trend calculations
    df = df.join(pd.DataFrame(new_features, index=df.index))
    new_features.clear()  # Clear to prepare for next batch

    # Drop ALL rr_* and br_* columns except KEPT_RAW_COLS to save memory
    # All features requiring BR/RR columns have been computed above (including OCF, Altman, leverage)
    # KEPT_RAW_COLS only contains nominal values needed for log transform + YoY/trend calculations
    cols_to_drop = [col for col in (RR_SOURCE_COLS + BR_SOURCE_COLS) if col in df.columns and col not in KEPT_RAW_COLS]
    df.drop(columns=cols_to_drop, inplace=True)
    gc.collect()  # Force garbage collection to free memory immediately
    logger.debug("Dropped {} raw rr/br columns after computing all ratio features (kept only KEPT_RAW_COLS)", len(cols_to_drop))

    # Recreate groupby after joining new features
    group = df.groupby(level=0, group_keys=False)

    logger.info("Computing year-over-year deltas and immediate trends")
    for col in ["rr07_rorresul"]:
        new_features[f"{col}_yoy_pct"] = _safe_pct_change(df[col])
    for col in ["rr01_ntoms"]:
        new_features[f"{col}_yoy_abs"] = group[col].diff()

    new_features.update({
        "ny_solid_yoy_diff": group["ny_solid"].diff(),
        "ratio_cash_liquidity_yoy_pct": _safe_pct_change(df["ratio_cash_liquidity"]),
        "ratio_cash_liquidity_yoy_abs": group["ratio_cash_liquidity"].diff(),
        "dso_days_yoy_diff": group["dso_days"].diff(),
        "current_ratio_yoy_pct": _safe_pct_change(df["current_ratio_temp"]),
    })

    # Compute CAGR features (3-year)
    for source, target, window in [
        ("rr01_ntoms", "revenue_cagr_3y", 3),
        ("rr15_resar", "profit_cagr_3y", 3),
    ]:
        new_features[target] = _compute_cagr(df[source], window)

    # Join trends and CAGR before computing rolling features that depend on them
    df = df.join(pd.DataFrame(new_features, index=df.index))
    new_features.clear()

    # Drop temporary current_ratio column (we only need the YoY change)
    df.drop(columns=["current_ratio_temp"], inplace=True, errors="ignore")

    logger.info("Computing selected temporal features (working capital trends & drawdowns)")
    # Selected via comprehensive feature selection pipeline (Strategy 4: Hybrid)
    # Removed: inventory_days_trend_3y, equity_drawdown_5y

    # Working capital trends (3y) - early warning signals for operational deterioration
    new_features.update({
        # Trend and YoY diffs don't need additional bounds - base features already bounded
        "dpo_days_trend_3y": _rolling_slope(df["dpo_days"], window=3),
        "inventory_days_yoy_diff": group["inventory_days"].diff(),
        "dpo_days_yoy_diff": group["dpo_days"].diff(),
        # Risk metrics (drawdown) - capture downside exposure
        "revenue_drawdown_5y": _rolling_drawdown(df["rr01_ntoms"], window=5),
    })

    # V2 Feature: EBITDA Volatility (3Y) = StdDev(EBITDA_3y) / Total Assets
    # Captures operational risk/stability (Režňáková & Karas)
    # Uses pre-computed EBITDA stored in _ebitda_temp column
    ebitda_std_3y = _rolling_std(df["_ebitda_temp"], window=3)
    new_features["ebitda_volatility"] = _safe_div(ebitda_std_3y, df["br09_tillgsu"])

    # EXCLUDED TEMPORAL FEATURES (based on nested CV analysis):
    # - Margin trends/volatility/averages: Static values + YoY changes are sufficient
    # - Leverage trends/volatility: Static ny_skuldgrd + YoY changes are sufficient
    # - Cash liquidity trends/volatility/averages: Static ratio + YoY changes are sufficient
    # - 5y CAGR variants: 3y windows provide optimal signal without overfitting
    # These exclusions improve model parsimony without sacrificing performance

    # Join all rolling features at once
    df = df.join(pd.DataFrame(new_features, index=df.index))
    new_features.clear()

    # Recreate groupby for YoY and trend calculations on OCF/leverage features
    group = df.groupby(level=0, group_keys=False)

    logger.info("Computing YoY changes and trends for OCF and leverage metrics")
    new_features.update({
        # Leverage YoY
        "net_debt_to_ebitda_yoy_diff": group["net_debt_to_ebitda"].diff(),
        # OCF YoY changes
        "ocf_proxy_yoy_pct": _safe_pct_change(df["ocf_proxy"]),
        "ratio_ocf_to_debt_yoy_diff": group["ratio_ocf_to_debt"].diff(),
        # OCF 3-year trend
        "ocf_proxy_trend_3y": _rolling_slope(df["ocf_proxy"], window=3),
    })

    # Join YoY and trend features
    df = df.join(pd.DataFrame(new_features, index=df.index))
    new_features.clear()

    # REMOVED: years_since_last_credit_event - potential data leakage
    # credit_events = df["credit_event"] == 1
    # event_years = df["ser_year"].where(credit_events)
    # last_event_year = event_years.groupby(level=0).ffill()
    # df["years_since_last_credit_event"] = df["ser_year"] - last_event_year
    # df.loc[last_event_year.isna(), "years_since_last_credit_event"] = np.nan

    # REMOVED: Redundant binary flags - years_since_last_credit_event is sufficient
    # horizons = {
    #     "last_event_within_1y": 1,
    #     "last_event_within_2y": 2,
    #     "last_event_within_3y": 3,
    #     "last_event_within_5y": 5,
    # }
    # for col, limit in horizons.items():
    #     df[col] = (
    #         df["years_since_last_credit_event"].le(limit).fillna(False).astype("Int8")
    #     )

    # Binary indicator: Any credit event in PAST 5 years (excludes current year to prevent data leakage)
    # Lookback: years t-5, t-4, t-3, t-2, t-1 (NOT year t)
    # This follows Basel III 5-year PD estimation window standard
    # Binary (0/1) retains 100% of predictive signal vs count (companies with 2+ events all default)
    df["any_event_last_5y"] = (
        group["credit_event"]
        .shift(1)  # Shift by 1 to exclude current year (prevent data leakage)
        .rolling(window=5, min_periods=1)
        .sum()
        .gt(0)  # Convert to boolean: True if any event in window
        .astype("Int8")  # 0 or 1
    )

    logger.info("Credit event history features computed")

    if macro_df is None:
        raise ValueError("Macro dataframe is required. Run the macro preprocessing step first.")

    logger.info("Merging macroeconomic indicators and derived comparisons")
    # Optimized: Use single merge instead of loop with map
    macro_aligned = macro_df.set_index("ser_year")

    ser_year_values = df["ser_year"].to_numpy()
    macro_matched = macro_aligned.reindex(ser_year_values)
    macro_matched.index = df.index

    for col in macro_aligned.columns:
        df[col] = macro_matched[col].values

    # df["real_revenue_growth"] = (
    #     df["rr01_ntoms_yoy_pct"] - df["inflation_yoy"]
    # ) # REMOVED: Redundant with ny_omsf
    # df["revenue_vs_gdp"] = (  # REMOVED: Nearly identical to real_revenue_growth (r=0.999996)
    #     df["rr01_ntoms_yoy_pct"] - df["gdp_growth"]
    # )
    # df["profit_vs_gdp"] = (
    #     df["rr07_rorresul_yoy_pct"] - df["gdp_growth"]
    # ) # REMOVED: Redundant

    # Compute revenue beta (cyclicality): beta = cov(revenue, gdp) / var(gdp)
    # Beta interpretation: For every 1% GDP growth, revenue grows by beta%
    # Beta > 1: Cyclical (high risk), Beta < 1: Defensive (low risk)
    # Note: This is the OLS slope coefficient from regressing revenue growth on GDP growth

    logger.info("Calculating rolling revenue beta (streaming sums)")
    # Use ny_omsf (revenue growth) instead of rr01_ntoms_yoy_pct (r=1.0, identical)
    df["revenue_beta_gdp_5y"] = _rolling_beta(
        df["ny_omsf"],
        df["gdp_growth"],
        window=5,
        min_periods=4,
    )

    logger.info("Macro indicators merged (including revenue beta)")

    # -------------------------------------------------------------------------
    # Remove duplicate accounting year observations (data leakage fix)
    # -------------------------------------------------------------------------
    # Detection rule: ny_omsf == 0 AND dso_days_yoy_diff == 0
    #
    # Background: Some companies with non-calendar fiscal years have duplicate
    # accounting snapshots in consecutive Serrano years. This causes all YoY
    # features to be exactly 0, which the model learns as a spurious "default
    # signal" (Cluster 0 in SHAP analysis had 94% default rate, 44x enrichment).
    #
    # Fix: Transfer credit_event to the previous (valid) year, then remove
    # the duplicate row. This preserves the default signal while eliminating
    # the leaky "all YoY = 0" pattern.
    # -------------------------------------------------------------------------
    logger.info("Detecting and removing duplicate accounting year observations (data leakage fix)")

    # Identify leaky observations: both revenue growth AND DSO change exactly 0
    # This combination is virtually impossible for real year-over-year changes
    leaky_mask = (df["ny_omsf"] == 0.0) & (df["dso_days_yoy_diff"] == 0.0)
    n_leaky = leaky_mask.sum()

    if n_leaky > 0:
        logger.warning("Found {:,} duplicate accounting year observations to remove", n_leaky)

        # Simply remove the duplicate rows. No need to transfer credit_event because:
        #
        # The data structure is: Year T-1 (valid) → Year T (duplicate) → Year T+1 (bankruptcy)
        # - Year T has YoY features = 0 (because T's financials are copied from T-1)
        # - Year T has target_next_year = 1 (because T+1 has credit_event = 1)
        # - The credit_event = 1 is on Year T+1, NOT on Year T
        #
        # After removing Year T:
        # - Year T-1 still exists with valid features
        # - Year T+1 still exists with credit_event = 1
        # - When we compute target_next_year via shift(-1), Year T-1 will correctly
        #   get target_next_year = 1 (pointing to T+1's credit_event)
        #
        # The target "transfer" happens automatically via the shift operation!

        # Remove all leaky rows
        df = df[~leaky_mask].copy()
        logger.success("Removed {:,} duplicate observations", n_leaky)
    else:
        logger.info("No duplicate accounting year observations detected")

    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)
    df.set_index(["ORGNR", "ser_year"], drop=False, inplace=True)

    # Computing target variable
    logger.info("Creating target variable for next-year credit events")
    df["target_next_year"] = (
        df.groupby(level=0, group_keys=False)["credit_event"].shift(-1).astype("Int8")
    )
    df.reset_index(drop=True, inplace=True)

    # Note: Raw rr_*/br_* columns (except KEPT_RAW_COLS) were already dropped after computing ratios
    # So drop_raw_sources parameter is now redundant, but we keep it for API compatibility

    # Drop temporary columns used for intermediate calculations
    temp_cols_to_drop = ["_ebitda_temp"]
    df.drop(columns=[c for c in temp_cols_to_drop if c in df.columns], inplace=True, errors="ignore")

    # Note: Categorical dtypes are already set in make_dataset.py
    # They are preserved through parquet as dictionary<values=string>

    # Optimize data types for memory efficiency
    logger.info("Optimizing data types for memory efficiency")

    # any_event_last_5y: binary (0/1) already as Int8 (optimal)

    # Downcast float64 to float32 where precision not critical
    float64_to_32 = [
        'revenue_drawdown_5y', 'dpo_days_trend_3y',
        # V2 features
        'working_capital_ta', 'retained_earnings_ta', 'interest_coverage',
        'gross_margin', 'ebitda_volatility',
    ]
    for col in float64_to_32:
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
            logger.debug(f"  Optimized {col}: float64 → float32")

    logger.success("Feature engineering completed (rows={:,}, columns={})", len(df), df.shape[1])

    return df


def prepare_modeling_data(
    df: pd.DataFrame,
    features: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract modeling features and target from a pre-filtered DataFrame.

    Filters for valid targets (non-null target_next_year) and returns views
    of the feature matrix X and target vector y.

    Args:
        df: DataFrame with engineered features and target_next_year column.
            Should be pre-filtered by user (e.g., revenue threshold, SME category).
        features: Optional list of features to use. If None, uses ACTIVE_FEATURES
            from config (which respects ACTIVE_MODEL_VERSION setting).

    Returns:
        X: Feature matrix view (selected feature columns)
        y: Target vector view (target_next_year)
    """
    # Import here to avoid circular imports and get latest ACTIVE_FEATURES
    from credit_risk_xai.config import ACTIVE_FEATURES

    if features is None:
        features = ACTIVE_FEATURES

    valid_mask = df["target_next_year"].notna()
    X = df.loc[valid_mask, features]
    y = df.loc[valid_mask, "target_next_year"]
    return X, y


def build_feature_matrix(
    base_path: Path = BASE_CACHE_PATH,
    macro_path: Path = MACRO_CACHE_PATH,
    output_path: Path = FEATURE_CACHE_PATH,
    force: bool = False,
) -> Path:
    """
    Load interim caches, engineer features, and persist to processed parquet.
    """
    if output_path.exists() and not force:
        logger.info("Feature cache already exists at {} (use --force to rebuild)", output_path)
        return output_path

    if not base_path.exists():
        raise FileNotFoundError(
            f"Interim Serrano dataset not found: {base_path}. Run the raw preprocessing step first."
        )
    if not macro_path.exists():
        raise FileNotFoundError(
            f"Macro summary not found: {macro_path}. Run the macro preprocessing step first."
        )

    base_df = pd.read_parquet(base_path)
    macro_df = pd.read_parquet(macro_path)

    logger.info("Engineering features from interim dataset (rows={:,})", len(base_df))
    engineered_df = create_engineered_features(base_df, macro_df=macro_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    logger.success("Feature matrix saved to {} (rows={:,})", output_path, len(engineered_df))
    return output_path


if __name__ == "__main__":
    build_feature_matrix()
