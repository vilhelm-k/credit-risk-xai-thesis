from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import typer
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

app = typer.Typer(help="Feature engineering commands.")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator
    mask = denominator.isna() | (denominator == 0)
    if mask.any():
        result = result.mask(mask)
    return result.replace([np.inf, -np.inf], np.nan)


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

    # Set index without dropping columns (we'll drop at the end if needed)
    df.set_index(["ORGNR", "ser_year"], drop=False, inplace=True)

    # Create groupby object once and reuse it throughout
    group = df.groupby(level=0, group_keys=False)

    # Collect ALL new features in a single dictionary to minimize joins
    new_features = {}

    logger.info("Computing cost structure and profitability ratios")
    # Compute intermediate values once
    ebitda = df["rr07_rorresul"] + df["rr05_avskriv"]
    financial_cost_net = df["rr09_finkostn"] - df["rr09d_jfrstfin"]

    new_features.update({
        "ratio_personnel_cost": _safe_div(
            df["rr04_perskos"], df["rr01_ntoms"]
        ),
        "ratio_depreciation_cost": _safe_div(
            df["rr05_avskriv"], df["rr01_ntoms"]
        ),
        "ratio_other_operating_cost": _safe_div(
            df["rr06_rorkoov"], df["rr01_ntoms"]
        ),
        "ratio_financial_cost": _safe_div(
            df["rr09_finkostn"], df["rr01_ntoms"]
        ),
        "ratio_ebitda_margin": _safe_div(ebitda, df["rr01_ntoms"]),
        "ratio_ebit_interest_cov": _safe_div(
            df["rr07_rorresul"], financial_cost_net
        ),
        # "ratio_ebitda_interest_cov": _safe_div(ebitda, financial_cost_net),  # REMOVED: r=0.99 with ratio_ebit_interest_cov
        "ratio_cash_interest_cov": _safe_div(
            df["br07b_kabasu"], financial_cost_net
        ),
        "ratio_intragroup_financing_share": _safe_div(
            df["rr08a_rteinknc"] + df["rr09a_rtekoknc"],
            df["rr08_finintk"] + df["rr09_finkostn"],
        ),
    })

    logger.info("Computing liquidity and working-capital efficiencies")
    total_debt = df["br13_ksksu"] + df["br15_lsksu"]

    new_features.update({
        "ratio_cash_liquidity": _safe_div(
            df["br07b_kabasu"] + df["br07a_kplacsu"],
            df["br13_ksksu"],
        ),
        "dso_days": _safe_div(df["br06g_kfordsu"], df["rr01_ntoms"]) * 365,
        "inventory_days": _safe_div(df["br06c_lagersu"], df["rr06a_prodkos"])
        * 365,
        "dpo_days": _safe_div(df["br13a_ksklev"], df["rr06a_prodkos"]) * 365,
        "cash_conversion_cycle": _safe_div(df["br06g_kfordsu"], df["rr01_ntoms"]) * 365
        + _safe_div(df["br06c_lagersu"], df["rr06a_prodkos"]) * 365
        - _safe_div(df["br13a_ksklev"], df["rr06a_prodkos"]) * 365,
        "ratio_nwc_sales": _safe_div(
            df["br06_lagerkford"]
            + df["br07_kplackaba"]
            - df["br13_ksksu"],
            df["rr01_ntoms"],
        ),
    })

    logger.info("Computing capital structure and payout ratios")
    new_features.update({
        "ratio_short_term_debt_share": _safe_div(df["br13_ksksu"], total_debt),
        "ratio_secured_debt_assets": _safe_div(
            df["br14_kskkrin"] + df["br16_lskkrin"], df["br09_tillgsu"]
        ),
        "ratio_retained_earnings_equity": _safe_div(
            df["br10e_balres"], df["br10_eksu"]
        ),
        # "ratio_share_capital_equity": _safe_div(
        #     df["br10a_aktiekap"], df["br10_eksu"]
        # ), # REMOVED: Low importance
        "equity_to_sales": _safe_div(df["br10_eksu"], df["rr01_ntoms"]),
        "equity_to_profit": _safe_div(df["br10_eksu"], df["rr15_resar"]),
        "assets_to_profit": _safe_div(df["br09_tillgsu"], df["rr15_resar"]),
        "ratio_dividend_payout": _safe_div(
            df["rr00_utdbel"], df["rr15_resar"]
        ),
        "ratio_group_support": _safe_div(
            df["br10f_kncbdrel"] + df["br10g_agtskel"], df["rr01_ntoms"]
        ),
    })

    # Join basic ratios early so they can be used in trend calculations
    df = df.join(pd.DataFrame(new_features, index=df.index))
    new_features.clear()  # Clear to prepare for next batch

    # Drop ALL rr_* and br_* columns except KEPT_RAW_COLS to save memory
    # KEPT_RAW_COLS = rr01_ntoms, br09_tillgsu, br10_eksu, bslov_antanst,
    #                 br07b_kabasu, br13_ksksu, br15_lsksu, rr07_rorresul, rr15_resar
    cols_to_drop = [col for col in (RR_SOURCE_COLS + BR_SOURCE_COLS) if col in df.columns and col not in KEPT_RAW_COLS]
    df.drop(columns=cols_to_drop, inplace=True)
    gc.collect()  # Force garbage collection to free memory immediately
    logger.debug("Dropped {} raw rr/br columns after computing basic ratios (kept only KEPT_RAW_COLS)", len(cols_to_drop))

    # Recreate groupby after joining new features
    group = df.groupby(level=0, group_keys=False)

    logger.info("Computing year-over-year deltas and immediate trends")
    for col in ["rr01_ntoms", "rr07_rorresul", "br09_tillgsu"]:
        new_features[f"{col}_yoy_pct"] = group[col].pct_change(fill_method=None)
        # new_features[f"{col}_yoy_abs"] = group[col].diff() # REMOVED: Redundant with _pct
    new_features.update(
        {
            "ny_solid_yoy_diff": group["ny_solid"].diff(),
            "ny_skuldgrd_yoy_diff": group["ny_skuldgrd"].diff(),
            "ratio_cash_liquidity_yoy_pct": group["ratio_cash_liquidity"].pct_change(
                fill_method=None
            ),
            "ratio_cash_liquidity_yoy_abs": group["ratio_cash_liquidity"].diff(),
            "ratio_ebit_interest_cov_yoy_pct": group["ratio_ebit_interest_cov"].pct_change(
                fill_method=None
            ),
        }
    )

    # Compute CAGR features
    for source, target, window in [
        ("rr01_ntoms", "revenue_cagr_3y", 3),
        ("br09_tillgsu", "assets_cagr_3y", 3),
        ("br10_eksu", "equity_cagr_3y", 3),
        ("rr15_resar", "profit_cagr_3y", 3),
        ("rr01_ntoms", "revenue_cagr_5y", 5),
        ("br09_tillgsu", "assets_cagr_5y", 5),
        ("br10_eksu", "equity_cagr_5y", 5),
        ("rr15_resar", "profit_cagr_5y", 5),
    ]:
        new_features[target] = _compute_cagr(df[source], window)

    # Join trends and CAGR before computing rolling features that depend on them
    df = df.join(pd.DataFrame(new_features, index=df.index))
    new_features.clear()

    logger.info("Computing rolling slopes, volatility, averages, and drawdowns")
    # Continue collecting features to minimize joins
    new_features.update({
        "ny_rormarg_trend_3y": _rolling_slope(df["ny_rormarg"], window=3),
        "ny_nettomarg_trend_3y": _rolling_slope(df["ny_nettomarg"], window=3),
        "ny_skuldgrd_trend_3y": _rolling_slope(df["ny_skuldgrd"], window=3),
        "ratio_cash_liquidity_trend_3y": _rolling_slope(df["ratio_cash_liquidity"], window=3),
        "dso_days_yoy_diff": group["dso_days"].diff(),
        "inventory_days_yoy_diff": group["inventory_days"].diff(),
        "dpo_days_yoy_diff": group["dpo_days"].diff(),
        "dso_days_trend_3y": _rolling_slope(df["dso_days"], window=3),
        "inventory_days_trend_3y": _rolling_slope(df["inventory_days"], window=3),
        "dpo_days_trend_3y": _rolling_slope(df["dpo_days"], window=3),
        "ny_rormarg_trend_5y": _rolling_slope(df["ny_rormarg"], window=5),
        "ny_nettomarg_trend_5y": _rolling_slope(df["ny_nettomarg"], window=5),
        "ny_skuldgrd_trend_5y": _rolling_slope(df["ny_skuldgrd"], window=5),
        "ratio_cash_liquidity_trend_5y": _rolling_slope(df["ratio_cash_liquidity"], window=5),
        "ny_rormarg_vol_3y": _rolling_std(df["ny_rormarg"], window=3),
        "ny_nettomarg_vol_3y": _rolling_std(df["ny_nettomarg"], window=3),
        "ny_skuldgrd_vol_3y": _rolling_std(df["ny_skuldgrd"], window=3),
        "ratio_cash_liquidity_vol_3y": _rolling_std(df["ratio_cash_liquidity"], window=3),
        "ny_rormarg_vol_5y": _rolling_std(df["ny_rormarg"], window=5),
        "ny_nettomarg_vol_5y": _rolling_std(df["ny_nettomarg"], window=5),
        "ny_skuldgrd_vol_5y": _rolling_std(df["ny_skuldgrd"], window=5),
        "ny_rormarg_avg_2y": _rolling_avg(df["ny_rormarg"], window=2),
        "ny_nettomarg_avg_2y": _rolling_avg(df["ny_nettomarg"], window=2),
        "ratio_cash_liquidity_avg_2y": _rolling_avg(df["ratio_cash_liquidity"], window=2),
        "ny_rormarg_avg_5y": _rolling_avg(df["ny_rormarg"], window=5),
        "ny_nettomarg_avg_5y": _rolling_avg(df["ny_nettomarg"], window=5),
        "ratio_cash_liquidity_avg_5y": _rolling_avg(df["ratio_cash_liquidity"], window=5),
        "revenue_drawdown_5y": _rolling_drawdown(df["rr01_ntoms"], window=5),
        "equity_drawdown_5y": _rolling_drawdown(df["br10_eksu"], window=5),
    })

    # Join all rolling features at once
    df = df.join(pd.DataFrame(new_features, index=df.index))
    new_features.clear()

    credit_events = df["credit_event"] == 1
    event_years = df["ser_year"].where(credit_events)
    last_event_year = event_years.groupby(level=0).ffill()
    df["years_since_last_credit_event"] = df["ser_year"] - last_event_year
    df.loc[last_event_year.isna(), "years_since_last_credit_event"] = np.nan

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

    df["event_count_total"] = group["credit_event"].cumsum().astype("Int16")
    df["event_count_last_5y"] = (
        group["credit_event"]
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .astype("Int16")
    )
    # df["ever_failed"] = (df["event_count_total"] > 0).astype("Int8")  # REMOVED: Zero importance, redundant with event_count_total

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
    df["revenue_beta_gdp_5y"] = _rolling_beta(
        df["rr01_ntoms_yoy_pct"],
        df["gdp_growth"],
        window=5,
        min_periods=4,
    )

    logger.info("Macro indicators merged (including revenue beta)")

    # Computing target variable
    logger.info("Creating target variable for next-year credit events")
    df["target_next_year"] = (
        df.groupby(level=0, group_keys=False)["credit_event"].shift(-1).astype("Int8")
    )
    df.reset_index(drop=True, inplace=True)

    # Note: Raw rr_*/br_* columns (except KEPT_RAW_COLS) were already dropped after computing ratios
    # So drop_raw_sources parameter is now redundant, but we keep it for API compatibility

    logger.success("Feature engineering completed (rows={:,}, columns={})", len(df), df.shape[1])

    return df


def prepare_modeling_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract modeling features and target from a pre-filtered DataFrame.

    Filters for valid targets (non-null target_next_year) and returns views
    of the feature matrix X and target vector y.

    Args:
        df: DataFrame with engineered features and target_next_year column.
            Should be pre-filtered by user (e.g., revenue threshold, SME category).

    Returns:
        X: Feature matrix view (FEATURES_FOR_MODEL columns)
        y: Target vector view (target_next_year)
    """
    valid_mask = df["target_next_year"].notna()
    X = df.loc[valid_mask, FEATURES_FOR_MODEL]
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


@app.command()
def main(
    base_path: Path = typer.Option(BASE_CACHE_PATH, help="Input parquet from make_dataset."),
    macro_path: Path = typer.Option(MACRO_CACHE_PATH, help="Macro parquet (optional)."),
    output_path: Path = typer.Option(FEATURE_CACHE_PATH, help="Destination feature parquet."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing cache."),
) -> None:
    """CLI entry-point to engineer features from interim caches."""
    build_feature_matrix(base_path=base_path, macro_path=macro_path, output_path=output_path, force=force)


if __name__ == "__main__":
    app()
