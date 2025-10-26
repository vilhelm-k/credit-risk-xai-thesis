from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import typer
from loguru import logger
from numba import jit

from credit_risk_xai.config import (
    BASE_CACHE_PATH,
    BR_SOURCE_COLS,
    FEATURE_CACHE_PATH,
    FEATURES_FOR_MODEL,
    KEPT_RAW_COLS,
    MACRO_CACHE_PATH,
    MIN_REVENUE_KSEK,
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


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Vectorized rolling slope calculation using fast linear regression formula."""
    def _slope_vectorized(values: np.ndarray) -> float:
        if np.isnan(values).all():
            return np.nan
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return np.nan

        # Fast vectorized linear regression: slope = cov(x,y) / var(x)
        x = np.arange(len(values))
        x_masked = x[mask]
        y_masked = values[mask]

        x_mean = x_masked.mean()
        y_mean = y_masked.mean()

        # Slope = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
        numerator = np.sum((x_masked - x_mean) * (y_masked - y_mean))
        denominator = np.sum((x_masked - x_mean) ** 2)

        return numerator / denominator if denominator != 0 else np.nan

    return (
        series.groupby(level=0, group_keys=False)
        .rolling(window=window, min_periods=window)
        .apply(_slope_vectorized, raw=True)
    )


def _rolling_avg(series: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(level=0, group_keys=False)
        .rolling(window=window, min_periods=2)
        .mean()
    )


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(level=0, group_keys=False)
        .rolling(window=window, min_periods=window)
        .std()
    )


def _rolling_drawdown(series: pd.Series, window: int) -> pd.Series:
    """Optimized rolling drawdown using vectorized operations."""
    def _max_drawdown_fast(values: np.ndarray) -> float:
        if np.isnan(values).all():
            return np.nan
        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            return np.nan

        valid_values = values[valid_mask]
        running_max = np.maximum.accumulate(valid_values)
        drawdowns = valid_values / running_max - 1.0
        return drawdowns.min()

    return (
        series.groupby(level=0, group_keys=False)
        .rolling(window=window, min_periods=window)
        .apply(_max_drawdown_fast, raw=True)
    )


def _compute_cagr(series: pd.Series, periods: int) -> pd.Series:
    shifted = series.groupby(level=0).shift(periods)
    ratio = _safe_div(series, shifted)
    mask = (series > 0) & (shifted > 0)
    cagr = ratio.pow(1 / periods) - 1
    return cagr.where(mask, np.nan)


@jit(nopython=True)
def _streak_numba(condition_values: np.ndarray) -> np.ndarray:
    """Numba-compiled streak counter for maximum performance."""
    out = np.zeros(len(condition_values), dtype=np.float64)
    count = 0.0
    for idx in range(len(condition_values)):
        if np.isnan(condition_values[idx]):
            out[idx] = np.nan
            count = 0.0
        elif condition_values[idx]:
            count += 1.0
            out[idx] = count
        else:
            count = 0.0
            out[idx] = 0.0
    return out


def _streak(series: pd.Series, comparator) -> pd.Series:
    """Optimized streak calculation using numba-compiled function."""
    diffs = series.groupby(level=0).diff()

    def _streak_group(group: pd.Series) -> pd.Series:
        group_diffs = diffs.loc[group.index].to_numpy()
        group_cond = comparator(group_diffs)
        result = _streak_numba(group_cond)
        return pd.Series(result, index=group.index)

    return series.groupby(level=0, group_keys=False).apply(_streak_group)


# -----------------------------------------------------------------------------
# Feature engineering core
# -----------------------------------------------------------------------------


def create_engineered_features(
    df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    drop_raw_sources: bool = True,
) -> pd.DataFrame:
    """Return a dataframe enriched with engineered features."""
    # Avoid unnecessary copy - work with sorted view
    working_df = df.sort_values(["ORGNR", "ser_year"])
    working_df["ser_stklf"] = working_df["ser_stklf"].replace(9, pd.NA)

    # Set index without dropping columns (we'll drop at the end if needed)
    working_df = working_df.set_index(["ORGNR", "ser_year"], drop=True)
    required_columns = set(
        KEPT_RAW_COLS + RR_SOURCE_COLS + BR_SOURCE_COLS + NY_COLS + ["credit_event"]
    )
    for col in required_columns:
        if col not in working_df.columns:
            working_df[col] = np.nan

    # Create groupby object once and reuse it throughout
    group = working_df.groupby(level=0, group_keys=False)

    # Collect ALL new features in a single dictionary to minimize joins
    new_features = {}

    logger.info("Computing cost structure and profitability ratios")
    # Compute intermediate values once
    ebitda = working_df["rr07_rorresul"] + working_df["rr05_avskriv"]
    financial_cost_net = working_df["rr09_finkostn"] - working_df["rr09d_jfrstfin"]

    new_features.update({
        "ratio_personnel_cost": _safe_div(
            working_df["rr04_perskos"], working_df["rr01_ntoms"]
        ),
        "ratio_depreciation_cost": _safe_div(
            working_df["rr05_avskriv"], working_df["rr01_ntoms"]
        ),
        "ratio_other_operating_cost": _safe_div(
            working_df["rr06_rorkoov"], working_df["rr01_ntoms"]
        ),
        "ratio_financial_cost": _safe_div(
            working_df["rr09_finkostn"], working_df["rr01_ntoms"]
        ),
        "ratio_ebitda_margin": _safe_div(ebitda, working_df["rr01_ntoms"]),
        "ratio_ebit_interest_cov": _safe_div(
            working_df["rr07_rorresul"], financial_cost_net
        ),
        "ratio_ebitda_interest_cov": _safe_div(ebitda, financial_cost_net),
        "ratio_cash_interest_cov": _safe_div(
            working_df["br07b_kabasu"], financial_cost_net
        ),
    })

    logger.info("Computing liquidity and working-capital efficiencies")
    total_debt = working_df["br13_ksksu"] + working_df["br15_lsksu"]

    new_features.update({
        "ratio_cash_liquidity": _safe_div(
            working_df["br07b_kabasu"] + working_df["br07a_kplacsu"],
            working_df["br13_ksksu"],
        ),
        "dso_days": _safe_div(working_df["br06g_kfordsu"], working_df["rr01_ntoms"]) * 365,
        "inventory_days": _safe_div(working_df["br06c_lagersu"], working_df["rr06a_prodkos"])
        * 365,
        "dpo_days": _safe_div(working_df["br13a_ksklev"], working_df["rr06a_prodkos"]) * 365,
        "ratio_nwc_sales": _safe_div(
            working_df["br06_lagerkford"]
            + working_df["br07_kplackaba"]
            - working_df["br13_ksksu"],
            working_df["rr01_ntoms"],
        ),
    })

    logger.info("Computing capital structure and payout ratios")
    new_features.update({
        "ratio_short_term_debt_share": _safe_div(working_df["br13_ksksu"], total_debt),
        "ratio_secured_debt_assets": _safe_div(
            working_df["br14_kskkrin"] + working_df["br16_lskkrin"], working_df["br09_tillgsu"]
        ),
        "ratio_retained_earnings_equity": _safe_div(
            working_df["br10e_balres"], working_df["br10_eksu"]
        ),
        "ratio_share_capital_equity": _safe_div(
            working_df["br10a_aktiekap"], working_df["br10_eksu"]
        ),
        "ratio_dividend_payout": _safe_div(
            working_df["rr00_utdbel"], working_df["rr15_resar"]
        ),
        "ratio_group_support": _safe_div(
            working_df["br10f_kncbdrel"] + working_df["br10g_agtskel"], working_df["rr01_ntoms"]
        ),
    })

    # Join basic ratios early so they can be used in trend calculations
    working_df = working_df.join(pd.DataFrame(new_features, index=working_df.index))
    new_features.clear()  # Clear to prepare for next batch

    logger.info("Computing year-over-year deltas and immediate trends")
    # Reuse the same groupby object - no need to recreate
    for col in ["rr01_ntoms", "rr07_rorresul", "br09_tillgsu"]:
        new_features[f"{col}_yoy_pct"] = group[col].pct_change(fill_method=None)
        new_features[f"{col}_yoy_abs"] = group[col].diff()
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
        new_features[target] = _compute_cagr(working_df[source], window)

    # Join trends and CAGR before computing rolling features that depend on them
    working_df = working_df.join(pd.DataFrame(new_features, index=working_df.index))
    new_features.clear()

    logger.info("Computing rolling slopes, volatility, averages, and drawdowns")
    # Continue collecting features to minimize joins
    new_features.update({
        "ny_rormarg_trend_3y": _rolling_slope(working_df["ny_rormarg"], window=3),
        "ny_skuldgrd_trend_3y": _rolling_slope(working_df["ny_skuldgrd"], window=3),
        "ratio_cash_liquidity_trend_3y": _rolling_slope(
            working_df["ratio_cash_liquidity"], window=3
        ),
        "dso_days_yoy_diff": group["dso_days"].diff(),
        "inventory_days_yoy_diff": group["inventory_days"].diff(),
        "dpo_days_yoy_diff": group["dpo_days"].diff(),
        "dso_days_trend_3y": _rolling_slope(working_df["dso_days"], window=3),
        "inventory_days_trend_3y": _rolling_slope(working_df["inventory_days"], window=3),
        "dpo_days_trend_3y": _rolling_slope(working_df["dpo_days"], window=3),
        "ny_rormarg_trend_5y": _rolling_slope(working_df["ny_rormarg"], window=5),
        "ny_skuldgrd_trend_5y": _rolling_slope(working_df["ny_skuldgrd"], window=5),
        "ratio_cash_liquidity_trend_5y": _rolling_slope(
            working_df["ratio_cash_liquidity"], window=5
        ),
        "ny_rormarg_vol_3y": _rolling_std(working_df["ny_rormarg"], window=3),
        "ny_skuldgrd_vol_3y": _rolling_std(working_df["ny_skuldgrd"], window=3),
        "ratio_cash_liquidity_vol_3y": _rolling_std(
            working_df["ratio_cash_liquidity"], window=3
        ),
        "ny_rormarg_vol_5y": _rolling_std(working_df["ny_rormarg"], window=5),
        "ny_skuldgrd_vol_5y": _rolling_std(working_df["ny_skuldgrd"], window=5),
        "ny_rormarg_avg_2y": _rolling_avg(working_df["ny_rormarg"], window=2),
        "ratio_cash_liquidity_avg_2y": _rolling_avg(
            working_df["ratio_cash_liquidity"], window=2
        ),
        "ny_rormarg_avg_5y": _rolling_avg(working_df["ny_rormarg"], window=5),
        "ratio_cash_liquidity_avg_5y": _rolling_avg(
            working_df["ratio_cash_liquidity"], window=5
        ),
        "revenue_drawdown_5y": _rolling_drawdown(working_df["rr01_ntoms"], window=5),
        "equity_drawdown_5y": _rolling_drawdown(working_df["br10_eksu"], window=5),
    })

    logger.info("Computing streak indicators")
    new_features.update({
        "ny_rormarg_down_streak": _streak(
            working_df["ny_rormarg"], lambda diff: diff < 0
        ),
        "ny_skuldgrd_up_streak": _streak(
            working_df["ny_skuldgrd"], lambda diff: diff > 0
        ),
        "ratio_cash_liquidity_down_streak": _streak(
            working_df["ratio_cash_liquidity"], lambda diff: diff < 0
        ),
    })

    # Join all rolling and streak features at once
    working_df = working_df.join(pd.DataFrame(new_features, index=working_df.index))
    new_features.clear()

    credit_events = working_df["credit_event"] == 1
    event_years = working_df["ser_year"].where(credit_events)
    last_event_year = event_years.groupby(level=0).ffill()
    working_df["years_since_last_credit_event"] = working_df["ser_year"] - last_event_year
    working_df.loc[last_event_year.isna(), "years_since_last_credit_event"] = np.nan

    horizons = {
        "last_event_within_1y": 1,
        "last_event_within_2y": 2,
        "last_event_within_3y": 3,
        "last_event_within_5y": 5,
    }
    for col, limit in horizons.items():
        working_df[col] = (
            working_df["years_since_last_credit_event"].le(limit).fillna(False).astype("Int8")
        )

    working_df["event_count_total"] = group["credit_event"].cumsum().astype("Int16")
    working_df["event_count_last_5y"] = (
        group["credit_event"]
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .astype("Int16")
    )
    working_df["ever_failed"] = (working_df["event_count_total"] > 0).astype("Int8")

    logger.info("Credit event history features computed")

    if macro_df is None:
        raise ValueError("Macro dataframe is required. Run the macro preprocessing step first.")

    logger.info("Merging macroeconomic indicators and derived comparisons")
    # Optimized: Use single merge instead of loop with map
    macro_aligned = macro_df.set_index("ser_year")

    # Create temporary DataFrame with ser_year from index to merge on
    temp_df = working_df.reset_index()[['ser_year']].join(macro_aligned, on='ser_year')
    temp_df.index = working_df.index

    # Add all macro columns at once
    for col in macro_aligned.columns:
        working_df[col] = temp_df[col].values

    working_df["real_revenue_growth"] = (
        working_df["rr01_ntoms_yoy_pct"] - working_df["inflation_yoy"]
    )
    working_df["revenue_vs_gdp"] = (
        working_df["rr01_ntoms_yoy_pct"] - working_df["gdp_growth"]
    )
    working_df["profit_vs_gdp"] = (
        working_df["rr07_rorresul_yoy_pct"] - working_df["gdp_growth"]
    )

    corr = (
        working_df["rr01_ntoms_yoy_pct"]
        .groupby(level=0, group_keys=False)
        .rolling(window=5, min_periods=4)
        .corr(working_df["gdp_growth"])
    )
    working_df["correlation_revenue_gdp_5y"] = corr.reset_index(level=0, drop=True)
    logger.info("Macro indicators merged")

    working_df = working_df.reset_index(drop=True)

    if drop_raw_sources:
        drop_cols = [col for col in RR_SOURCE_COLS + BR_SOURCE_COLS if col in working_df.columns]
        working_df = working_df.drop(columns=drop_cols)
        logger.debug("Dropped {} raw source columns", len(drop_cols))

    logger.success("Feature engineering completed (rows={:,}, columns={})", len(working_df), working_df.shape[1])

    return working_df


def apply_modeling_filters(df: pd.DataFrame, min_revenue_ksek: int = MIN_REVENUE_KSEK) -> pd.DataFrame:
    mask = (df["ser_aktiv"] == 1) & (df["rr01_ntoms"] >= min_revenue_ksek)
    # No need for .copy() - .loc[mask] already returns a new DataFrame
    return df.loc[mask]


def create_target_variable(df: pd.DataFrame) -> pd.Series:
    df.sort_values(["ORGNR", "ser_year"], inplace=True)
    df["target_next_year"] = (
        df.groupby("ORGNR", group_keys=False)["credit_event"].shift(-1).astype("Int8")
    )
    return df["target_next_year"].notna()


def prepare_modeling_data(df: pd.DataFrame, valid_mask: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
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
