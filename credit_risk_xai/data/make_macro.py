from __future__ import annotations

from pathlib import Path
import pandas as pd
from loguru import logger

from credit_risk_xai.config import EXTERNAL_DATA_DIR, MACRO_CACHE_PATH

# Lending rates CSV from SCB
LENDING_RATES_FILE = "000004ZT_20251025-133456.csv"


def build_macro_summary(
    data_dir: Path = EXTERNAL_DATA_DIR,
    output_path: Path = MACRO_CACHE_PATH,
    force: bool = False,
) -> Path:
    """
    Transform raw lending rates CSV into an annual term spread table and cache to parquet.

    Data Source: Statistics Sweden (SCB) - Lending Rates to Households and Non-Financial
    Corporations, Breakdown by Fixation Periods.
    URL: https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__FM__FM5001__FM5001C/RantaT01N/

    Steps:
        1. Load monthly lending rate data
        2. Filter for non-financial corporations, new and renegotiated agreements
        3. Compute annual averages for short-term and long-term rates
        4. Calculate term_spread = long_term_rate - short_term_rate
    """
    if output_path.exists() and not force:
        logger.info("Macro summary already exists at {} (use --force to rebuild)", output_path)
        return output_path

    data_dir = Path(data_dir)
    rates_path = data_dir / LENDING_RATES_FILE
    if not rates_path.exists():
        raise FileNotFoundError(f"Lending rates CSV not found: {rates_path}")

    logger.info("Building macro summary from {}", rates_path)

    # Load and process lending rates
    df = pd.read_csv(rates_path)
    df = df.rename(
        columns={
            "month": "period",
            "Up to 3 months (floating rate)": "rate_short",
            "Over 5 years": "rate_long",
        }
    )

    # Filter for relevant observations
    selector = (
        (df["reference sector"].str.startswith("1 monetary financial institutions"))
        & (df["counterparty sector"].str.contains("Non-financial corporations"))
        & (df["agreement"].str.contains("new and renegotiated agreements"))
    )
    df = df.loc[selector].copy()
    df["ser_year"] = df["period"].str.slice(0, 4).astype("Int32")

    # Convert to numeric
    for col in ["rate_short", "rate_long"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Compute annual averages and term spread
    annual = (
        df.groupby("ser_year")[["rate_short", "rate_long"]]
        .mean()
        .reset_index()
    )
    annual["term_spread"] = annual["rate_long"] - annual["rate_short"]

    # Keep only the columns needed for the model
    macro = annual[["ser_year", "term_spread"]].copy()
    macro["term_spread"] = macro["term_spread"].astype("float32")

    logger.info("Computed term_spread for {} years", len(macro))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(output_path, index=False)
    logger.success("Macro summary saved to {} (rows={:,})", output_path, len(macro))
    return output_path


if __name__ == "__main__":
    build_macro_summary()
