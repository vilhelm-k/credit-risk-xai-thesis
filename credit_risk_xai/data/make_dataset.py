from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from credit_risk_xai.config import (
    BASE_CACHE_PATH,
    CATEGORICAL_COLS,
    COLS_TO_LOAD,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    SME_CATEGORIES,
)


def classify_sme_eu_vectorized(
    ser_stklf: pd.Series, revenue_ksek: pd.Series, total_assets_ksek: pd.Series
) -> pd.Series:
    """Vectorized SME classification using the strict EU definition."""
    # Convert to MEUR (11 SEK ~= 1 EUR exchange rate)
    rev_meur = revenue_ksek.fillna(0) / 11_000
    assets_meur = total_assets_ksek.fillna(0) / 11_000

    # Define conditions in order of priority (most specific first)
    conditions = [
        # Unknown: missing or invalid employee count
        ser_stklf.isna() | (ser_stklf == 9),
        # Micro: <= 2 employees AND (revenue <= 2M EUR OR assets <= 2M EUR)
        (ser_stklf <= 2) & ((rev_meur <= 2) | (assets_meur <= 2)),
        # Small: <= 4 employees AND (revenue <= 10M EUR OR assets <= 10M EUR)
        (ser_stklf <= 4) & ((rev_meur <= 10) | (assets_meur <= 10)),
        # Medium: <= 6 employees AND (revenue <= 50M EUR OR assets <= 43M EUR)
        (ser_stklf <= 6) & ((rev_meur <= 50) | (assets_meur <= 43)),
    ]

    choices = ["Unknown", "Micro", "Small", "Medium"]

    # Default to "Large" if none of the conditions match
    return pd.Series(
        np.select(conditions, choices, default="Large"),
        index=ser_stklf.index,
        dtype="category",
    )


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize column data types to reduce memory usage.

    Conversions:
    - Identifiers (ORGNR, ser_year) → Int64/Int32
    - Binary indicators (0/1) → Int8
    - Small integer codes → Int8/Int32 or category (for CATEGORICAL_COLS)
    - Financial data (kSEK, ratios) → float32

    Expected memory reduction: 40-50%
    """
    logger.debug("Optimizing data types...")

    # Phase 0: Core identifiers
    df["ORGNR"] = pd.to_numeric(df["ORGNR"], errors="coerce").astype("Int64")
    df["ser_year"] = pd.to_numeric(df["ser_year"], errors="coerce").astype("Int32")

    # Phase 1: Binary indicators (0/1) → Int8
    binary_cols = ["bol_konkurs", "ser_aktiv", "ser_nystartat"]
    for col in binary_cols:
        df[col] = df[col].astype("Int8")

    # Phase 2: Small ordinal/categorical integers
    df["ser_stklf"] = df["ser_stklf"].astype("Int8")
    df["bslov_antanst"] = df["bslov_antanst"].astype("Int32")

    # Categorical columns: convert to string (will apply category dtype after concat)
    # This avoids concat losing categorical dtypes for high-cardinality columns
    # Skip sme_category as it's already a properly formatted category from classify_sme_eu_vectorized
    for col in CATEGORICAL_COLS:
        if col in df.columns and col != "sme_category":
            df[col] = df[col].astype(str)
            logger.debug(f"  Converted {col} to string")

    # Other numeric codes that are NOT in CATEGORICAL_COLS
    if "knc_kncfall" not in CATEGORICAL_COLS:
        df["knc_kncfall"] = df["knc_kncfall"].astype("Int8")

    # Phase 3: Downcast all financial data to float32
    # Financial ratios (ny_* columns)
    ny_cols = [col for col in df.columns if col.startswith("ny_")]
    for col in ny_cols:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    # Income statement (rr_* columns) - all in kSEK
    rr_cols = [col for col in df.columns if col.startswith("rr")]
    for col in rr_cols:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    # Balance sheet (br_* columns) - all in kSEK
    br_cols = [col for col in df.columns if col.startswith("br")]
    for col in br_cols:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    logger.debug("Data type optimization completed")
    return df


def generate_serrano_base(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = BASE_CACHE_PATH,
    force: bool = False,
) -> Path:
    """
    Read raw Serrano Stata files and produce a cleaned interim parquet.

    Steps:
        - Load limited-liability companies (ser_jurform == 49)
        - Add company_age, credit_event, SME classification
        - Enforce categorical dtypes
        - Persist to Parquet for downstream feature engineering
    """
    if output_path.exists() and not force:
        logger.info("Interim base already exists at {} (use --force to rebuild)", output_path)
        return output_path

    raw_files = sorted(glob.glob(str(raw_dir / "serrano*.dta")))
    if not raw_files:
        raise FileNotFoundError(f"No Serrano .dta files found under {raw_dir}")

    logger.info("Building interim dataset from {} source files", len(raw_files))
    frames = []
    total_rows = 0

    for idx, path_str in enumerate(raw_files, start=1):
        path = Path(path_str)
        logger.info("[{}/{}] Loading {}", idx, len(raw_files), path.name)
        df = pd.read_stata(path, columns=COLS_TO_LOAD)
        logger.debug("    Raw rows loaded: {:,}", len(df))
        total_rows += len(df)

        # Filter for limited liability companies only (no need for .copy())
        df = df[df["ser_jurform"] == 49.0]
        df.drop(columns=["ser_jurform"], inplace=True)

        # Compute derived features (using optimized dtypes)
        df["company_age"] = (df["ser_year"] - df["ser_regdat"].dt.year).astype("Int16")

        # credit_event: Binary indicator for credit distress events in THIS specific year
        # - bol_konkurs == 1: Bankruptcy filed this year
        # - bol_q80dat: Date of last reorganization (persists across years)
        #   → Only flag as event if the YEAR of bol_q80dat matches ser_year
        # This ensures we only mark the year when the event OCCURRED, not subsequent years
        q80_year_matches = df["bol_q80dat"].dt.year == df["ser_year"]
        credit_event_mask = (df["bol_konkurs"] == 1) | q80_year_matches
        df["credit_event"] = credit_event_mask.astype("Int8")

        # Vectorized SME classification (much faster than apply)
        df["sme_category"] = classify_sme_eu_vectorized(
            df["ser_stklf"], df["rr01_ntoms"], df["br09_tillgsu"]
        )
        # Optimize all data types in one place
        df = _optimize_dtypes(df)

        # Ensure proper categorical with all possible categories
        df["sme_category"] = df["sme_category"].cat.set_categories(SME_CATEGORIES + ["Unknown"])

        frames.append(df)

    if not frames:
        raise RuntimeError("No frames loaded from raw files.")

    interim_df = pd.concat(frames, ignore_index=True)
    interim_df.sort_values(["ORGNR", "ser_year"], inplace=True)

    # Apply categorical dtypes after concat (columns are strings at this point, except sme_category)
    # This ensures PyArrow can preserve them as dictionary<values=string>
    logger.info("Applying categorical dtypes to configured columns")
    for col in CATEGORICAL_COLS:
        if col in interim_df.columns:
            # sme_category is already category dtype, just verify it
            if col == "sme_category":
                if interim_df[col].dtype.name != 'category':
                    interim_df[col] = interim_df[col].astype('category')
                logger.debug(f"  Verified {col} is category dtype")
            else:
                # Other columns are strings, convert to category
                interim_df[col] = interim_df[col].astype('category')
                logger.debug(f"  Converted {col} to category dtype")

    # Log memory usage
    memory_mb = interim_df.memory_usage(deep=True).sum() / 1024**2
    logger.info("Dataset memory usage: {:.2f} MB ({:.2f} GB)", memory_mb, memory_mb / 1024)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Writing interim dataset with {:,} rows to {}", len(interim_df), output_path)
    interim_df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)

    logger.success(
        "Interim dataset built: {} (raw rows={:,}, filtered rows={:,})",
        output_path,
        total_rows,
        len(interim_df),
    )
    return output_path


if __name__ == "__main__":
    generate_serrano_base()
