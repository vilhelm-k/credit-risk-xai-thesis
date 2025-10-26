from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import typer
from loguru import logger

from credit_risk_xai.config import (
    BASE_CACHE_PATH,
    CATEGORICAL_COLS,
    COLS_TO_LOAD,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    SME_CATEGORIES,
)

app = typer.Typer(help="Commands for building interim Serrano datasets.")


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
    - Binary indicators (0/1) → Int8
    - Small integer codes → Int8/Int32
    - Financial data (kSEK, ratios) → float32
    - Categorical codes → category (after int conversion)

    Expected memory reduction: 40-50%
    """
    logger.debug("Optimizing data types...")

    # Phase 1: Binary indicators (0/1) → Int8
    binary_cols = ["bol_konkurs", "ser_aktiv", "ser_nystartat"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int8")

    # Phase 2: Small ordinal/categorical integers
    # ser_stklf: employment size class (0-9)
    if "ser_stklf" in df.columns:
        df["ser_stklf"] = pd.to_numeric(df["ser_stklf"], errors="coerce").astype("Int8")

    # bslov_antanst: employee count (can be large)
    if "bslov_antanst" in df.columns:
        df["bslov_antanst"] = pd.to_numeric(df["bslov_antanst"], errors="coerce").astype("Int32")

    # Industry/sector codes (prepare for categorical)
    # bransch_sni071_konv: 5-digit SNI code (already converted to Int32 above)
    # bransch_borsbransch_konv: 11 sector categories
    if "bransch_borsbransch_konv" in df.columns:
        df["bransch_borsbransch_konv"] = pd.to_numeric(
            df["bransch_borsbransch_konv"], errors="coerce"
        ).astype("Int8")

    # ser_laen: county code (1-21)
    if "ser_laen" in df.columns:
        df["ser_laen"] = pd.to_numeric(df["ser_laen"], errors="coerce").astype("Int8")

    # knc_kncfall: group situation indicator
    if "knc_kncfall" in df.columns:
        df["knc_kncfall"] = pd.to_numeric(df["knc_kncfall"], errors="coerce").astype("Int8")

    # Phase 3: Downcast all financial data to float32
    # Financial ratios (ny_* columns)
    ny_cols = [col for col in df.columns if col.startswith("ny_")]
    for col in ny_cols:
        if col in df.columns and df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    # Income statement (rr_* columns) - all in kSEK
    rr_cols = [col for col in df.columns if col.startswith("rr")]
    for col in rr_cols:
        if col in df.columns and df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    # Balance sheet (br_* columns) - all in kSEK
    br_cols = [col for col in df.columns if col.startswith("br")]
    for col in br_cols:
        if col in df.columns and df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    logger.debug("Data type optimization completed")
    return df


def _ensure_categories(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert specified columns to categorical type."""
    for col in columns:
        if col in df.columns:
            # Convert to category (the column should already be the right integer type)
            df[col] = df[col].astype("category")
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

        # read_stata handles datetime conversion automatically
        # Ensure ORGNR and ser_year have correct types first
        df["ser_year"] = pd.to_numeric(df["ser_year"], errors="coerce").astype("Int32")
        df["ORGNR"] = pd.to_numeric(df["ORGNR"], errors="coerce").astype("Int64")

        # Optimize data types BEFORE computing derived features
        # This includes bransch_sni071_konv conversion to Int32
        if "bransch_sni071_konv" in df.columns:
            df["bransch_sni071_konv"] = pd.to_numeric(
                df["bransch_sni071_konv"], errors="coerce"
            ).astype("Int32")

        df = _optimize_dtypes(df)

        # Compute derived features (using optimized dtypes)
        df["company_age"] = df["ser_year"] - df["ser_regdat"].dt.year
        df["credit_event"] = (
            ((df["bol_konkurs"] == 1) | (df["bol_q80dat"].notna())).astype("int8")
        )

        # Vectorized SME classification (much faster than apply)
        df["sme_category"] = classify_sme_eu_vectorized(
            df["ser_stklf"], df["rr01_ntoms"], df["br09_tillgsu"]
        )
        # Ensure proper categorical with all possible categories
        df["sme_category"] = df["sme_category"].cat.set_categories(SME_CATEGORIES + ["Unknown"])

        frames.append(df)

    if not frames:
        raise RuntimeError("No frames loaded from raw files.")

    interim_df = pd.concat(frames, ignore_index=True)

    # Apply categorical conversion for specified columns
    interim_df = _ensure_categories(interim_df, CATEGORICAL_COLS)
    interim_df.sort_values(["ORGNR", "ser_year"], inplace=True)

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


@app.command()
def main(
    raw_dir: Path = typer.Option(RAW_DATA_DIR, help="Directory containing raw Serrano .dta files."),
    output_path: Path = typer.Option(BASE_CACHE_PATH, help="Destination parquet path."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing cache."),
) -> None:
    """CLI wrapper for generating the interim Serrano dataset."""
    generate_serrano_base(raw_dir=raw_dir, output_path=output_path, force=force)


if __name__ == "__main__":
    app()
