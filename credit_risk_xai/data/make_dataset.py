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


def _ensure_categories(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
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

        # read_stata handles datetime conversion automatically, so these lines are removed
        # Ensure numeric types for key columns
        df["ser_year"] = pd.to_numeric(df["ser_year"], errors="coerce").astype("Int32")
        df["ORGNR"] = pd.to_numeric(df["ORGNR"], errors="coerce").astype("Int64")

        # Compute derived features
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

        if "bransch_sni071_konv" in df.columns:
            df["bransch_sni071_konv"] = df["bransch_sni071_konv"].astype("Int32")

        frames.append(df)

    if not frames:
        raise RuntimeError("No frames loaded from raw files.")

    interim_df = pd.concat(frames, ignore_index=True)
    interim_df = _ensure_categories(interim_df, CATEGORICAL_COLS)
    interim_df.sort_values(["ORGNR", "ser_year"], inplace=True)

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
