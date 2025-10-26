from __future__ import annotations

from pathlib import Path
import pandas as pd
import typer
from loguru import logger

from credit_risk_xai.config import EXTERNAL_DATA_DIR, MACRO_CACHE_PATH

app = typer.Typer(help="Commands for aggregating macroeconomic datasets.")


def build_macro_summary(
    data_dir: Path = EXTERNAL_DATA_DIR,
    output_path: Path = MACRO_CACHE_PATH,
    force: bool = False,
) -> Path:
    """
    Transform raw macro CSVs into an annual summary table and cache to parquet.
    """
    if output_path.exists() and not force:
        logger.info("Macro summary already exists at {} (use --force to rebuild)", output_path)
        return output_path

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Macro data directory not found: {data_dir}")

    logger.info("Building macro summary from {}", data_dir)
    datasets = [
        ("GDP", _load_gdp(data_dir / "000004KA_20251025-125856.csv")),
        ("Borrowing rates", _load_rates(data_dir / "000004ZT_20251025-133456.csv")),
        ("Inflation (KPIF)", _load_inflation(data_dir / "000005HR_20251025-132523.csv")),
        ("Unemployment", _load_unemployment(data_dir / "AM04011Q_20251025-130707.csv")),
    ]

    for name, frame in datasets:
        logger.info("Loaded {} table with {} annual rows", name, len(frame))

    frames = [frame for _, frame in datasets]

    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="ser_year", how="outer")

    macro = macro.sort_values("ser_year").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(output_path, index=False)
    logger.success("Macro summary saved to {} (rows={:,})", output_path, len(macro))
    return output_path


def _load_gdp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "year": "ser_year",
            "0010 GDP at market prices": "gdp_growth",
        }
    )
    df["ser_year"] = pd.to_numeric(df["ser_year"], errors="coerce").astype("Int64")
    df["gdp_growth"] = pd.to_numeric(df["gdp_growth"], errors="coerce")
    df = df.dropna(subset=["ser_year", "gdp_growth"])
    df["gdp_growth_3y_avg"] = df["gdp_growth"].rolling(window=3, min_periods=2).mean()
    return df[["ser_year", "gdp_growth", "gdp_growth_3y_avg"]]


def _load_rates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "month": "period",
            "Up to 3 months (floating rate)": "rate_short",
            "One to five years (1-5 years)": "rate_medium",
            "Over 5 years": "rate_long",
        }
    )

    selector = (
        (df["reference sector"].str.startswith("1 monetary financial institutions"))
        & (df["counterparty sector"].str.contains("Non-financial corporations"))
        & (df["agreement"].str.contains("new and renegotiated agreements"))
    )
    df = df.loc[selector].copy()
    df["ser_year"] = df["period"].str.slice(0, 4).astype(int)

    numeric_cols = ["All accounts", "rate_short", "rate_medium", "rate_long"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    annual = (
        df.groupby("ser_year")[numeric_cols]
        .mean()
        .rename(
            columns={
                "All accounts": "interest_avg_all",
                "rate_short": "interest_avg_short",
                "rate_medium": "interest_avg_medium",
                "rate_long": "interest_avg_long",
            }
        )
        .reset_index()
    )
    annual["interest_delta_short"] = annual["interest_avg_short"].diff()
    annual["term_spread"] = annual["interest_avg_long"] - annual["interest_avg_short"]
    annual["term_spread_delta"] = annual["term_spread"].diff()
    return annual.drop(columns=["interest_avg_all"])


def _load_inflation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "period", df.columns[1]: "kpif_index"})
    df["ser_year"] = df["period"].str.slice(0, 4).astype(int)
    df["kpif_index"] = pd.to_numeric(df["kpif_index"], errors="coerce")

    annual = (
        df.groupby("ser_year")["kpif_index"]
        .mean()
        .rename("kpif_index_avg")
        .reset_index()
    )
    annual["inflation_yoy"] = annual["kpif_index_avg"].pct_change()
    annual["inflation_trailing_3y"] = (
        annual["inflation_yoy"].rolling(window=3, min_periods=2).mean()
    )
    return annual.drop(columns=["kpif_index_avg"])


def _load_unemployment(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    df = df.rename(columns={"år": "ser_year", "arbetslösa": "unemp_rate"})
    df = df[df["kön"] == "totalt"]
    df = df[df["ålder"].str.startswith("totalt", na=False)]
    df["ser_year"] = pd.to_numeric(df["ser_year"], errors="coerce").astype("Int64")
    df["unemp_rate"] = pd.to_numeric(df["unemp_rate"], errors="coerce")
    df = df.dropna(subset=["ser_year", "unemp_rate"])
    df = (
        df[["ser_year", "unemp_rate"]]
        .drop_duplicates("ser_year")
        .sort_values("ser_year")
        .reset_index(drop=True)
    )
    df["unemp_delta"] = df["unemp_rate"].diff()
    return df


@app.command()
def main(
    data_dir: Path = typer.Option(EXTERNAL_DATA_DIR, help="Directory with macro CSV files."),
    output_path: Path = typer.Option(MACRO_CACHE_PATH, help="Destination parquet file."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing cache."),
) -> None:
    """CLI wrapper for building the macroeconomic summary parquet."""
    build_macro_summary(data_dir=data_dir, output_path=output_path, force=force)


if __name__ == "__main__":
    app()
