from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from credit_risk_xai.config import BASE_CACHE_PATH, FEATURE_CACHE_PATH, MACRO_CACHE_PATH
from credit_risk_xai.data.make_dataset import generate_serrano_base
from credit_risk_xai.data.make_macro import build_macro_summary
from credit_risk_xai.features.engineer import build_feature_matrix

app = typer.Typer(help="Pipeline helpers for building intermediate and processed datasets.")


@app.command()
def run(
    raw: bool = typer.Option(
        False,
        "--raw/--no-raw",
        help="Rebuild raw → interim Serrano dataset.",
    ),
    macro: bool = typer.Option(
        False,
        "--macro/--no-macro",
        help="Rebuild macroeconomic summary parquet.",
    ),
    features: bool = typer.Option(
        True,
        "--features/--no-features",
        help="Rebuild engineered feature matrix.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing caches for selected stages.",
    ),
) -> None:
    """
    Run selected stages of the data-processing pipeline.

    Defaults to engineering features only. Use flags to include raw or macro rebuilds.
    """
    if raw:
        generate_serrano_base(force=force)
    if macro:
        build_macro_summary(force=force)
    if features:
        build_feature_matrix(force=force)

    if not any([raw, macro, features]):
        logger.warning("No stages selected. Use --raw/--macro/--features to run pipeline steps.")


if __name__ == "__main__":
    app()
