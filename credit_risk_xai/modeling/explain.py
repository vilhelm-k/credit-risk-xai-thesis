from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 2_000):
    """
    Compute SHAP values for a fitted tree-based model.

    Parameters
    ----------
    model : LightGBM/XGBoost model with `predict` interface.
    X : pd.DataFrame
        Feature matrix.
    max_samples : int
        Optional subsample to control runtime for large datasets.
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("shap is required for explanation utilities. Install with `pip install shap`.") from exc

    if max_samples and len(X) > max_samples:
        X_sampled = X.sample(max_samples, random_state=42)
    else:
        X_sampled = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sampled)
    return explainer, shap_values, X_sampled


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    max_samples: int = 2_000,
    show: bool = False,
    output_path: Optional[Path] = None,
) -> None:
    """
    Generate a SHAP summary plot and optionally save to disk.
    """
    import shap

    explainer, shap_values, X_sampled = compute_shap_values(model, X, max_samples=max_samples)

    shap.summary_plot(shap_values, X_sampled, show=show)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
