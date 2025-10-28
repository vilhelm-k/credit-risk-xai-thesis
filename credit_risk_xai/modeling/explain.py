from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def get_feature_importance(
    model,
    X: pd.DataFrame,
    importance_type: Literal["split", "gain", "cover"] = "gain",
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract feature importance from a LightGBM model.

    Parameters
    ----------
    model : LightGBM model
        Fitted LightGBM model with feature_name_ and feature_importances_ attributes.
    X : pd.DataFrame
        Feature matrix (used to get column names if model doesn't have feature_name_).
    importance_type : str, default="gain"
        Type of feature importance to extract:
        - 'split': Number of times feature is used in a split
        - 'gain': Total gain of splits using the feature
        - 'cover': Total coverage of splits using the feature (XGBoost only)
    top_n : int, optional
        Return only top N features. If None, returns all features.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance'] sorted by importance descending.
    """
    # Get feature names
    if hasattr(model, "feature_name_"):
        feature_names = model.feature_name_
    else:
        feature_names = X.columns.tolist()

    # Get importance values
    if importance_type == "split":
        importances = model.feature_importances_
    elif importance_type == "gain":
        if hasattr(model, "booster_"):
            importances = model.booster_.feature_importance(importance_type="gain")
        else:
            raise ValueError("Model doesn't support gain importance type")
    elif importance_type == "cover":
        if hasattr(model, "booster_"):
            importances = model.booster_.feature_importance(importance_type="cover")
        else:
            raise ValueError("Model doesn't support cover importance type")
    else:
        raise ValueError(f"Invalid importance_type: {importance_type}. Choose from 'split', 'gain', or 'cover'.")

    # Create DataFrame
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    if top_n is not None:
        importance_df = importance_df.head(top_n)

    return importance_df.reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple = (10, 8),
    show: bool = True,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot feature importance as a horizontal bar chart.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with columns ['feature', 'importance'] from get_feature_importance().
    top_n : int, default=20
        Number of top features to display.
    figsize : tuple, default=(10, 8)
        Figure size (width, height).
    show : bool, default=True
        Whether to display the plot.
    output_path : Path, optional
        If provided, saves the plot to this path.
    """
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(plot_df["feature"], plot_df["importance"], color="#2E86AB")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def analyze_feature_groups(
    model,
    X: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    importance_type: Literal["split", "gain", "cover"] = "gain",
) -> pd.DataFrame:
    """
    Aggregate feature importance by predefined feature groups.

    Parameters
    ----------
    model : LightGBM model
        Fitted model.
    X : pd.DataFrame
        Feature matrix.
    feature_groups : dict
        Dictionary mapping group names to lists of feature names.
        Example: {"RATIO": ["ratio_x", "ratio_y"], "TREND": ["trend_a", "trend_b"]}
    importance_type : str, default="gain"
        Type of feature importance to use.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['group', 'total_importance', 'mean_importance', 'feature_count']
        sorted by total_importance descending.
    """
    # Get full feature importance
    importance_df = get_feature_importance(model, X, importance_type=importance_type)

    # Create mapping from feature to group
    feature_to_group = {}
    for group_name, features in feature_groups.items():
        for feature in features:
            feature_to_group[feature] = group_name

    # Add group column
    importance_df["group"] = importance_df["feature"].map(feature_to_group)

    # Handle features not in any group
    importance_df["group"] = importance_df["group"].fillna("OTHER")

    # Aggregate by group
    group_stats = (
        importance_df.groupby("group")["importance"]
        .agg(["sum", "mean", "count"])
        .reset_index()
        .rename(columns={"sum": "total_importance", "mean": "mean_importance", "count": "feature_count"})
        .sort_values("total_importance", ascending=False)
    )

    return group_stats


def get_shap_feature_importance(
    model, X: pd.DataFrame, max_samples: int = 2_000, top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute feature importance based on mean absolute SHAP values.

    Parameters
    ----------
    model : LightGBM/XGBoost model
        Fitted tree-based model.
    X : pd.DataFrame
        Feature matrix.
    max_samples : int, default=2_000
        Maximum samples to use for SHAP computation.
    top_n : int, optional
        Return only top N features. If None, returns all features.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
    """
    explainer, shap_values, X_sampled = compute_shap_values(model, X, max_samples=max_samples)

    # Handle binary classification (returns list of [negative_class, positive_class])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Calculate mean absolute SHAP values per feature
    shap_importance = pd.DataFrame(
        {"feature": X_sampled.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)

    if top_n is not None:
        shap_importance = shap_importance.head(top_n)

    return shap_importance.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Correlation Analysis Functions
# -----------------------------------------------------------------------------


def get_feature_correlations_by_source(
    X: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    method: Literal["pearson", "spearman"] = "pearson",
    min_periods: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Compute correlation matrices separately for each feature group.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    feature_groups : dict
        Dictionary mapping group names to lists of feature names.
        Example: {"BALANCE_SHEET": [...], "INCOME_STATEMENT": [...]}
    method : str, default="pearson"
        Correlation method: 'pearson' or 'spearman'.
    min_periods : int, default=10
        Minimum number of observations required per pair.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping group name to correlation matrix.
    """
    correlations = {}

    for group_name, features in feature_groups.items():
        # Filter to features that exist in X
        available_features = [f for f in features if f in X.columns]

        if len(available_features) < 2:
            continue

        # Compute correlation matrix
        corr_matrix = X[available_features].corr(method=method, min_periods=min_periods)
        correlations[group_name] = corr_matrix

    return correlations


def find_high_correlations(
    correlation_matrix: pd.DataFrame, threshold: float = 0.7, top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract all feature pairs with absolute correlation above a threshold.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix from get_feature_correlations_by_source().
    threshold : float, default=0.7
        Minimum |correlation| to report.
    top_n : int, optional
        Return only top N pairs by absolute correlation.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature_1', 'feature_2', 'correlation', 'abs_correlation']
        sorted by abs_correlation descending.
    """
    # Extract upper triangle (avoid duplicates and diagonal)
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Stack and filter
    corr_pairs = upper_tri.stack().reset_index()
    corr_pairs.columns = ["feature_1", "feature_2", "correlation"]
    corr_pairs["abs_correlation"] = corr_pairs["correlation"].abs()

    # Filter by threshold
    high_corrs = corr_pairs[corr_pairs["abs_correlation"] >= threshold].sort_values(
        "abs_correlation", ascending=False
    )

    if top_n is not None:
        high_corrs = high_corrs.head(top_n)

    return high_corrs.reset_index(drop=True)


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    figsize: tuple = (12, 10),
    cmap: str = "RdBu_r",
    annot: bool = False,
    mask_diagonal: bool = True,
    show: bool = True,
    output_path: Optional[Path] = None,
) -> None:
    """
    Visualize correlation matrix as a heatmap.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix to plot.
    title : str
        Plot title.
    figsize : tuple, default=(12, 10)
        Figure size (width, height).
    cmap : str, default='RdBu_r'
        Colormap (RdBu_r: red=negative, white=0, blue=positive).
    annot : bool, default=False
        Whether to annotate cells with correlation values.
    mask_diagonal : bool, default=True
        Hide diagonal (always 1.0).
    show : bool, default=True
        Whether to display the plot.
    output_path : Path, optional
        If provided, saves the plot to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for diagonal
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool)) if mask_diagonal else None

    # Create heatmap
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=annot,
        fmt=".2f" if annot else None,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def analyze_correlation_with_target(
    X: pd.DataFrame,
    y: pd.Series,
    method: Literal["pearson", "spearman"] = "pearson",
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute correlation between each feature and the target variable.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    method : str, default="pearson"
        Correlation method: 'pearson' (for continuous) or 'spearman' (for binary/ordinal).
    top_n : int, optional
        Return only top N most correlated features.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'correlation', 'abs_correlation']
        sorted by abs_correlation descending.
    """
    correlations = []

    for col in X.columns:
        # Compute correlation, handling missing values
        valid_mask = X[col].notna() & y.notna()
        if valid_mask.sum() < 10:  # Skip if too few observations
            continue

        if method == "pearson":
            corr = X.loc[valid_mask, col].corr(y[valid_mask])
        elif method == "spearman":
            corr = X.loc[valid_mask, col].corr(y[valid_mask], method="spearman")
        else:
            raise ValueError(f"Invalid method: {method}. Choose 'pearson' or 'spearman'.")

        correlations.append({"feature": col, "correlation": corr})

    # Create DataFrame
    corr_df = pd.DataFrame(correlations)
    corr_df["abs_correlation"] = corr_df["correlation"].abs()
    corr_df = corr_df.sort_values("abs_correlation", ascending=False)

    if top_n is not None:
        corr_df = corr_df.head(top_n)

    return corr_df.reset_index(drop=True)


def compare_correlation_vs_importance(
    X: pd.DataFrame, y: pd.Series, importance_df: pd.DataFrame, top_n: int = 30
) -> pd.DataFrame:
    """
    Merge correlation with target and feature importance to identify patterns.

    High importance + Low correlation → Nonlinear effects or interactions.
    High correlation + Low importance → Redundant with other features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    importance_df : pd.DataFrame
        Feature importance DataFrame from get_feature_importance().
        Must have columns ['feature', 'importance'].
    top_n : int, default=30
        Number of top features to analyze.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance', 'importance_rank',
        'target_corr', 'corr_rank', 'rank_diff'] sorted by rank_diff descending.
    """
    # Get target correlations
    target_corr = analyze_correlation_with_target(X, y, method="pearson")

    # Merge with importance
    comparison = importance_df.merge(target_corr[["feature", "correlation"]], on="feature", how="inner")
    comparison.rename(columns={"correlation": "target_corr"}, inplace=True)

    # Compute ranks
    comparison["importance_rank"] = comparison["importance"].rank(ascending=False)
    comparison["corr_rank"] = comparison["target_corr"].abs().rank(ascending=False)
    comparison["rank_diff"] = abs(comparison["importance_rank"] - comparison["corr_rank"])

    # Sort by rank difference (most interesting features first)
    comparison = comparison.sort_values("rank_diff", ascending=False).head(top_n)

    return comparison.reset_index(drop=True)


def summarize_within_group_correlations(
    X: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    """
    Compute summary statistics of correlations within each feature group.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    feature_groups : dict
        Dictionary mapping group names to lists of feature names.
    method : str, default="pearson"
        Correlation method.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['group', 'mean_corr', 'median_corr', 'max_corr',
        'min_corr', 'high_corr_pairs'] where high_corr_pairs = count of |r| > 0.7.
    """
    group_summaries = []

    corr_by_source = get_feature_correlations_by_source(X, feature_groups, method=method)

    for group_name, corr_matrix in corr_by_source.items():
        # Extract upper triangle (exclude diagonal)
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_values = upper_tri.stack().values

        if len(corr_values) == 0:
            continue

        # Compute statistics
        group_summaries.append(
            {
                "group": group_name,
                "feature_count": len(corr_matrix),
                "mean_corr": corr_values.mean(),
                "median_corr": np.median(corr_values),
                "max_corr": corr_values.max(),
                "min_corr": corr_values.min(),
                "high_corr_pairs": (np.abs(corr_values) > 0.7).sum(),
            }
        )

    summary_df = pd.DataFrame(group_summaries).sort_values("mean_corr", ascending=False)

    return summary_df.reset_index(drop=True)
