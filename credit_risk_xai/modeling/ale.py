"""
Accumulated Local Effects (ALE) computation utilities.

ALE plots show how features affect predictions on average, accounting for
feature correlations. Unlike Partial Dependence Plots (PDP), ALE is unbiased
when features are correlated.

Reference: Apley & Zhu (2020) "Visualizing the effects of predictor variables
in black box supervised learning models"

This module uses alibi for ALE computation - a well-validated implementation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from alibi.explainers import ALE


def compute_ale_for_feature(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature: str,
    feature_bounds: tuple[float, float] | None = None,
    min_bin_points: int = 4,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Compute ALE for a single continuous feature using alibi.

    Key implementation details:
    - alibi requires no NaN values, so we fill with median for grid computation
    - We preserve DataFrame dtypes (especially categorical) by slicing the original
      DataFrame rather than converting through numpy
    - This is critical for LightGBM which validates categorical feature consistency

    Parameters
    ----------
    predict_fn : callable
        Prediction function that takes a DataFrame and returns probabilities
        with shape (n_samples, n_classes). For binary classification, should
        return (n_samples, 2) where column 1 is the positive class probability.
    X : pd.DataFrame
        Input features (may contain NaN and categorical columns).
    feature : str
        Feature name to compute ALE for.
    feature_bounds : tuple[float, float] or None
        (lower, upper) bounds for filtering the output grid. If None, uses
        5th-95th percentile of the feature values.
    min_bin_points : int
        Minimum points per bin for ALE discretization. Higher values give
        smoother curves but may miss fine-grained patterns. Default: 4.

    Returns
    -------
    grid : np.ndarray
        Feature values at which ALE is computed (filtered to bounds).
    ale_values : np.ndarray
        ALE values at each grid point (for positive class in binary classification).
    bounds : tuple[float, float]
        The (lower, upper) bounds used for filtering.

    Examples
    --------
    >>> def predict_fn(X):
    ...     return model.predict_proba(X)
    >>> grid, ale, bounds = compute_ale_for_feature(predict_fn, X_val, 'age')
    >>> plt.plot(grid, ale)
    """
    # Extract single feature and fill NaN for alibi's grid computation
    X_single = X[[feature]].copy()
    median_val = X_single[feature].median()
    X_single_filled = X_single.fillna(median_val).astype(np.float64)

    # Store reference to original DataFrame (NOT numpy!) to preserve dtypes
    # This is critical for LightGBM with categorical features
    X_base_df = X.copy()
    n_total = len(X_base_df)

    def wrapped_predict(X_modified: np.ndarray) -> np.ndarray:
        """
        Wrapper that preserves DataFrame dtypes when alibi modifies feature values.

        Parameters
        ----------
        X_modified : np.ndarray
            Array with shape (n_samples, 1) containing modified feature values.

        Returns
        -------
        np.ndarray
            Prediction probabilities.
        """
        n_samples = X_modified.shape[0]

        # Slice original DataFrame to preserve all dtypes (especially categorical)
        if n_samples == n_total:
            X_for_pred = X_base_df.copy()
        else:
            X_for_pred = X_base_df.iloc[:n_samples].copy()

        # Update only the feature we're computing ALE for
        X_for_pred[feature] = X_modified[:, 0]

        return predict_fn(X_for_pred)

    # Compute ALE using alibi
    ale_explainer = ALE(wrapped_predict, feature_names=[feature])
    explanation = ale_explainer.explain(X_single_filled.values, min_bin_points=min_bin_points)

    # Extract results (index 0 since we only have 1 feature)
    feature_values = explanation.data["feature_values"][0]
    ale_values_raw = explanation.data["ale_values"][0]

    # For binary classification, select positive class (column 1)
    if ale_values_raw.ndim == 2:
        ale_values = ale_values_raw[:, 1]
    else:
        ale_values = ale_values_raw

    # Determine bounds
    if feature_bounds is not None:
        lower_bound, upper_bound = feature_bounds
    else:
        # Default: 5th-95th percentile
        feat_original = X[feature].dropna().values
        lower_bound = float(np.percentile(feat_original, 5))
        upper_bound = float(np.percentile(feat_original, 95))

    # Filter to bounds
    mask = (feature_values >= lower_bound) & (feature_values <= upper_bound)
    grid = feature_values[mask]
    ale_filtered = ale_values[mask]

    return grid, ale_filtered, (lower_bound, upper_bound)


def compute_ale_binary(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Compute ALE for a binary (0/1) feature.

    For binary features, ALE is simply the difference in mean prediction
    between feature=1 and feature=0 groups, centered at zero.

    Parameters
    ----------
    predict_fn : callable
        Prediction function that takes a DataFrame and returns probabilities
        with shape (n_samples, n_classes).
    X : pd.DataFrame
        Input features.
    feature : str
        Binary feature name to compute ALE for.

    Returns
    -------
    grid : np.ndarray
        Array [0, 1] representing the two feature values.
    ale_values : np.ndarray
        ALE values for feature=0 and feature=1, centered at zero.
    bounds : tuple[int, int]
        Always (0, 1) for binary features.

    Examples
    --------
    >>> grid, ale, bounds = compute_ale_binary(predict_fn, X_val, 'dividend_payer')
    >>> print(f"Effect of paying dividends: {ale[1] - ale[0]:.4f}")
    """
    X_0 = X.copy()
    X_1 = X.copy()

    # Preserve categorical dtype if present (critical for LightGBM)
    original_dtype = X[feature].dtype
    if hasattr(original_dtype, "categories"):
        # Categorical column - need to use category codes
        X_0[feature] = pd.Categorical.from_codes(
            [0] * len(X), categories=original_dtype.categories
        )
        X_1[feature] = pd.Categorical.from_codes(
            [1] * len(X), categories=original_dtype.categories
        )
    else:
        # Non-categorical - simple assignment
        X_0[feature] = 0
        X_1[feature] = 1

    pred_0 = predict_fn(X_0)[:, 1].mean()
    pred_1 = predict_fn(X_1)[:, 1].mean()

    # Center at zero (mean effect is 0)
    mean_pred = (pred_0 + pred_1) / 2
    ale_0 = pred_0 - mean_pred
    ale_1 = pred_1 - mean_pred

    return np.array([0, 1]), np.array([ale_0, ale_1]), (0, 1)


def compute_ale_for_features(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    features: list[str],
    feature_bounds: dict[str, tuple[float, float] | None] | None = None,
    min_bin_points: int = 4,
    verbose: bool = True,
) -> dict[str, dict[str, np.ndarray | tuple]]:
    """
    Compute ALE for multiple features.

    Parameters
    ----------
    predict_fn : callable
        Prediction function.
    X : pd.DataFrame
        Input features.
    features : list[str]
        List of feature names to compute ALE for.
    feature_bounds : dict or None
        Mapping from feature name to (lower, upper) bounds. Features with
        None bounds are treated as binary. If not provided, uses default
        percentile-based bounds.
    min_bin_points : int
        Minimum points per bin for continuous features.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    results : dict
        Nested dict with structure: {feature: {'grid': ..., 'ale': ..., 'bounds': ...}}
    """
    results = {}

    for i, feature in enumerate(features, 1):
        if verbose:
            print(f"[{i}/{len(features)}] {feature}...", end=" ")

        bounds = None
        if feature_bounds is not None:
            bounds = feature_bounds.get(feature)

        try:
            if bounds is None:
                # Binary feature
                grid, ale, feat_bounds = compute_ale_binary(predict_fn, X, feature)
                if verbose:
                    print("(binary)")
            else:
                # Continuous feature
                grid, ale, feat_bounds = compute_ale_for_feature(
                    predict_fn, X, feature, feature_bounds=bounds, min_bin_points=min_bin_points
                )
                if verbose:
                    print(f"(continuous, {len(grid)} points)")

            results[feature] = {
                "grid": grid,
                "ale": ale,
                "bounds": feat_bounds,
            }

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results[feature] = {"grid": np.array([]), "ale": np.array([]), "bounds": (0, 0)}

    return results
