from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def classification_summary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    labels: Optional[list[str]] = None,
) -> Dict[str, float]:
    """
    Compute key classification metrics given true labels and predicted probabilities.
    """
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics.update(
        {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "support_pos": report["1"]["support"],
            "support_neg": report["0"]["support"],
        }
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot ROC curve and return the matplotlib axis."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot Precision-Recall curve and return the matplotlib axis."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    return ax


def compute_ece(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 100,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average difference between predicted probabilities and
    actual outcomes across probability bins. Lower values indicate better
    calibration.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins for calibration curve. Default: 100.

    Returns
    -------
    ece : float
        Expected Calibration Error (0 = perfectly calibrated).

    Examples
    --------
    >>> ece = compute_ece(y_val, model.predict_proba(X_val)[:, 1])
    >>> print(f"ECE: {ece:.4f} - {'Well calibrated' if ece < 0.05 else 'Needs calibration'}")
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="quantile"
    )
    ece = float(np.mean(np.abs(fraction_of_positives - mean_predicted_value)))
    return ece


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 100,
) -> Dict[str, float]:
    """
    Compute calibration-related metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins for ECE computation.

    Returns
    -------
    metrics : dict
        Dictionary with 'brier_score' and 'ece' keys.
    """
    return {
        "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
        "ece": compute_ece(y_true, y_pred_proba, n_bins=n_bins),
    }


def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    metrics : dict
        Dictionary with 'AUC', 'PR-AUC', 'Brier Score', and 'ECE' keys.
    """
    return {
        "AUC": roc_auc_score(y_true, y_pred_proba),
        "PR-AUC": average_precision_score(y_true, y_pred_proba),
        "Brier Score": brier_score_loss(y_true, y_pred_proba),
        "ECE": compute_ece(y_true, y_pred_proba),
    }


def calculate_shap_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP importance for features.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features).
    feature_names : list[str]
        List of feature names corresponding to columns of shap_values.

    Returns
    -------
    importance : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns, sorted by
        descending importance (mean absolute SHAP value).
    """
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("importance", ascending=False)

    return importance
