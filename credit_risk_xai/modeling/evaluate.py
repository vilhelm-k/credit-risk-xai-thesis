from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
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
