from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convenience wrapper for train/validation splitting with stratification."""
    stratify_arg = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )


def positive_class_weight(y: pd.Series) -> float:
    """Return heuristic class weight ratio (negative:positive)."""
    positives = (y == 1).sum()
    negatives = (y == 0).sum()
    if positives == 0:
        return 1.0
    return negatives / positives
