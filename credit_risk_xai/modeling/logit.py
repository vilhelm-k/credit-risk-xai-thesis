"""
Logistic regression model with domain-informed preprocessing for credit risk.

This module provides a complete logistic regression pipeline optimized for
credit risk modeling, including:
- Domain-informed clipping for features with known economic bounds
- Winsorization for remaining outliers
- Robust scaling (IQR-based)
- Weight of Evidence (WoE) encoding for categorical features
- Statsmodels GLM with HC3 robust standard errors

The preprocessing is designed to keep the model interpretable while handling
the extreme values common in financial data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM


# =============================================================================
# Domain-Informed Clipping Bounds
# =============================================================================
# These bounds reflect economic thresholds beyond which precise values don't
# add predictive information for default risk.
#
# interest_coverage: EBIT / Interest Expense
#   - Values < -5: Severely loss-making (no meaningful distinction below)
#   - Values > 20: Very comfortable coverage (no meaningful distinction above)

DOMAIN_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "interest_coverage": (-5.0, 20.0),
}


# =============================================================================
# Preprocessing Components
# =============================================================================


class DomainClipper(BaseEstimator, TransformerMixin):
    """
    Apply domain-informed clipping to specific features.

    Unlike winsorization (which learns bounds from data), domain clipping uses
    fixed bounds based on economic interpretation. This ensures consistent
    treatment across train/validation/test sets.

    Parameters
    ----------
    clip_bounds : dict[str, tuple[float, float]]
        Mapping from feature name to (lower, upper) bounds.
    feature_names : list[str]
        Names of features in the input array (in order).
    """

    def __init__(
        self,
        clip_bounds: dict[str, tuple[float, float]] | None = None,
        feature_names: list[str] | None = None,
    ):
        self.clip_bounds = clip_bounds or DOMAIN_CLIP_BOUNDS
        self.feature_names = feature_names

    def fit(self, X: np.ndarray, y: Any = None) -> "DomainClipper":
        """No fitting required - bounds are fixed."""
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Apply domain clipping to relevant features."""
        if self.feature_names is None:
            return X

        # Handle both DataFrame and array inputs
        if isinstance(X, pd.DataFrame):
            X_clipped = X.copy()
            for feat_name, (lower, upper) in self.clip_bounds.items():
                if feat_name in X_clipped.columns:
                    X_clipped[feat_name] = X_clipped[feat_name].clip(lower, upper)
            return X_clipped
        else:
            X_clipped = X.copy()
            for feat_name, (lower, upper) in self.clip_bounds.items():
                if feat_name in self.feature_names:
                    idx = self.feature_names.index(feat_name)
                    X_clipped[:, idx] = np.clip(X_clipped[:, idx], lower, upper)
            return X_clipped

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Clip values at specified percentiles to handle outliers.

    This is applied after domain clipping to handle remaining outliers
    in features without explicit domain bounds.

    Parameters
    ----------
    lower_percentile : float
        Lower percentile for clipping (default: 1).
    upper_percentile : float
        Upper percentile for clipping (default: 99).
    """

    def __init__(self, lower_percentile: float = 1, upper_percentile: float = 99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "Winsorizer":
        """Learn percentile bounds from training data."""
        self.lower_bounds_ = np.nanpercentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.nanpercentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned percentile clipping."""
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("Winsorizer must be fitted before transform")
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


def woe_encode_feature(
    train_series: pd.Series,
    y_train: pd.Series,
    smoothing: float = 0.5,
) -> tuple[dict[str, float], float]:
    """
    Compute Weight of Evidence (WoE) encoding for a categorical feature.

    WoE measures the predictive power of each category by comparing the
    distribution of events (defaults) vs non-events within each category.

    WoE_i = ln(Distribution of Events_i / Distribution of Non-Events_i)
          = ln((n_events_i / total_events) / (n_non_events_i / total_non_events))

    Positive WoE indicates higher-than-average default rate for that category.
    Negative WoE indicates lower-than-average default rate.

    Parameters
    ----------
    train_series : pd.Series
        Training set categorical values.
    y_train : pd.Series
        Training set target values (0/1 for non-event/event).
    smoothing : float
        Laplace smoothing parameter to avoid division by zero and stabilize
        estimates for rare categories (default: 0.5).

    Returns
    -------
    woe_map : dict[str, float]
        Mapping from category to WoE value.
    default_woe : float
        Default WoE for unseen categories (0.0, indicating neutral).
    """
    train_df = pd.DataFrame({
        "cat": train_series.astype(str),
        "target": y_train.values
    })

    total_events = train_df["target"].sum()
    total_non_events = len(train_df) - total_events

    agg = train_df.groupby("cat")["target"].agg(["sum", "count"])
    agg.columns = ["events", "total"]
    agg["non_events"] = agg["total"] - agg["events"]

    # Apply Laplace smoothing and compute WoE
    dist_events = (agg["events"] + smoothing) / (total_events + smoothing * len(agg))
    dist_non_events = (agg["non_events"] + smoothing) / (total_non_events + smoothing * len(agg))

    woe = np.log(dist_events / dist_non_events)
    woe_map = woe.to_dict()

    # Default WoE for unseen categories is 0 (neutral, equivalent to population average)
    default_woe = 0.0

    return woe_map, default_woe


# =============================================================================
# Main Preprocessor Class
# =============================================================================


class LogitPreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline for credit risk logistic regression.

    Pipeline steps:
    1. Domain clipping (fixed bounds for specific features)
    2. Median imputation
    3. Winsorization (P1-P99)
    4. Robust scaling (IQR-based)
    5. Weight of Evidence (WoE) encoding for categoricals

    Parameters
    ----------
    lower_percentile : float
        Lower percentile for winsorization (default: 1).
    upper_percentile : float
        Upper percentile for winsorization (default: 99).
    scaling_quantile_range : tuple[float, float]
        Quantile range for RobustScaler (default: (5.0, 95.0)).
    woe_smoothing : float
        Laplace smoothing parameter for WoE encoding (default: 0.5).
    domain_clip_bounds : dict[str, tuple[float, float]] | None
        Custom domain clipping bounds (default: use DOMAIN_CLIP_BOUNDS).
    """

    def __init__(
        self,
        lower_percentile: float = 1,
        upper_percentile: float = 99,
        scaling_quantile_range: tuple[float, float] = (5.0, 95.0),
        woe_smoothing: float = 0.5,
        domain_clip_bounds: dict[str, tuple[float, float]] | None = None,
    ):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.scaling_quantile_range = scaling_quantile_range
        self.woe_smoothing = woe_smoothing
        self.domain_clip_bounds = domain_clip_bounds or DOMAIN_CLIP_BOUNDS

        # Fitted attributes
        self.numeric_features_: list[str] | None = None
        self.categorical_features_: list[str] | None = None
        self.numeric_pipeline_: Pipeline | None = None
        self.cat_encodings_: dict[str, dict[str, Any]] | None = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "LogitPreprocessor":
        """
        Fit the preprocessing pipeline on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series or np.ndarray
            Training target (required for target encoding).

        Returns
        -------
        self
        """
        if y is None:
            raise ValueError("y is required for fitting (needed for target encoding)")

        # Identify feature types
        self.numeric_features_ = [
            c for c in X.columns if X[c].dtype.name != "category"
        ]
        self.categorical_features_ = [
            c for c in X.columns if X[c].dtype.name == "category"
        ]

        # Build and fit numeric pipeline
        self.numeric_pipeline_ = Pipeline(
            [
                (
                    "domain_clipper",
                    DomainClipper(
                        clip_bounds=self.domain_clip_bounds,
                        feature_names=self.numeric_features_,
                    ),
                ),
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "winsorizer",
                    Winsorizer(
                        lower_percentile=self.lower_percentile,
                        upper_percentile=self.upper_percentile,
                    ),
                ),
                ("scaler", RobustScaler(quantile_range=self.scaling_quantile_range)),
            ]
        )
        self.numeric_pipeline_.fit(X[self.numeric_features_])

        # Fit WoE encoding for categoricals
        if self.categorical_features_:
            y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
            self.cat_encodings_ = {}

            for cat_col in self.categorical_features_:
                woe_map, default_woe = woe_encode_feature(
                    X[cat_col],
                    y_series,
                    smoothing=self.woe_smoothing,
                )
                self.cat_encodings_[cat_col] = {"map": woe_map, "default": default_woe}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.

        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed features.
        """
        if self.numeric_pipeline_ is None:
            raise ValueError("Preprocessor must be fitted before transform")

        # Transform numeric features
        X_numeric = self.numeric_pipeline_.transform(X[self.numeric_features_])
        X_numeric_df = pd.DataFrame(
            X_numeric, columns=self.numeric_features_, index=X.index
        )

        # Transform categorical features
        if self.categorical_features_ and self.cat_encodings_:
            cat_list = []
            for cat_col in self.categorical_features_:
                enc_map = self.cat_encodings_[cat_col]["map"]
                default = self.cat_encodings_[cat_col]["default"]
                encoded = X[cat_col].astype(str).map(enc_map).fillna(default)
                cat_list.append(encoded.values)

            X_cat = pd.DataFrame(
                np.column_stack(cat_list),
                columns=self.categorical_features_,
                index=X.index,
            )
            X_transformed = pd.concat([X_numeric_df, X_cat], axis=1)
        else:
            X_transformed = X_numeric_df

        return X_transformed.astype(float)

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> list[str]:
        """Get output feature names after transformation."""
        if self.numeric_features_ is None:
            raise ValueError("Preprocessor must be fitted first")
        return self.numeric_features_ + (self.categorical_features_ or [])


# =============================================================================
# Main Model Class
# =============================================================================


class CreditRiskLogit(BaseEstimator):
    """
    Credit risk logistic regression with domain-informed preprocessing.

    This class wraps statsmodels GLM with a sklearn-compatible interface,
    including SHAP LinearExplainer compatibility via coef_ and intercept_.

    Parameters
    ----------
    preprocessor_kwargs : dict | None
        Keyword arguments passed to LogitPreprocessor.
    cov_type : str
        Covariance type for robust standard errors (default: "HC3").

    Attributes
    ----------
    preprocessor_ : LogitPreprocessor
        Fitted preprocessor.
    glm_result_ : statsmodels GLMResults
        Fitted GLM result object.
    coef_ : np.ndarray
        Model coefficients (shape: (1, n_features)).
    intercept_ : np.ndarray
        Model intercept (shape: (1,)).

    Examples
    --------
    >>> model = CreditRiskLogit()
    >>> model.fit(X_train, y_train)
    >>> proba = model.predict_proba(X_val)[:, 1]
    >>> # For SHAP
    >>> explainer = shap.LinearExplainer(model, X_train_processed)
    """

    def __init__(
        self,
        preprocessor_kwargs: dict[str, Any] | None = None,
        cov_type: str = "HC3",
    ):
        self.preprocessor_kwargs = preprocessor_kwargs or {}
        self.cov_type = cov_type

        # Fitted attributes
        self.preprocessor_: LogitPreprocessor | None = None
        self.glm_result_: Any | None = None  # statsmodels GLMResults
        self._coef: np.ndarray | None = None
        self._intercept: np.ndarray | None = None

    @property
    def coef_(self) -> np.ndarray:
        """Model coefficients (SHAP compatibility)."""
        if self._coef is None:
            raise ValueError("Model must be fitted first")
        return self._coef

    @property
    def intercept_(self) -> np.ndarray:
        """Model intercept (SHAP compatibility)."""
        if self._intercept is None:
            raise ValueError("Model must be fitted first")
        return self._intercept

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> "CreditRiskLogit":
        """
        Fit the preprocessing pipeline and logistic regression model.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series or np.ndarray
            Training target.

        Returns
        -------
        self
        """
        # Fit preprocessor
        self.preprocessor_ = LogitPreprocessor(**self.preprocessor_kwargs)
        X_processed = self.preprocessor_.fit_transform(X, y)

        # Fit GLM
        y_np = np.asarray(y, dtype=np.float64)
        X_with_const = sm.add_constant(X_processed)

        glm_model = GLM(y_np, X_with_const, family=Binomial())
        self.glm_result_ = glm_model.fit(cov_type=self.cov_type)

        # Store coefficients for SHAP
        self._coef = self.glm_result_.params.values[1:].reshape(1, -1)
        self._intercept = np.array([self.glm_result_.params.values[0]])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw features to preprocessed features.

        Parameters
        ----------
        X : pd.DataFrame
            Raw features.

        Returns
        -------
        X_processed : pd.DataFrame
            Preprocessed features.
        """
        if self.preprocessor_ is None:
            raise ValueError("Model must be fitted first")
        return self.preprocessor_.transform(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features (raw DataFrame or preprocessed array).

        Returns
        -------
        proba : np.ndarray
            Class probabilities, shape (n_samples, 2).
        """
        if self.glm_result_ is None or self.preprocessor_ is None:
            raise ValueError("Model must be fitted first")

        # Handle both raw DataFrames and preprocessed arrays
        if isinstance(X, pd.DataFrame):
            # Check if this is raw data (has original feature names)
            if set(self.preprocessor_.numeric_features_ or []).issubset(X.columns):
                X_processed = self.transform(X)
                X_arr = X_processed.values
            else:
                X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        # Add constant and predict
        ones = np.ones((X_arr.shape[0], 1), dtype=np.float64)
        X_with_const = np.hstack([ones, X_arr])

        z = X_with_const @ self.glm_result_.params.values
        probs = 1.0 / (1.0 + np.exp(-z))

        return np.column_stack([1 - probs, probs])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features.

        Returns
        -------
        labels : np.ndarray
            Predicted class labels (0 or 1).
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def summary(self) -> pd.DataFrame:
        """
        Get coefficient summary with robust standard errors.

        Returns
        -------
        summary_df : pd.DataFrame
            DataFrame with columns: feature, coef, robust_se, z_score, p_value.
        """
        if self.glm_result_ is None or self.preprocessor_ is None:
            raise ValueError("Model must be fitted first")

        feature_names = ["intercept"] + self.preprocessor_.get_feature_names_out()

        return pd.DataFrame(
            {
                "feature": feature_names,
                "coef": self.glm_result_.params.values,
                "robust_se": self.glm_result_.bse.values,
                "z_score": self.glm_result_.tvalues.values,
                "p_value": self.glm_result_.pvalues.values,
                "abs_coef": np.abs(self.glm_result_.params.values),
            }
        ).sort_values("p_value")

    @property
    def converged(self) -> bool:
        """Check if the GLM optimization converged."""
        if self.glm_result_ is None:
            raise ValueError("Model must be fitted first")
        return self.glm_result_.converged
