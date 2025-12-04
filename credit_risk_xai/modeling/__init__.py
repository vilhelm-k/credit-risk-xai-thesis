"""Model training, evaluation, and explainability utilities."""

from .train import train_lightgbm, run_lightgbm_training  # noqa: F401
from .evaluate import (  # noqa: F401
    classification_summary,
    plot_roc_curve,
    plot_pr_curve,
    compute_ece,
    compute_calibration_metrics,
)
from .logit import (  # noqa: F401
    CreditRiskLogit,
    LogitPreprocessor,
    DOMAIN_CLIP_BOUNDS,
)
from .ale import (  # noqa: F401
    compute_ale_for_feature,
    compute_ale_binary,
    compute_ale_for_features,
)
from .explain import (  # noqa: F401
    compute_shap_values,
    plot_shap_summary,
    get_feature_importance,
    plot_feature_importance,
    analyze_feature_groups,
    get_shap_feature_importance,
    get_feature_correlations_by_source,
    find_high_correlations,
    plot_correlation_heatmap,
    analyze_correlation_with_target,
    compare_correlation_vs_importance,
    summarize_within_group_correlations,
)
from .utils import split_train_validation  # noqa: F401
