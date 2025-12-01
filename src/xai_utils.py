"""
XAI Utilities for Credit Risk Analysis

This module provides reusable functions for explainable AI analysis,
including SHAP computation, ALE plots, and visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats.mstats import winsorize


# ==============================================================================
# PREPROCESSING UTILITIES
# ==============================================================================

def winsorize_features(X, limits=(0.01, 0.01), categorical_features=None, verbose=True):
    """
    Winsorize continuous features at specified percentiles.

    Categorical features are excluded from winsorization.
    Extreme values beyond percentile limits are capped to those thresholds.

    Parameters:
    -----------
    X : pd.DataFrame
        Features to winsorize
    limits : tuple
        (lower, upper) proportions to winsorize (default: 1st and 99th percentile)
    categorical_features : list
        List of categorical feature names to exclude from winsorization
    verbose : bool
        Print progress messages

    Returns:
    --------
    pd.DataFrame
        Winsorized features
    """
    X_wins = X.copy()

    if categorical_features is None:
        categorical_features = []

    # Identify categorical columns (category dtype)
    cat_cols = X.select_dtypes(include=['category']).columns.tolist()
    categorical_features = list(set(categorical_features + cat_cols))

    continuous_cols = [col for col in X.columns if col not in categorical_features]

    if verbose:
        print(f"  Winsorizing {len(continuous_cols)} continuous features at "
              f"{limits[0]*100:.0f}th/{(1-limits[1])*100:.0f}th percentiles")
        print(f"  Excluding {len(categorical_features)} categorical features: {categorical_features}")

    for col in continuous_cols:
        # Winsorize only non-NaN values
        mask = X_wins[col].notna()
        if mask.sum() > 0:
            # Convert to numpy array to handle pandas nullable dtypes (Int16, etc.)
            values = X_wins.loc[mask, col].values.astype(float)
            winsorized_values = winsorize(values, limits=limits)
            X_wins.loc[mask, col] = winsorized_values

    return X_wins


def target_encode_categorical(X_train, X_val, y_train, categorical_cols, smoothing=10.0):
    """
    Target encoding for categorical variables using smoothed mean of target.

    Formula: encoded_value = (n_cat * mean_cat + smoothing * global_mean) / (n_cat + smoothing)

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features
    y_train : pd.Series
        Training target
    categorical_cols : list
        Categorical column names
    smoothing : float
        Smoothing factor (higher = more regularization toward global mean)

    Returns:
    --------
    tuple : (X_train_encoded, X_val_encoded, encoding_maps)
    """
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()

    global_mean = y_train.mean()
    encoding_maps = {}

    print(f"\n  Target encoding {len(categorical_cols)} categorical features:")
    print(f"  Global default rate: {global_mean:.4f}")
    print(f"  Smoothing parameter: {smoothing}")

    for col in categorical_cols:
        if col not in X_train.columns:
            continue

        # Compute smoothed mean for each category
        category_stats = pd.DataFrame({
            'count': X_train.groupby(col, observed=True).size(),
            'sum': y_train.groupby(X_train[col], observed=True).sum()
        })

        category_stats['smoothed_mean'] = (
            (category_stats['sum'] + smoothing * global_mean) /
            (category_stats['count'] + smoothing)
        )

        encoding_map = category_stats['smoothed_mean'].to_dict()
        encoding_maps[col] = encoding_map

        # Apply encoding
        X_train_enc[col] = X_train[col].map(encoding_map).fillna(global_mean)
        X_val_enc[col] = X_val[col].map(encoding_map).fillna(global_mean)

        n_categories = len(encoding_map)
        min_enc = category_stats['smoothed_mean'].min()
        max_enc = category_stats['smoothed_mean'].max()

        print(f"    {col}: {n_categories} categories → range [{min_enc:.4f}, {max_enc:.4f}]")

    return X_train_enc, X_val_enc, encoding_maps


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def compute_ece(y_true, y_pred_proba, n_bins=100):
    """Compute Expected Calibration Error"""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='quantile'
    )
    ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    return ece


def compute_performance_metrics(y_true, y_pred_proba):
    """
    Compute comprehensive performance metrics.

    Returns:
    --------
    dict : Dictionary with AUC, PR-AUC, Brier Score, ECE
    """
    return {
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'PR-AUC': average_precision_score(y_true, y_pred_proba),
        'Brier Score': brier_score_loss(y_true, y_pred_proba),
        'ECE': compute_ece(y_true, y_pred_proba)
    }


# ==============================================================================
# SHAP UTILITIES
# ==============================================================================

def calculate_shap_importance(shap_values, feature_names):
    """
    Calculate mean absolute SHAP importance for features.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values (n_samples, n_features)
    feature_names : list
        Feature names

    Returns:
    --------
    pd.DataFrame : Features sorted by importance
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    return importance


def compute_shap_interaction_summary(shap_interaction_values, feature_names, top_k=10):
    """
    Summarize SHAP interaction strengths for top features.

    Parameters:
    -----------
    shap_interaction_values : np.ndarray
        SHAP interaction values (n_samples, n_features, n_features)
    feature_names : list
        Feature names
    top_k : int
        Number of top features to analyze

    Returns:
    --------
    pd.DataFrame : Interaction summary for top features
    """
    # Mean absolute interaction strength
    interaction_matrix = np.abs(shap_interaction_values).mean(axis=0)
    np.fill_diagonal(interaction_matrix, 0)  # Exclude self-interactions

    # For each feature, find strongest interaction partner
    interaction_summary = []
    for feat_idx in range(min(top_k, len(feature_names))):
        feature = feature_names[feat_idx]

        # Find strongest interaction
        strongest_idx = np.argmax(interaction_matrix[feat_idx, :])
        strongest_strength = interaction_matrix[feat_idx, strongest_idx]

        interaction_summary.append({
            'feature': feature,
            'top_interacting_feature': feature_names[strongest_idx],
            'interaction_strength': strongest_strength
        })

    return pd.DataFrame(interaction_summary).sort_values(
        'interaction_strength', ascending=False
    )


# ==============================================================================
# ALE (ACCUMULATED LOCAL EFFECTS) PLOTS - Using alibi
# ==============================================================================

from alibi.explainers import ALE


def compute_ale_1d(model, X, feature, grid_size=50, predict_fn=None, percentile_range=(5, 95),
                   target_class=1, min_bin_points=4):
    """
    Compute 1D ALE plot for a given feature using alibi.

    This is a wrapper around alibi's ALE implementation that maintains
    backward compatibility with the original API.

    Parameters:
    -----------
    model : trained model with predict_proba method
    X : pd.DataFrame
        Input features
    feature : str
        Feature name to compute ALE for
    grid_size : int
        Number of grid points (not directly used by alibi, kept for compatibility)
    predict_fn : callable
        Custom prediction function. Should return array of shape (n_samples,) or (n_samples, n_classes).
        For classification, alibi expects predict_proba output.
    percentile_range : tuple
        (lower, upper) percentiles to filter the output (default: 5th-95th)
    target_class : int
        For classification, which class to return ALE for (default: 1 for binary positive class)
    min_bin_points : int
        Minimum number of points per bin for alibi (default: 4)

    Returns:
    --------
    tuple : (grid_centers, ale_values, counts, percentile_bounds)
        - grid_centers: Feature values at which ALE is computed
        - ale_values: ALE effect values (centered)
        - counts: Approximate bin counts (based on deciles)
        - percentile_bounds: (lower_bound, upper_bound) for the feature
    """
    # Get feature index
    feature_names = X.columns.tolist()
    feature_idx = feature_names.index(feature)

    # Set up prediction function
    # alibi expects predict_fn to return (n_samples, n_classes) for classification
    # We wrap user-provided functions to ensure correct format
    if predict_fn is None:
        # Default: use model's predict_proba
        alibi_predict_fn = lambda x: model.predict_proba(x)
    else:
        # Check if user's predict_fn returns 1D or 2D
        # We'll wrap it to always return 2D for alibi
        def alibi_predict_fn(x):
            result = predict_fn(x)
            if result.ndim == 1:
                # User returned 1D (e.g., just positive class probs)
                # Convert to 2D: [1-p, p] format for binary classification
                return np.column_stack([1 - result, result])
            return result

    # Create alibi ALE explainer
    ale_explainer = ALE(alibi_predict_fn, feature_names=feature_names)

    # Compute ALE - alibi expects numpy array
    X_array = X.values.astype(np.float64)
    explanation = ale_explainer.explain(X_array, min_bin_points=min_bin_points)

    # Extract results for this feature
    feature_values = explanation.data['feature_values'][feature_idx]
    ale_values_raw = explanation.data['ale_values'][feature_idx]

    # For classification, ale_values has shape (n_bins, n_classes)
    # Select the target class (default: positive class = 1)
    if ale_values_raw.ndim == 2:
        ale_values = ale_values_raw[:, target_class]
    else:
        ale_values = ale_values_raw

    # Apply percentile filtering
    p_lower, p_upper = percentile_range
    feat_original = X[feature].dropna().values
    lower_bound = np.percentile(feat_original, p_lower)
    upper_bound = np.percentile(feat_original, p_upper)

    # Filter to percentile range
    mask = (feature_values >= lower_bound) & (feature_values <= upper_bound)
    grid_centers = feature_values[mask]
    ale_filtered = ale_values[mask]

    # Approximate counts from deciles (alibi doesn't return exact counts)
    # Use feature_deciles to estimate distribution
    deciles = explanation.data['feature_deciles'][feature_idx]
    counts = np.ones(len(grid_centers)) * (len(X) / len(deciles))  # Approximate

    return grid_centers, ale_filtered, counts, (lower_bound, upper_bound)


def compute_ale_for_models(models_dict, X, features, percentile_range=(5, 95), min_bin_points=4):
    """
    Compute ALE for multiple models on the same features.

    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: (model, predict_fn)}
        where predict_fn takes X and returns probabilities
    X : pd.DataFrame
        Input features (same for all models, in raw/untransformed space)
    features : list
        List of feature names to compute ALE for
    percentile_range : tuple
        Percentile range for filtering
    min_bin_points : int
        Minimum points per bin

    Returns:
    --------
    dict : {feature: {model_name: {'grid': array, 'ale': array, 'bounds': tuple}}}
    """
    ale_results = {}

    for feature in features:
        ale_results[feature] = {}

        for model_name, (model, predict_fn) in models_dict.items():
            try:
                grid, ale, counts, bounds = compute_ale_1d(
                    model=model,
                    X=X,
                    feature=feature,
                    predict_fn=predict_fn,
                    percentile_range=percentile_range,
                    min_bin_points=min_bin_points
                )

                ale_results[feature][model_name] = {
                    'grid': grid,
                    'ale': ale,
                    'counts': counts,
                    'bounds': bounds
                }
            except Exception as e:
                print(f"  Warning: Failed to compute ALE for {feature} with {model_name}: {e}")
                continue

    return ale_results


# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def plot_calibration_comparison(y_true, y_pred_proba_dict, n_bins=100, figsize=(14, 5)):
    """
    Plot calibration curves for multiple models side by side.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba_dict : dict
        Dictionary of {model_name: predicted_probabilities}
    n_bins : int
        Number of bins for calibration curve
    figsize : tuple
        Figure size
    """
    n_models = len(y_pred_proba_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, y_pred_proba) in zip(axes, y_pred_proba_dict.items()):
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy='quantile'
        )

        ece = compute_ece(y_true, y_pred_proba, n_bins=n_bins)

        ax.plot(mean_pred, fraction_pos, 's-', label=model_name, linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
        ax.set_xlabel('Mean predicted probability', fontsize=12)
        ax.set_ylabel('Fraction of positives', fontsize=12)
        ax.set_title(f'{model_name} Calibration (ECE={ece:.4f})', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance_comparison(importance_df, top_k=15, figsize=(12, 8)):
    """
    Plot feature importance comparison between models.

    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with columns: feature, model1_importance, model2_importance, ...
    top_k : int
        Number of top features to display
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get top features by first importance column
    importance_cols = [col for col in importance_df.columns if col != 'feature']
    top_features = importance_df.nlargest(top_k, importance_cols[0])

    y_pos = np.arange(len(top_features))

    # Plot primary importance as bars
    ax.barh(y_pos, top_features[importance_cols[0]],
            color='steelblue', alpha=0.7, label=importance_cols[0], height=0.6)

    # Overlay additional importances as markers
    colors = ['orange', 'green', 'red']
    markers = ['D', 'o', 's']

    for i, col in enumerate(importance_cols[1:]):
        ax.scatter(top_features[col], y_pos,
                   color=colors[i % len(colors)],
                   s=100,
                   marker=markers[i % len(markers)],
                   label=col,
                   zorder=3,
                   edgecolors='black',
                   linewidths=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=11)
    ax.set_xlabel('Importance', fontsize=13)
    ax.set_title('Feature Importance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ale_comparison(ale_results, figsize=(20, 12), n_cols=5):
    """
    Plot ALE curves for multiple features and models.

    Parameters:
    -----------
    ale_results : dict
        Dictionary of {feature: {grid_model1, ale_model1, ...}}
    figsize : tuple
        Figure size
    n_cols : int
        Number of columns in subplot grid
    """
    n_plots = len(ale_results)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, (feature, data) in enumerate(ale_results.items()):
        ax = axes[idx]

        if len(data['grid_lgbm']) == 0:
            ax.text(0.5, 0.5, f"{feature}\n(no variation)",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature, fontsize=9, fontweight='bold')
            continue

        is_binary = len(data['grid_lgbm']) <= 3

        # Plot LightGBM
        ax.plot(data['grid_lgbm'], data['ale_lgbm'],
                label='LightGBM', linewidth=2.5, color='steelblue', alpha=0.9,
                marker='o' if is_binary else None, markersize=8)

        # Plot Logistic
        if 'grid_logit' in data and 'ale_logit' in data:
            ax.plot(data['grid_logit'], data['ale_logit'],
                    label='Logistic', linewidth=2.5, color='orange', alpha=0.9,
                    linestyle='--', marker='s' if is_binary else None, markersize=8)

        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Set limits
        if not is_binary:
            all_x = np.concatenate([data['grid_lgbm'], data.get('grid_logit', [])])
            x_padding = (all_x.max() - all_x.min()) * 0.05
            ax.set_xlim(all_x.min() - x_padding, all_x.max() + x_padding)

        lower_bound, upper_bound = data['bounds']
        if is_binary:
            ax.set_title(f"{feature}*\n(Binary: 0 vs 1)", fontsize=9, fontweight='bold')
            ax.set_xticks(data['grid_lgbm'])
            ax.set_xticklabels([f"{int(v)}" for v in data['grid_lgbm']])
        else:
            ax.set_title(f"{feature}\n[{lower_bound:.2f}, {upper_bound:.2f}]",
                         fontsize=9, fontweight='bold')

        ax.set_ylabel('ALE (Δ PD)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def plot_interaction_heatmap(interaction_matrix, feature_names, top_k=15, figsize=(12, 10)):
    """
    Plot SHAP interaction heatmap for top features.

    Parameters:
    -----------
    interaction_matrix : np.ndarray
        Interaction matrix (n_features, n_features)
    feature_names : list
        Feature names
    top_k : int
        Number of top features to display
    figsize : tuple
        Figure size
    """
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Extract submatrix for top features
    top_indices = list(range(min(top_k, len(feature_names))))
    interaction_submatrix = interaction_matrix[np.ix_(top_indices, top_indices)]
    top_feature_names = [feature_names[i] for i in top_indices]

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'interaction', ['white', 'lightblue', 'steelblue', 'darkblue']
    )

    im = ax.imshow(interaction_submatrix, cmap=cmap, aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(top_feature_names)))
    ax.set_yticks(np.arange(len(top_feature_names)))
    ax.set_xticklabels(top_feature_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(top_feature_names, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean |SHAP interaction|', rotation=270, labelpad=20, fontsize=11)

    # Add title
    ax.set_title(f'SHAP Interaction Heatmap (Top {len(top_feature_names)} Features)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.set_xticks(np.arange(len(top_feature_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(top_feature_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

    plt.tight_layout()
    return fig
