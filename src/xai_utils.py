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
# ALE (ACCUMULATED LOCAL EFFECTS) PLOTS
# ==============================================================================

def compute_ale_1d(model, X, feature, grid_size=50, predict_fn=None, percentile_range=(5, 95)):
    """
    Compute 1D ALE plot for a given feature.

    Parameters:
    -----------
    model : trained model with predict_proba method
    X : pd.DataFrame
        Input features
    feature : str
        Feature name to compute ALE for
    grid_size : int
        Number of grid points (ignored for binary)
    predict_fn : callable
        Custom prediction function (default: model.predict_proba)
    percentile_range : tuple
        (lower, upper) percentiles to focus ALE on (default: 5th-95th)

    Returns:
    --------
    tuple : (grid_centers, ale_cumsum, counts, percentile_bounds)
    """
    if predict_fn is None:
        predict_fn = lambda x: model.predict_proba(x)[:, 1]

    X_work = X.copy()
    feat_values = X_work[feature].dropna().values

    # Check if binary/low cardinality
    unique_vals = np.unique(feat_values)
    is_binary = len(unique_vals) <= 3

    if is_binary:
        # Binary/categorical feature
        unique_vals = np.sort(unique_vals)
        grid = unique_vals

        ale_values = []
        counts_list = []

        for i in range(len(grid)):
            at_value = (X_work[feature] == grid[i]).sum()
            counts_list.append(at_value)

            if i == 0:
                ale_values.append(0)
            else:
                X_sample = X_work.copy()

                # Predict at current value
                X_curr = X_sample.copy()
                X_curr[feature] = grid[i]
                pred_curr = predict_fn(X_curr).mean()

                # Predict at previous value
                X_prev = X_sample.copy()
                X_prev[feature] = grid[i-1]
                pred_prev = predict_fn(X_prev).mean()

                ale_values.append(pred_curr - pred_prev)

        ale_cumsum = np.cumsum(ale_values)
        counts = np.array(counts_list)

        # Center at weighted mean
        if counts.sum() > 0:
            weighted_mean = np.average(ale_cumsum, weights=counts)
            ale_cumsum = ale_cumsum - weighted_mean

        grid_centers = grid
        percentile_bounds = (grid.min(), grid.max())

    else:
        # Continuous feature
        p_lower, p_upper = percentile_range
        lower_bound = np.percentile(feat_values, p_lower)
        upper_bound = np.percentile(feat_values, p_upper)

        interpretable_values = feat_values[
            (feat_values >= lower_bound) & (feat_values <= upper_bound)
        ]

        # Create quantile-based grid
        quantiles = np.linspace(0, 1, grid_size + 1)
        grid = np.quantile(interpretable_values, quantiles)
        grid = np.unique(grid)

        ale_values = np.zeros(len(grid) - 1)
        counts = np.zeros(len(grid) - 1)

        # Compute local effects for each interval
        for i in range(len(grid) - 1):
            in_interval = (X_work[feature] >= grid[i]) & (X_work[feature] < grid[i + 1])

            if in_interval.sum() == 0:
                continue

            X_interval = X_work[in_interval].copy()

            # Predict at lower bound
            X_lower = X_interval.copy()
            X_lower[feature] = grid[i]
            pred_lower = predict_fn(X_lower)

            # Predict at upper bound
            X_upper = X_interval.copy()
            X_upper[feature] = grid[i + 1]
            pred_upper = predict_fn(X_upper)

            # Local effect
            ale_values[i] = (pred_upper - pred_lower).mean()
            counts[i] = in_interval.sum()

        # Accumulate effects
        ale_cumsum = np.cumsum(ale_values)

        # Center ALE (mean = 0)
        valid_counts = counts[counts > 0]
        valid_ale = ale_cumsum[counts > 0]

        if len(valid_counts) > 0:
            ale_cumsum = ale_cumsum - np.average(valid_ale, weights=valid_counts)
        else:
            ale_cumsum = np.zeros_like(ale_cumsum)

        # Grid centers for plotting
        grid_centers = (grid[:-1] + grid[1:]) / 2
        percentile_bounds = (lower_bound, upper_bound)

    return grid_centers, ale_cumsum, counts, percentile_bounds


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
