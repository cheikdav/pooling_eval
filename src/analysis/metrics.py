"""Metrics for analyzing value estimator performance."""

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def compute_log_variance_ratio(baseline_stats, method_stats, epsilon=1e-10):
    """Compute log variance ratios relative to a baseline method.

    Args:
        baseline_stats: DataFrame with columns [state_idx, n_episodes, variance, ...]
        method_stats: DataFrame with columns [state_idx, n_episodes, variance, ...]
        epsilon: Small value added before taking log to avoid log(0)

    Returns:
        DataFrame with metric_value column using log(x+eps) - log(y+eps)
    """
    baseline_values = baseline_stats[['state_idx', 'n_episodes', 'variance']].copy()
    baseline_values.columns = ['state_idx', 'n_episodes', 'baseline_variance']

    merged = method_stats.merge(baseline_values, on=['state_idx', 'n_episodes'])

    # Compute log(method_variance + eps) - log(baseline_variance + eps)
    merged['metric_value'] = np.log(merged['variance'] + epsilon) - np.log(merged['baseline_variance'] + epsilon)

    if merged['metric_value'].isnull().any():
        null_count = merged['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return merged


@st.cache_data
def compute_log_mean_ratio(baseline_stats, method_stats, epsilon=1e-10):
    """Compute log mean ratios relative to a baseline method.

    Args:
        baseline_stats: DataFrame with columns [state_idx, n_episodes, mean, ...]
        method_stats: DataFrame with columns [state_idx, n_episodes, mean, ...]
        epsilon: Small value added before taking log to avoid log(0)

    Returns:
        DataFrame with metric_value column using log(|x|+eps) - log(|y|+eps)
    """
    baseline_values = baseline_stats[['state_idx', 'n_episodes', 'mean']].copy()
    baseline_values.columns = ['state_idx', 'n_episodes', 'baseline_mean']

    merged = method_stats.merge(baseline_values, on=['state_idx', 'n_episodes'])

    # Compute log(|method_mean| + eps) - log(|baseline_mean| + eps)
    merged['metric_value'] = np.log(merged['mean'].abs() + epsilon) - np.log(merged['baseline_mean'].abs() + epsilon)

    if merged['metric_value'].isnull().any():
        null_count = merged['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return merged


@st.cache_data
def compute_log_variance(baseline_stats, method_stats, epsilon=1e-10):
    """Compute log variance (no ratio).

    Args:
        baseline_stats: Not used (kept for signature compatibility)
        method_stats: DataFrame with columns [state_idx, n_episodes, variance, ...]
        epsilon: Small value added before taking log to avoid log(0)

    Returns:
        DataFrame with metric_value column using log(variance + eps)
    """
    result = method_stats.copy()
    result['metric_value'] = np.log(result['variance'] + epsilon)

    if result['metric_value'].isnull().any():
        null_count = result['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return result


@st.cache_data
def compute_variance(baseline_stats, method_stats, epsilon=1e-10):
    """Compute variance (no ratio).

    Args:
        baseline_stats: Not used (kept for signature compatibility)
        method_stats: DataFrame with columns [state_idx, n_episodes, variance, ...]
        epsilon: Not used (kept for signature compatibility)

    Returns:
        DataFrame with metric_value column using raw variance
    """
    result = method_stats.copy()
    result['metric_value'] = result['variance']

    if result['metric_value'].isnull().any():
        null_count = result['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return result


@st.cache_data
def compute_variance_by_value_decile(baseline_stats, method_stats, epsilon=1e-10):
    """Compute average variance per decile of state values.

    For each method/episode number:
    1. Sort states by mean value
    2. Group into 10 deciles (10% bins)
    3. Compute average variance in each decile

    Args:
        baseline_stats: Not used (kept for signature compatibility)
        method_stats: DataFrame with columns [state_idx, n_episodes, mean, variance, ...]
        epsilon: Not used (kept for signature compatibility)

    Returns:
        DataFrame with columns [decile, metric_value] where:
        - decile: 0-9 (0 = lowest 10% by mean value, 9 = highest 10%)
        - metric_value: average variance in that decile
    """
    result = method_stats.copy()

    # Sort states by mean value and assign deciles
    result = result.sort_values('mean').reset_index(drop=True)
    result['decile'] = pd.qcut(result['mean'], q=10, labels=False, duplicates='drop')

    # Compute average variance per decile
    decile_stats = result.groupby('decile')['variance'].mean().reset_index()
    decile_stats.columns = ['decile', 'metric_value']

    # Add n_episodes for consistency
    decile_stats['n_episodes'] = method_stats['n_episodes'].iloc[0]

    if decile_stats['metric_value'].isnull().any():
        null_count = decile_stats['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return decile_stats


@st.cache_data
def compute_normalized_variance_by_value_decile(baseline_stats, method_stats, epsilon=1e-10):
    """Compute average variance divided by average mean per decile of state values.

    For each method/episode number:
    1. Sort states by mean value
    2. Group into 10 deciles (10% bins)
    3. Compute average variance / average mean in each decile

    Args:
        baseline_stats: Not used (kept for signature compatibility)
        method_stats: DataFrame with columns [state_idx, n_episodes, mean, variance, ...]
        epsilon: Small value added to avoid division by zero

    Returns:
        DataFrame with columns [decile, metric_value] where:
        - decile: 0-9 (0 = lowest 10% by mean value, 9 = highest 10%)
        - metric_value: average variance / average mean in that decile
    """
    result = method_stats.copy()

    # Sort states by mean value and assign deciles
    result = result.sort_values('mean').reset_index(drop=True)
    result['decile'] = pd.qcut(result['mean'], q=10, labels=False, duplicates='drop')

    # Compute average variance and mean per decile
    decile_stats = result.groupby('decile').agg({
        'variance': 'mean',
        'mean': 'mean'
    }).reset_index()

    # Normalize: variance / mean (add epsilon to avoid division by zero)
    decile_stats['metric_value'] = decile_stats['variance'] / (decile_stats['mean'].abs() + epsilon)

    # Keep only necessary columns
    decile_stats = decile_stats[['decile', 'metric_value']]

    # Add n_episodes for consistency
    decile_stats['n_episodes'] = method_stats['n_episodes'].iloc[0]

    if decile_stats['metric_value'].isnull().any():
        null_count = decile_stats['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return decile_stats


METRICS = {
    'log_variance_ratio': {
        'name': 'Log Variance Ratio',
        'description': 'log(Method Variance + ε) - log(Baseline Variance + ε)',
        'reference_line': 0,
        'reference_label': 'Equal to Baseline',
        'compute_fn': compute_log_variance_ratio
    },
    'log_mean_ratio': {
        'name': 'Log Mean Ratio',
        'description': 'log(|Method Mean| + ε) - log(|Baseline Mean| + ε)',
        'reference_line': 0,
        'reference_label': 'Equal to Baseline',
        'compute_fn': compute_log_mean_ratio
    },
    'log_variance': {
        'name': 'Log Variance',
        'description': 'log(Variance + ε)',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_log_variance
    },
    'variance': {
        'name': 'Variance',
        'description': 'Raw variance values',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_variance
    },
    'variance_by_value_decile': {
        'name': 'Variance by Value Decile',
        'description': 'Average variance per decile of state mean values (0=lowest 10%, 9=highest 10%)',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_variance_by_value_decile
    },
    'normalized_variance_by_value_decile': {
        'name': 'Normalized Variance by Value Decile',
        'description': 'Average (variance / mean) per decile of state mean values (0=lowest 10%, 9=highest 10%)',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_normalized_variance_by_value_decile
    }
}


@st.cache_data
def compute_metric(baseline_stats, method_stats, metric_key, epsilon=1e-10):
    """Compute the specified metric with error handling.

    Args:
        baseline_stats: DataFrame with baseline method stats
        method_stats: DataFrame with method stats to compare
        metric_key: Key identifying which metric to compute
        epsilon: Small value added before taking log to avoid log(0)

    Returns:
        DataFrame with metric_value column
    """
    try:
        if metric_key not in METRICS:
            raise ValueError(f"Unknown metric: {metric_key}")

        compute_fn = METRICS[metric_key]['compute_fn']
        return compute_fn(baseline_stats, method_stats, epsilon=epsilon)
    except Exception as e:
        st.error(f"Error computing {metric_key}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        raise
