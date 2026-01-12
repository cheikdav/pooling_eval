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
