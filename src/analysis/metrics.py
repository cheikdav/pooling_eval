"""Metrics for analyzing value estimator performance."""

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def compute_log_variance_ratio(baseline_stats, method_stats):
    """Compute log variance ratios relative to a baseline method.

    Args:
        baseline_stats: DataFrame with columns [state_idx, n_episodes, variance, ...]
        method_stats: DataFrame with columns [state_idx, n_episodes, variance, ...]

    Returns:
        DataFrame with metric_value column
    """
    baseline_values = baseline_stats[['state_idx', 'n_episodes', 'variance']].copy()
    baseline_values.columns = ['state_idx', 'n_episodes', 'baseline_variance']

    if (baseline_values['baseline_variance'] == 0).any():
        zero_count = (baseline_values['baseline_variance'] == 0).sum()
        st.warning(f"Found {zero_count} baseline variance values equal to 0. Replacing with epsilon=1e-10.")
        baseline_values['baseline_variance'] = baseline_values['baseline_variance'].replace(0, 1e-10)

    merged = method_stats.merge(baseline_values, on=['state_idx', 'n_episodes'])
    merged['variance_ratio'] = merged['variance'] / merged['baseline_variance']

    if (merged['variance_ratio'] <= 0).any():
        invalid_count = (merged['variance_ratio'] <= 0).sum()
        st.warning(f"Found {invalid_count} variance ratios ≤ 0. Replacing with epsilon=1e-10.")
        merged['variance_ratio'] = merged['variance_ratio'].clip(lower=1e-10)

    merged['metric_value'] = np.log(merged['variance_ratio'])

    if merged['metric_value'].isnull().any():
        null_count = merged['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return merged


@st.cache_data
def compute_log_mean_ratio(baseline_stats, method_stats):
    """Compute log mean ratios relative to a baseline method.

    Args:
        baseline_stats: DataFrame with columns [state_idx, n_episodes, mean, ...]
        method_stats: DataFrame with columns [state_idx, n_episodes, mean, ...]

    Returns:
        DataFrame with metric_value column
    """
    baseline_values = baseline_stats[['state_idx', 'n_episodes', 'mean']].copy()
    baseline_values.columns = ['state_idx', 'n_episodes', 'baseline_mean']

    merged = method_stats.merge(baseline_values, on=['state_idx', 'n_episodes'])

    # Handle division by zero or negative values
    eps = 1e-10
    merged['mean_abs'] = merged['mean'].abs().clip(lower=eps)
    merged['baseline_mean_abs'] = merged['baseline_mean'].abs().clip(lower=eps)

    merged['mean_ratio'] = merged['mean_abs'] / merged['baseline_mean_abs']
    merged['metric_value'] = np.log(merged['mean_ratio'])

    if merged['metric_value'].isnull().any():
        null_count = merged['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return merged


METRICS = {
    'log_variance_ratio': {
        'name': 'Log Variance Ratio',
        'description': 'log(Method Variance / Baseline Variance)',
        'reference_line': 0,
        'reference_label': 'Equal to Baseline',
        'compute_fn': compute_log_variance_ratio
    },
    'log_mean_ratio': {
        'name': 'Log Mean Ratio',
        'description': 'log(|Method Mean| / |Baseline Mean|)',
        'reference_line': 0,
        'reference_label': 'Equal to Baseline',
        'compute_fn': compute_log_mean_ratio
    }
}


@st.cache_data
def compute_metric(baseline_stats, method_stats, metric_key):
    """Compute the specified metric with error handling.

    Args:
        baseline_stats: DataFrame with baseline method stats
        method_stats: DataFrame with method stats to compare
        metric_key: Key identifying which metric to compute

    Returns:
        DataFrame with metric_value column
    """
    try:
        if metric_key not in METRICS:
            raise ValueError(f"Unknown metric: {metric_key}")

        compute_fn = METRICS[metric_key]['compute_fn']
        return compute_fn(baseline_stats, method_stats)
    except Exception as e:
        st.error(f"Error computing {metric_key}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        raise
