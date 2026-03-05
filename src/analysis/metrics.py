"""Metrics for analyzing value estimator performance."""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


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
def compute_mean(baseline_stats, method_stats, epsilon=1e-10):
    """Compute mean (no ratio).

    Args:
        baseline_stats: Not used (kept for signature compatibility)
        method_stats: DataFrame with columns [state_idx, n_episodes, mean, ...]
        epsilon: Not used (kept for signature compatibility)

    Returns:
        DataFrame with metric_value column using raw mean
    """
    result = method_stats.copy()
    result['metric_value'] = result['mean']

    if result['metric_value'].isnull().any():
        null_count = result['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return result


@st.cache_data
def compute_variance_by_value_decile(baseline_stats, method_stats, epsilon=1e-10, n_buckets=10):
    """Compute average variance per bucket of state values.

    For each method/episode number:
    1. Sort states by mean value
    2. Group into n_buckets bins
    3. Compute average variance in each bucket

    Args:
        baseline_stats: Not used (kept for signature compatibility)
        method_stats: DataFrame with columns [state_idx, n_episodes, mean, variance, ...]
        epsilon: Not used (kept for signature compatibility)
        n_buckets: Number of buckets to divide states into (default: 10)

    Returns:
        DataFrame with columns [decile, metric_value] where:
        - decile: 0 to n_buckets-1 (0 = lowest bucket by mean value)
        - metric_value: average variance in that bucket
    """
    result = method_stats.copy()

    # Sort states by mean value and assign buckets
    result = result.sort_values('mean').reset_index(drop=True)
    result['decile'] = pd.qcut(result['mean'], q=n_buckets, labels=False, duplicates='drop')

    # Compute average variance per bucket
    decile_stats = result.groupby('decile')['variance'].mean().reset_index()
    decile_stats.columns = ['decile', 'metric_value']

    # Add n_episodes for consistency
    decile_stats['n_episodes'] = method_stats['n_episodes'].iloc[0]

    if decile_stats['metric_value'].isnull().any():
        null_count = decile_stats['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return decile_stats


@st.cache_data
def compute_normalized_variance_by_value_decile(baseline_stats, method_stats, epsilon=1e-10, n_buckets=10):
    """Compute average of (variance / mean²) per bucket of state values.

    For each method/episode number:
    1. Compute variance / mean² for each state
    2. Sort states by mean value
    3. Group into n_buckets bins
    4. Average the ratios within each bucket

    Args:
        baseline_stats: Not used (kept for signature compatibility)
        method_stats: DataFrame with columns [state_idx, n_episodes, mean, variance, ...]
        epsilon: Small value added to avoid division by zero
        n_buckets: Number of buckets to divide states into (default: 10)

    Returns:
        DataFrame with columns [decile, metric_value] where:
        - decile: 0 to n_buckets-1 (0 = lowest bucket by mean value)
        - metric_value: average of (variance / mean²) in that bucket
    """
    result = method_stats.copy()

    # Compute normalized variance for each state FIRST
    result['normalized_variance'] = result['variance'] / ((result['mean'].abs() + epsilon) ** 2)

    # Sort states by mean value and assign buckets
    result = result.sort_values('mean').reset_index(drop=True)
    result['decile'] = pd.qcut(result['mean'], q=n_buckets, labels=False, duplicates='drop')

    # Average the normalized variance per bucket
    decile_stats = result.groupby('decile')['normalized_variance'].mean().reset_index()
    decile_stats.columns = ['decile', 'metric_value']

    # Add n_episodes for consistency
    decile_stats['n_episodes'] = method_stats['n_episodes'].iloc[0]

    if decile_stats['metric_value'].isnull().any():
        null_count = decile_stats['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return decile_stats


@st.cache_data
def compute_variance_percentiles(baseline_stats, method_stats, epsilon=1e-10, n_buckets=10):
    """Compute variance at different percentiles.

    Args:
        baseline_stats: Not used (kept for signature compatibility)
        method_stats: DataFrame with columns [state_idx, n_episodes, variance, ...]
        epsilon: Not used (kept for signature compatibility)
        n_buckets: Not used (kept for signature compatibility)

    Returns:
        DataFrame with columns [percentile, metric_value]
    """
    result = method_stats.copy()
    percentiles = np.arange(1, 100, 1)  # 1st to 99th percentile

    percentile_values = np.percentile(result['variance'].values, percentiles)

    df = pd.DataFrame({
        'percentile': percentiles,
        'metric_value': percentile_values,
        'n_episodes': method_stats['n_episodes'].iloc[0]
    })

    return df


@st.cache_data
def compute_ground_truth_error(baseline_stats, method_stats, epsilon=1e-10, n_buckets=10):
    """Compute prediction error relative to ground truth returns.

    Args:
        baseline_stats: Not used (kept for signature compatibility)
        method_stats: DataFrame with metadata columns predictions_path and results_dir
        epsilon: Not used (kept for signature compatibility)
        n_buckets: Not used (kept for signature compatibility)

    Returns:
        DataFrame with metric_value column containing (predicted - ground_truth)
    """
    # Extract metadata from method_stats
    if 'predictions_path' not in method_stats.columns or 'results_dir' not in method_stats.columns:
        st.error("Ground truth metric requires predictions_path and results_dir in stats")
        return pd.DataFrame()

    predictions_path = method_stats['predictions_path'].iloc[0]
    results_dir = method_stats['results_dir'].iloc[0]

    # Load ground truth
    ground_truth_file = Path(results_dir) / "ground_truth" / "ground_truth_returns.parquet"
    if not ground_truth_file.exists():
        st.warning(f"Ground truth file not found: {ground_truth_file}")
        return pd.DataFrame()

    ground_truth_df = pd.read_parquet(ground_truth_file)

    # Load predictions (need raw predictions, not aggregated stats)
    predictions_df = pd.read_parquet(predictions_path)

    # Compute mean prediction per state across all batches
    pred_mean = predictions_df.groupby('state_idx')['predicted_value'].mean().reset_index()
    pred_mean.columns = ['state_idx', 'mean_predicted']

    # Merge with ground truth
    merged = pred_mean.merge(
        ground_truth_df[['state_idx', 'ground_truth_return']],
        on='state_idx',
        how='inner'
    )

    # Compute error: predicted - ground_truth
    merged['metric_value'] = merged['mean_predicted'] - merged['ground_truth_return']

    result = merged[['state_idx', 'metric_value']].copy()
    result['n_episodes'] = method_stats['n_episodes'].iloc[0]

    if result['metric_value'].isnull().any():
        null_count = result['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values in ground truth error computation!")

    return result


METRICS = {
    'log_variance_ratio': {
        'name': 'Log Variance Ratio',
        'description': 'log(Method Variance + ε) - log(Baseline Variance + ε)',
        'reference_line': 0,
        'reference_label': 'Equal to Baseline',
        'compute_fn': compute_log_variance_ratio,
        'is_comparison': True,
        'plot_type': 'histogram'
    },
    'log_mean_ratio': {
        'name': 'Log Mean Ratio',
        'description': 'log(|Method Mean| + ε) - log(|Baseline Mean| + ε)',
        'reference_line': 0,
        'reference_label': 'Equal to Baseline',
        'compute_fn': compute_log_mean_ratio,
        'is_comparison': True,
        'plot_type': 'histogram'
    },
    'log_variance': {
        'name': 'Log Variance',
        'description': 'log(Variance + ε)',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_log_variance,
        'is_comparison': False,
        'plot_type': 'histogram'
    },
    'variance': {
        'name': 'Variance',
        'description': 'Raw variance values',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_variance,
        'is_comparison': False,
        'plot_type': 'histogram'
    },
    'mean': {
        'name': 'Mean',
        'description': 'Raw mean values',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_mean,
        'is_comparison': False,
        'plot_type': 'histogram'
    },
    'variance_by_value_decile': {
        'name': 'Variance by Value Decile',
        'description': 'Average variance per decile of state mean values (0=lowest 10%, 9=highest 10%)',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_variance_by_value_decile,
        'is_comparison': False,
        'plot_type': 'bar'
    },
    'normalized_variance_by_value_decile': {
        'name': 'Normalized Variance by Value Decile',
        'description': 'Average (variance / mean²) per decile of state mean values (0=lowest 10%, 9=highest 10%)',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_normalized_variance_by_value_decile,
        'is_comparison': False,
        'plot_type': 'bar'
    },
    'variance_percentiles': {
        'name': 'Variance by Percentile',
        'description': 'Variance values at each percentile (shows distribution from min to max)',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_variance_percentiles,
        'is_comparison': False,
        'plot_type': 'line'
    },
    'ground_truth_error': {
        'name': 'Ground Truth Error',
        'description': 'Prediction error relative to ground truth (Predicted - Ground Truth)',
        'reference_line': 0,
        'reference_label': 'Perfect Prediction',
        'compute_fn': compute_ground_truth_error,
        'is_comparison': False,
        'plot_type': 'histogram'
    },
    'batch_constants': {
        'name': 'Batch Adjustment Constants',
        'description': 'Per-batch constants used for adjustment: c = mean(ground_truth) - mean(predictions_batch)',
        'reference_line': 0,
        'reference_label': 'No Adjustment',
        'compute_fn': None,  # Special handling required
        'is_comparison': False,
        'plot_type': 'histogram',
        'requires_special_handling': True  # Flag for special processing
    }
}


def get_metrics_by_type(is_comparison):
    """Get metric keys filtered by comparison type.

    Args:
        is_comparison: If True, return comparison metrics; if False, return absolute metrics

    Returns:
        List of metric keys
    """
    return [key for key, info in METRICS.items() if info['is_comparison'] == is_comparison]


@st.cache_data
def compute_metric(baseline_stats, method_stats, metric_key, epsilon=1e-10, n_buckets=10):
    """Compute the specified metric with error handling.

    Args:
        baseline_stats: DataFrame with baseline method stats
        method_stats: DataFrame with method stats to compare
        metric_key: Key identifying which metric to compute
        epsilon: Small value added before taking log to avoid log(0)
        n_buckets: Number of buckets for decile-based metrics

    Returns:
        DataFrame with metric_value column
    """
    try:
        if metric_key not in METRICS:
            raise ValueError(f"Unknown metric: {metric_key}")

        compute_fn = METRICS[metric_key]['compute_fn']

        # Pass n_buckets parameter for decile-based metrics
        if metric_key in ['variance_by_value_decile', 'normalized_variance_by_value_decile']:
            return compute_fn(baseline_stats, method_stats, epsilon=epsilon, n_buckets=n_buckets)
        else:
            return compute_fn(baseline_stats, method_stats, epsilon=epsilon)
    except Exception as e:
        st.error(f"Error computing {metric_key}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        raise
