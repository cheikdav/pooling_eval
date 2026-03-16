"""Metrics for analyzing value estimator performance."""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from common import MetricContext


@st.cache_data
def compute_log_variance_ratio(context: MetricContext, method: str):
    """Compute log variance ratios relative to a baseline method.

    Args:
        context: MetricContext with all data and parameters
        method: Method name to compute ratio for

    Returns:
        DataFrame with metric_value column using log(x+eps) - log(y+eps)
    """
    baseline_stats = context.get_baseline_stats()
    method_stats = context.method_stats.get(method)

    if baseline_stats is None or method_stats is None:
        return pd.DataFrame()

    baseline_values = baseline_stats[['state_idx', 'n_episodes', 'variance']].copy()
    baseline_values.columns = ['state_idx', 'n_episodes', 'baseline_variance']

    merged = method_stats.merge(baseline_values, on=['state_idx', 'n_episodes'])

    # Compute log(method_variance + eps) - log(baseline_variance + eps)
    merged['metric_value'] = np.log(merged['variance'] + context.epsilon) - np.log(merged['baseline_variance'] + context.epsilon)

    if merged['metric_value'].isnull().any():
        null_count = merged['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return merged


@st.cache_data
def compute_log_mean_ratio(context: MetricContext, method: str):
    """Compute log mean ratios relative to a baseline method.

    Args:
        context: MetricContext with all data and parameters
        method: Method name to compute ratio for

    Returns:
        DataFrame with metric_value column using log(|x|+eps) - log(|y|+eps)
    """
    baseline_stats = context.get_baseline_stats()
    method_stats = context.method_stats.get(method)

    if baseline_stats is None or method_stats is None:
        return pd.DataFrame()

    baseline_values = baseline_stats[['state_idx', 'n_episodes', 'mean']].copy()
    baseline_values.columns = ['state_idx', 'n_episodes', 'baseline_mean']

    merged = method_stats.merge(baseline_values, on=['state_idx', 'n_episodes'])

    # Compute log(|method_mean| + eps) - log(|baseline_mean| + eps)
    merged['metric_value'] = np.log(merged['mean'].abs() + context.epsilon) - np.log(merged['baseline_mean'].abs() + context.epsilon)

    if merged['metric_value'].isnull().any():
        null_count = merged['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return merged


@st.cache_data
def compute_log_variance(context: MetricContext, method: str):
    """Compute log variance (no ratio).

    Args:
        context: MetricContext with all data and parameters
        method: Method name to compute for

    Returns:
        DataFrame with metric_value column using log(variance + eps)
    """
    method_stats = context.method_stats.get(method)

    if method_stats is None:
        return pd.DataFrame()

    result = method_stats.copy()
    result['metric_value'] = np.log(result['variance'] + context.epsilon)

    if result['metric_value'].isnull().any():
        null_count = result['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return result


@st.cache_data
def compute_variance(context: MetricContext, method: str):
    """Compute variance (no ratio).

    Args:
        context: MetricContext with all data and parameters
        method: Method name to compute for

    Returns:
        DataFrame with metric_value column using raw variance
    """
    method_stats = context.method_stats.get(method)

    if method_stats is None:
        return pd.DataFrame()

    result = method_stats.copy()
    result['metric_value'] = result['variance']

    if result['metric_value'].isnull().any():
        null_count = result['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return result


@st.cache_data
def compute_mean(context: MetricContext, method: str):
    """Compute mean (no ratio).

    Args:
        context: MetricContext with all data and parameters
        method: Method name to compute for

    Returns:
        DataFrame with metric_value column using raw mean
    """
    method_stats = context.method_stats.get(method)

    if method_stats is None:
        return pd.DataFrame()

    result = method_stats.copy()
    result['metric_value'] = result['mean']

    if result['metric_value'].isnull().any():
        null_count = result['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return result


@st.cache_data
def compute_variance_by_value_decile(context: MetricContext, method: str):
    """Compute average variance per bucket of state values.

    For each method/episode number:
    1. Sort states by mean value
    2. Group into n_buckets bins
    3. Compute average variance in each bucket

    Args:
        context: MetricContext with all data and parameters
        method: Method name to compute for

    Returns:
        DataFrame with columns [decile, metric_value] where:
        - decile: 0 to n_buckets-1 (0 = lowest bucket by mean value)
        - metric_value: average variance in that bucket
    """
    method_stats = context.method_stats.get(method)

    if method_stats is None:
        return pd.DataFrame()

    result = method_stats.copy()

    # Sort states by mean value and assign buckets
    result = result.sort_values('mean').reset_index(drop=True)
    result['decile'] = pd.qcut(result['mean'], q=context.n_buckets, labels=False, duplicates='drop')

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
def compute_normalized_variance_by_value_decile(context: MetricContext, method: str):
    """Compute average of (variance / mean²) per bucket of state values.

    For each method/episode number:
    1. Compute variance / mean² for each state
    2. Sort states by mean value
    3. Group into n_buckets bins
    4. Average the ratios within each bucket

    Args:
        context: MetricContext with all data and parameters
        method: Method name to compute for

    Returns:
        DataFrame with columns [decile, metric_value] where:
        - decile: 0 to n_buckets-1 (0 = lowest bucket by mean value)
        - metric_value: average of (variance / mean²) in that bucket
    """
    method_stats = context.method_stats.get(method)

    if method_stats is None:
        return pd.DataFrame()

    result = method_stats.copy()

    # Compute normalized variance for each state FIRST
    result['normalized_variance'] = result['variance'] / ((result['mean'].abs() + context.epsilon) ** 2)

    # Sort states by mean value and assign buckets
    result = result.sort_values('mean').reset_index(drop=True)
    result['decile'] = pd.qcut(result['mean'], q=context.n_buckets, labels=False, duplicates='drop')

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
def compute_variance_percentiles(context: MetricContext, method: str):
    """Compute variance at different percentiles.

    Args:
        context: MetricContext with all data and parameters
        method: Method name to compute for

    Returns:
        DataFrame with columns [percentile, metric_value]
    """
    method_stats = context.method_stats.get(method)

    if method_stats is None:
        return pd.DataFrame()

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
def compute_ground_truth_error(context: MetricContext, method: str):
    """Compute prediction error relative to ground truth returns.

    Args:
        context: MetricContext with all data and parameters
        method: Method name to compute for

    Returns:
        DataFrame with metric_value column containing (predicted - ground_truth)
    """
    method_stats = context.method_stats.get(method)

    if method_stats is None:
        return pd.DataFrame()

    if context.ground_truth_stats is None or context.ground_truth_stats.empty:
        st.warning("Ground truth data not available")
        return pd.DataFrame()

    # Use already computed mean from method_stats (which includes any adjustments)
    result = method_stats[['state_idx', 'mean', 'n_episodes']].copy()

    # Merge with ground truth
    result = result.merge(
        context.ground_truth_stats[['state_idx', 'ground_truth_value']],
        on='state_idx',
        how='inner'
    )

    # Compute error: predicted - ground_truth
    result['metric_value'] = result['mean'] - result['ground_truth_value']

    # Keep only needed columns
    result = result[['state_idx', 'metric_value', 'n_episodes']].copy()

    if result['metric_value'].isnull().any():
        null_count = result['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values in ground truth error computation!")

    return result


@st.cache_data
def compute_ground_truth_error_squared(context: MetricContext, method: str):
    """Compute squared prediction error relative to ground truth returns.

    Returns:
        DataFrame with metric_value column containing (predicted - ground_truth)²
    """
    result = compute_ground_truth_error(context, method)
    if result.empty:
        return result

    result = result.copy()
    result['metric_value'] = result['metric_value'] ** 2
    return result


@st.cache_data
def compute_mse(context: MetricContext, method: str):
    """Compute MSE at each state: Var(V̂(s)) + (E[V̂(s)] - V*(s))².

    This is the bias-variance decomposition of E[(V̂(s) - V*(s))²].
    Also verifies equivalence with the direct computation.

    Returns:
        DataFrame with metric_value column containing per-state MSE
    """
    method_stats = context.method_stats.get(method)
    if method_stats is None:
        return pd.DataFrame()

    if context.ground_truth_stats is None or context.ground_truth_stats.empty:
        st.warning("Ground truth data not available")
        return pd.DataFrame()

    result = method_stats[['state_idx', 'mean', 'variance', 'n_episodes']].copy()
    result = result.merge(
        context.ground_truth_stats[['state_idx', 'ground_truth_value']],
        on='state_idx', how='inner'
    )

    # Approach 2: bias² + variance
    bias_squared = (result['mean'] - result['ground_truth_value']) ** 2
    result['metric_value'] = result['variance'] + bias_squared

    # Sanity check: approach 1 (direct) should give the same result
    # E[(V̂ - V*)²] = E[V̂²] - 2·V*·E[V̂] + V*²
    #               = (Var + mean²) - 2·V*·mean + V*²
    #               = Var + (mean - V*)²  ✓
    direct = (result['variance'] + result['mean']**2
              - 2 * result['ground_truth_value'] * result['mean']
              + result['ground_truth_value']**2)
    max_diff = (result['metric_value'] - direct).abs().max()
    if max_diff > 1e-10:
        print(f"MSE decomposition mismatch (catastrophic cancellation): max diff = {max_diff:.2e}")

    result = result[['state_idx', 'metric_value', 'n_episodes']].copy()
    return result


METRICS = {
    'log_variance': {
        'name': 'Log Variance',
        'description': 'log(Variance + ε)',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_log_variance,
        'is_comparison': False,
        'plot_type': 'histogram'
    },
    'ground_truth_error_squared': {
        'name': 'Bias²',
        'description': 'Squared bias: (E[V̂(s)] - V*(s))²',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_ground_truth_error_squared,
        'is_comparison': False,
        'plot_type': 'histogram'
    },
    'mse': {
        'name': 'MSE',
        'description': 'Per-state MSE: Var(V̂(s)) + (E[V̂(s)] - V*(s))² = E[(V̂(s) - V*(s))²]',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': compute_mse,
        'is_comparison': False,
        'plot_type': 'histogram'
    },
    'bias_variance_decomposition': {
        'name': 'Bias-Variance Decomposition',
        'description': 'Side-by-side: Variance, Bias², and MSE = Variance + Bias²',
        'reference_line': None,
        'reference_label': None,
        'compute_fn': None,
        'is_comparison': False,
        'plot_type': 'histogram',
        'requires_special_handling': True
    },
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
        'name': 'Bias',
        'description': 'Per-state bias: E[V̂(s)] - V*(s)',
        'reference_line': 0,
        'reference_label': 'Zero Bias',
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
