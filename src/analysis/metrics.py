"""Metrics for analyzing value estimator performance."""

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def compute_log_variance_ratio(df, stats):
    """Compute log variance ratios relative to Monte Carlo."""
    mc_values = stats[stats['method'] == 'monte_carlo'][['state_idx', 'n_episodes', 'variance']]

    if len(mc_values) == 0:
        raise ValueError("No Monte Carlo method found in data.")

    mc_values.columns = ['state_idx', 'n_episodes', 'mc_variance']

    if (mc_values['mc_variance'] == 0).any():
        zero_count = (mc_values['mc_variance'] == 0).sum()
        st.warning(f"Found {zero_count} MC variance values equal to 0. Replacing with epsilon=1e-10.")
        mc_values['mc_variance'] = mc_values['mc_variance'].replace(0, 1e-10)

    merged = stats.merge(mc_values, on=['state_idx', 'n_episodes'])
    merged['variance_ratio'] = merged['variance'] / merged['mc_variance']

    if (merged['variance_ratio'] <= 0).any():
        invalid_count = (merged['variance_ratio'] <= 0).sum()
        st.warning(f"Found {invalid_count} variance ratios ≤ 0. Replacing with epsilon=1e-10.")
        merged['variance_ratio'] = merged['variance_ratio'].clip(lower=1e-10)

    merged['metric_value'] = np.log(merged['variance_ratio'])

    if merged['metric_value'].isnull().any():
        null_count = merged['metric_value'].isnull().sum()
        st.error(f"Found {null_count} NaN values after computation!")

    return merged[merged['method'] != 'monte_carlo'].copy()


METRICS = {
    'log_variance_ratio': {
        'name': 'Log Variance Ratio',
        'description': 'log(Method Variance / MC Variance)',
        'reference_line': 0,
        'reference_label': 'Equal to MC',
        'compute_fn': compute_log_variance_ratio
    }
}


@st.cache_data
def compute_metric(df, stats, metric_key):
    """Compute the specified metric with error handling."""
    try:
        if metric_key not in METRICS:
            raise ValueError(f"Unknown metric: {metric_key}")

        compute_fn = METRICS[metric_key]['compute_fn']
        return compute_fn(df, stats)
    except Exception as e:
        st.error(f"Error computing {metric_key}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        raise
