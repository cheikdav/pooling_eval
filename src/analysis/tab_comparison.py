"""Tab for comparison metrics."""

import streamlit as st

from metrics import METRICS, get_metrics_by_type
from common import compute_stats_from_predictions, apply_data_filters
from plotting import plot_metric_for_single_episodes, plot_metric_evolution


def render_tab(filtered_metadata, methods, baseline_method, epsilon, dataset_type, n_buckets, filter_high_variance, filter_extreme_mean):
    """Render the comparison metrics tab.

    Args:
        filtered_metadata: DataFrame with experiment metadata
        methods: List of methods to display
        baseline_method: Baseline method name
        epsilon: Small value added before taking log
        dataset_type: 'full' or 'differences'
        n_buckets: Number of buckets for decile-based metrics
        filter_high_variance: Percentage of top variance states to exclude
        filter_extreme_mean: Percentage of top/bottom mean states to exclude
    """
    st.header("📈 Comparison Metrics")

    # Get comparison metrics only
    comparison_metrics = get_metrics_by_type(is_comparison=True)

    # Single metric selector (shared between both sections)
    metric_key = st.selectbox(
        "Select Metric:",
        comparison_metrics,
        format_func=lambda k: METRICS[k]['name'],
        key="comparison_metric"
    )

    st.markdown(f"**{METRICS[metric_key]['name']}**: {METRICS[metric_key]['description']}")
    st.markdown("---")

    # Section 1: Analysis for specific training size
    st.subheader("Analysis for Specific Training Size")

    n_episodes_values = sorted(filtered_metadata['n_episodes'].unique())
    selected_n_ep = st.selectbox(
        "Training data size:",
        n_episodes_values,
        format_func=lambda x: f"{x} episodes",
        key="comparison_n_ep"
    )

    # Load stats for selected n_episodes (include baseline method)
    methods_to_load = list(set(methods + [baseline_method]))
    filtered_for_n_ep = filtered_metadata[filtered_metadata['n_episodes'] == selected_n_ep]

    stats_dict_single = {}
    for _, row in filtered_for_n_ep.iterrows():
        if row['method'] in methods_to_load:
            stats = compute_stats_from_predictions(row['predictions_path'], row['n_episodes'], dataset_type=dataset_type)
            # Apply data filters
            stats = apply_data_filters(stats, filter_high_variance, filter_extreme_mean)
            stats_dict_single[row['method']] = stats

    if not stats_dict_single or baseline_method not in stats_dict_single:
        st.error(f"No data for {selected_n_ep} episodes")
        return

    plot_metric_for_single_episodes(stats_dict_single, metric_key, methods, selected_n_ep, baseline_method, epsilon, n_buckets)

    st.markdown("---")

    # Section 2: Evolution across training sizes
    st.subheader("Evolution Across Training Sizes")

    plot_metric_evolution(filtered_metadata, metric_key, methods, baseline_method, n_episodes_values, epsilon, dataset_type, n_buckets)
