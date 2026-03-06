"""Tab for absolute (non-comparison) metrics."""

import streamlit as st
import pandas as pd

from metrics import METRICS, get_metrics_by_type
from common import compute_stats_from_predictions, apply_data_filters, compute_ground_truth_stats
from plotting import plot_metric_for_single_episodes, plot_metric_evolution


def render_tab(filtered_metadata, methods, baseline_method, epsilon, dataset_type, n_buckets, filter_high_variance, filter_extreme_mean, temporal_p=0.2, adjust_constant=False):
    """Render the absolute metrics tab.

    Args:
        filtered_metadata: DataFrame with experiment metadata
        methods: List of methods to display
        baseline_method: Baseline method name (not used for absolute metrics, but passed for consistency)
        epsilon: Small value added before taking log
        dataset_type: 'full', 'differences', or 'temporal'
        n_buckets: Number of buckets for decile-based metrics
        filter_high_variance: Percentage of top variance states to exclude
        filter_extreme_mean: Percentage of top/bottom mean states to exclude
        temporal_p: Geometric distribution parameter for temporal differences
        adjust_constant: If True, add constant so mean(predictions) = mean(ground_truth)
    """
    st.header("📊 Absolute Metrics")

    # Get absolute metrics only
    absolute_metrics = get_metrics_by_type(is_comparison=False)

    # Single metric selector (shared between both sections)
    metric_key = st.selectbox(
        "Select Metric:",
        absolute_metrics,
        format_func=lambda k: METRICS[k]['name'],
        key="absolute_metric"
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
        key="absolute_n_ep"
    )

    # Load stats for selected n_episodes
    filtered_for_n_ep = filtered_metadata[filtered_metadata['n_episodes'] == selected_n_ep]

    stats_dict_single = {}
    for _, row in filtered_for_n_ep.iterrows():
        if row['method'] in methods:
            stats = compute_stats_from_predictions(row['predictions_path'], row['n_episodes'], dataset_type=dataset_type, temporal_p=temporal_p, adjust_constant=adjust_constant)
            # Apply data filters
            stats = apply_data_filters(stats, filter_high_variance, filter_extreme_mean)
            stats_dict_single[row['method']] = stats

    if not stats_dict_single:
        st.error(f"No data for {selected_n_ep} episodes")
        return

    # Compute ground truth stats once for all methods (used by ground_truth_error metric)
    ground_truth_stats = None
    if metric_key == 'ground_truth_error':
        # Get results_dir from any method's stats
        first_method_stats = next(iter(stats_dict_single.values()))
        if 'results_dir' in first_method_stats.columns:
            results_dir = first_method_stats['results_dir'].iloc[0]
            ground_truth_stats = compute_ground_truth_stats(results_dir, dataset_type=dataset_type,
                                                            s1_proportion=0.9, seed=42,
                                                            temporal_p=temporal_p)

    # Special handling for batch_constants metric
    if metric_key == 'batch_constants':
        from common import compute_all_batch_constants, get_method_display_name
        import plotly.express as px

        constants_df = compute_all_batch_constants(filtered_metadata, methods, selected_n_ep)

        if constants_df.empty:
            st.warning("No ground truth data available to compute adjustment constants")
        else:
            # Add display names
            constants_df['method_display'] = constants_df['method'].apply(get_method_display_name)

            col1, col2 = st.columns([3, 1])

            with col1:
                fig = px.histogram(
                    constants_df, x='constant', color='method_display',
                    nbins=40, opacity=0.7, barmode='overlay',
                    title=f"Batch Adjustment Constants Distribution ({selected_n_ep} episodes)",
                    labels={'constant': 'Adjustment Constant', 'method_display': 'Method'}
                )

                fig.add_vline(
                    x=0, line_dash="dash", line_color="red",
                    annotation_text="No Adjustment"
                )

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Statistics**")
                summary = constants_df.groupby('method_display')['constant'].agg(
                    mean='mean', std='std'
                ).reset_index()
                st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        plot_metric_for_single_episodes(stats_dict_single, metric_key, methods, selected_n_ep, baseline_method, epsilon, n_buckets, ground_truth_stats)

    st.markdown("---")

    # Section 2: Evolution across training sizes
    st.subheader("Evolution Across Training Sizes")

    # Special handling for batch_constants metric in evolution
    if metric_key == 'batch_constants':
        from common import compute_all_batch_constants, get_method_display_name
        import plotly.graph_objects as go

        # Collect constants for all n_episodes
        all_evolution_data = []
        for n_ep in n_episodes_values:
            constants_df = compute_all_batch_constants(filtered_metadata, methods, n_ep)
            if not constants_df.empty:
                # Compute mean and std for each method at this n_episodes
                for method in methods:
                    method_data = constants_df[constants_df['method'] == method]
                    if not method_data.empty:
                        all_evolution_data.append({
                            'method': get_method_display_name(method),
                            'n_episodes': n_ep,
                            'mean': method_data['constant'].mean(),
                            'std': method_data['constant'].std()
                        })

        if all_evolution_data:
            evolution_df = pd.DataFrame(all_evolution_data)

            # Create line plot
            fig = go.Figure()

            for method in evolution_df['method'].unique():
                method_data = evolution_df[evolution_df['method'] == method]
                fig.add_trace(go.Scatter(
                    x=method_data['n_episodes'],
                    y=method_data['mean'],
                    mode='lines+markers',
                    name=method,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))

            fig.update_layout(
                title="Mean Adjustment Constant vs Training Episodes",
                xaxis_title="Number of Training Episodes",
                yaxis_title="Mean Adjustment Constant",
                height=500,
                hovermode='x unified'
            )

            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No ground truth data available to compute adjustment constants")
    else:
        plot_metric_evolution(filtered_metadata, metric_key, methods, baseline_method, n_episodes_values, epsilon, dataset_type, n_buckets, temporal_p, adjust_constant)
