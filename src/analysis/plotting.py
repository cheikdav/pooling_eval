"""Shared plotting utilities for the dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from metrics import METRICS
from common import get_method_display_name, compute_stats_from_predictions, compute_all_batch_constants, compute_ground_truth_stats, MetricContext


def plot_metric_for_single_episodes(context: MetricContext, metric_key: str):
    """Plot metric for a single n_episodes value.

    Args:
        context: MetricContext with all data and parameters
        metric_key: Metric to compute
    """
    metric_info = METRICS[metric_key]
    plot_type = metric_info['plot_type']
    is_comparison = metric_info['is_comparison']

    if is_comparison and context.baseline_method not in context.method_stats:
        st.error(f"Baseline method {context.baseline_method} not found")
        return

    # Compute metrics for each method
    metric_list = []
    for method in context.methods_to_display:
        # For comparison metrics, skip the baseline method
        if is_comparison and method == context.baseline_method:
            continue

        if method not in context.method_stats:
            continue

        compute_fn = metric_info['compute_fn']
        metric_df = compute_fn(context, method)
        metric_df['method'] = get_method_display_name(method)
        metric_list.append(metric_df)

    if not metric_list:
        st.warning("No data available")
        return

    combined = pd.concat(metric_list, ignore_index=True)

    # Create appropriate plot based on metric type
    if plot_type == 'histogram':
        col1, col2 = st.columns([3, 1])

        with col1:
            fig = px.histogram(
                combined, x='metric_value', color='method',
                nbins=40, opacity=0.7, barmode='overlay',
                title=f"{metric_info['name']} Distribution ({context.n_episodes} episodes)",
                labels={'metric_value': metric_info['name']}
            )

            if metric_info['reference_line'] is not None:
                fig.add_vline(
                    x=metric_info['reference_line'], line_dash="dash", line_color="red",
                    annotation_text=metric_info['reference_label']
                )

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if is_comparison:
                st.markdown(f"**Statistics** (vs {get_method_display_name(context.baseline_method)})")
            else:
                st.markdown("**Statistics**")
            summary = combined.groupby('method')['metric_value'].agg(
                mean='mean', std='std'
            ).reset_index()
            st.dataframe(summary, use_container_width=True, hide_index=True)

    elif plot_type == 'bar':
        # Bar chart for decile-based metrics
        fig = px.bar(
            combined,
            x='decile',
            y='metric_value',
            color='method',
            barmode='group',
            title=f"{metric_info['name']} ({context.n_episodes} episodes, {context.n_buckets} buckets)",
            labels={
                'decile': f'Value Bucket (0=lowest, {context.n_buckets-1}=highest)',
                'metric_value': metric_info['name']
            }
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Show statistics table
        st.markdown("**Statistics by Bucket**")
        pivot = combined.pivot(index='decile', columns='method', values='metric_value')
        st.dataframe(pivot, use_container_width=True)

    elif plot_type == 'line':
        # Line plot for percentile-based metrics
        fig = go.Figure()

        for method in context.methods_to_display:
            # For comparison metrics, skip the baseline method
            if is_comparison and method == context.baseline_method:
                continue

            if method not in context.method_stats:
                continue

            method_data = combined[combined['method'] == get_method_display_name(method)]
            if not method_data.empty:
                fig.add_trace(go.Scatter(
                    x=method_data['percentile'],
                    y=method_data['metric_value'],
                    mode='lines',
                    name=get_method_display_name(method),
                    line=dict(width=2)
                ))

        fig.update_layout(
            title=f"{metric_info['name']} ({context.n_episodes} episodes)",
            xaxis_title="Percentile",
            yaxis_title=metric_info['name'],
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)


def plot_metric_evolution(metadata_df, metric_key, methods, baseline_method, n_episodes_values, epsilon, dataset_type, n_buckets, temporal_p=0.2, adjust_constant=False):
    """Create evolution plot across n_episodes.

    Args:
        metadata_df: DataFrame with experiment metadata
        metric_key: Metric to compute
        methods: List of methods to display
        baseline_method: Baseline method name (for comparison metrics)
        n_episodes_values: List of n_episodes values to plot
        epsilon: Small value added before taking log
        dataset_type: 'full', 'differences', or 'temporal'
        n_buckets: Number of buckets for decile-based metrics
        temporal_p: Geometric distribution parameter for temporal differences
        adjust_constant: If True, add constant so mean(predictions) = mean(ground_truth)
    """
    metric_info = METRICS[metric_key]
    is_comparison = metric_info['is_comparison']

    # Collect summary statistics for each method
    summary_records = []

    # Process one n_episodes at a time
    for n_ep in n_episodes_values:
        # Compute ground truth stats once per n_episodes (for ground_truth_error metric)
        ground_truth_stats = None
        if metric_key == 'ground_truth_error':
            # Get results_dir from any method at this n_episodes
            any_method_row = metadata_df[metadata_df['n_episodes'] == n_ep]
            if not any_method_row.empty:
                predictions_path = any_method_row.iloc[0]['predictions_path']
                from pathlib import Path
                results_dir = str(Path(predictions_path).parent.parent.parent)
                ground_truth_stats = compute_ground_truth_stats(results_dir, dataset_type=dataset_type,
                                                                s1_proportion=0.9, seed=42,
                                                                temporal_p=temporal_p)

        # Load stats for all methods at this n_episodes
        method_stats_dict = {}
        for method in methods:
            method_row = metadata_df[(metadata_df['method'] == method) &
                                    (metadata_df['n_episodes'] == n_ep)]
            if not method_row.empty:
                method_stats_dict[method] = compute_stats_from_predictions(
                    method_row.iloc[0]['predictions_path'],
                    n_ep,
                    dataset_type=dataset_type,
                    temporal_p=temporal_p,
                    adjust_constant=adjust_constant
                )

        if not method_stats_dict:
            continue

        # Create MetricContext for this n_episodes
        context = MetricContext(
            method_stats=method_stats_dict,
            baseline_method=baseline_method,
            methods_to_display=methods,
            ground_truth_stats=ground_truth_stats,
            n_episodes=n_ep,
            epsilon=epsilon,
            n_buckets=n_buckets,
            dataset_type=dataset_type,
            temporal_p=temporal_p
        )

        # Process each method for this n_episodes
        for method in methods:
            # For comparison metrics, skip the baseline method
            if is_comparison and method == baseline_method:
                continue

            if method not in method_stats_dict:
                continue

            # Compute metric using context
            compute_fn = metric_info['compute_fn']
            metric_df = compute_fn(context, method)

            # Store summary statistics
            if not metric_df.empty:
                summary_records.append({
                    'method': get_method_display_name(method),
                    'n_episodes': n_ep,
                    'mean': metric_df['metric_value'].mean(),
                    'std': metric_df['metric_value'].std()
                })

        # Clean up after processing this n_episodes
        del method_stats_dict, context

    if not summary_records:
        st.warning("No data available")
        return

    summary = pd.DataFrame(summary_records)

    # Create plot
    fig = go.Figure()

    for method in methods:
        # For comparison metrics, skip the baseline method
        if is_comparison and method == baseline_method:
            continue
        method_display = get_method_display_name(method)
        method_data = summary[summary['method'] == method_display]
        if not method_data.empty:
            fig.add_trace(go.Scatter(
                x=method_data['n_episodes'],
                y=method_data['mean'],
                error_y=dict(type='data', array=method_data['std']),
                mode='lines+markers',
                name=method_display,
                marker=dict(size=10)
            ))

    if metric_info['reference_line'] is not None:
        fig.add_hline(
            y=metric_info['reference_line'], line_dash="dash", line_color="red",
            annotation_text=metric_info['reference_label']
        )

    if is_comparison:
        title = f"{metric_info['name']} vs Training Data Size (baseline: {get_method_display_name(baseline_method)})"
    else:
        title = f"{metric_info['name']} vs Training Data Size"

    fig.update_layout(
        title=title,
        xaxis_title="Training Episodes",
        yaxis_title=f"Mean {metric_info['name']} (± std)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show data table"):
        st.dataframe(summary, use_container_width=True)
