"""Shared plotting utilities for the dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from metrics import METRICS, compute_metric
from common import get_method_display_name, compute_stats_from_predictions


def plot_metric_for_single_episodes(stats_dict, metric_key, methods, n_episodes, baseline_method, epsilon, n_buckets):
    """Plot metric for a single n_episodes value.

    Args:
        stats_dict: Dict mapping method names to DataFrames
        metric_key: Metric to compute
        methods: List of methods to display
        n_episodes: Number of episodes (for display)
        baseline_method: Baseline method name (for comparison metrics)
        epsilon: Small value added before taking log
        n_buckets: Number of buckets for decile-based metrics
    """
    metric_info = METRICS[metric_key]
    plot_type = metric_info['plot_type']
    is_comparison = metric_info['is_comparison']

    if is_comparison and baseline_method not in stats_dict:
        st.error(f"Baseline method {baseline_method} not found")
        return

    baseline_stats = stats_dict.get(baseline_method) if is_comparison else None

    # Compute metrics for each method
    metric_list = []
    for method in methods:
        # For comparison metrics, skip the baseline method
        if is_comparison and method == baseline_method:
            continue

        if method not in stats_dict:
            continue

        method_stats = stats_dict[method]
        metric_df = compute_metric(baseline_stats, method_stats, metric_key, epsilon=epsilon, n_buckets=n_buckets)
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
                title=f"{metric_info['name']} Distribution ({n_episodes} episodes)",
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
                st.markdown(f"**Statistics** (vs {get_method_display_name(baseline_method)})")
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
            title=f"{metric_info['name']} ({n_episodes} episodes, {n_buckets} buckets)",
            labels={
                'decile': f'Value Bucket (0=lowest, {n_buckets-1}=highest)',
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

        for method in methods:
            # For comparison metrics, skip the baseline method
            if is_comparison and method == baseline_method:
                continue

            if method not in stats_dict:
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
            title=f"{metric_info['name']} ({n_episodes} episodes)",
            xaxis_title="Percentile",
            yaxis_title=metric_info['name'],
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)


def plot_metric_evolution(metadata_df, metric_key, methods, baseline_method, n_episodes_values, epsilon, dataset_type, n_buckets):
    """Create evolution plot across n_episodes.

    Args:
        metadata_df: DataFrame with experiment metadata
        metric_key: Metric to compute
        methods: List of methods to display
        baseline_method: Baseline method name (for comparison metrics)
        n_episodes_values: List of n_episodes values to plot
        epsilon: Small value added before taking log
        dataset_type: 'full' or 'differences'
        n_buckets: Number of buckets for decile-based metrics
    """
    metric_info = METRICS[metric_key]
    is_comparison = metric_info['is_comparison']

    # Collect summary statistics for each method
    summary_records = []

    # Process one n_episodes at a time
    for n_ep in n_episodes_values:
        # Load baseline stats for this n_episodes (only needed for comparison metrics)
        if is_comparison:
            baseline_row = metadata_df[(metadata_df['method'] == baseline_method) &
                                       (metadata_df['n_episodes'] == n_ep)]

            if baseline_row.empty:
                continue

            baseline_stats = compute_stats_from_predictions(
                baseline_row.iloc[0]['predictions_path'],
                n_ep,
                dataset_type=dataset_type
            )
        else:
            baseline_stats = None

        # Process each method for this n_episodes
        for method in methods:
            # For comparison metrics, skip the baseline method
            if is_comparison and method == baseline_method:
                continue

            method_row = metadata_df[(metadata_df['method'] == method) &
                                    (metadata_df['n_episodes'] == n_ep)]

            if method_row.empty:
                continue

            # Load method stats
            method_stats = compute_stats_from_predictions(
                method_row.iloc[0]['predictions_path'],
                n_ep,
                dataset_type=dataset_type
            )

            # Compute metric for this method vs baseline
            metric_df = compute_metric(baseline_stats, method_stats, metric_key, epsilon=epsilon, n_buckets=n_buckets)

            # Store summary statistics
            if not metric_df.empty:
                summary_records.append({
                    'method': get_method_display_name(method),
                    'n_episodes': n_ep,
                    'mean': metric_df['metric_value'].mean(),
                    'std': metric_df['metric_value'].std()
                })

            # Discard method stats and metric immediately
            del method_stats, metric_df

        # Discard baseline stats after processing all methods for this n_episodes
        if baseline_stats is not None:
            del baseline_stats

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
