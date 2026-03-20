"""Shared plotting utilities for the dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from metrics import METRICS
from common import (get_method_display_name, compute_stats_from_predictions, compute_all_batch_constants,
                    compute_ground_truth_stats, MetricContext,
                    BOOTSTRAP_SUPPORTED_METRICS, compute_bootstrap_stderr_evolution)


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
            st.plotly_chart(fig, width='stretch')

        with col2:
            if is_comparison:
                st.markdown(f"**Statistics** (vs {get_method_display_name(context.baseline_method)})")
            else:
                st.markdown("**Statistics**")
            summary = combined.groupby('method')['metric_value'].agg(
                mean='mean', std='std'
            ).reset_index()
            st.dataframe(summary, width='stretch', hide_index=True)

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
        st.plotly_chart(fig, width='stretch')

        # Show statistics table
        st.markdown("**Statistics by Bucket**")
        pivot = combined.pivot(index='decile', columns='method', values='metric_value')
        st.dataframe(pivot, width='stretch')

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

        st.plotly_chart(fig, width='stretch')


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
    use_bootstrap = metric_key in BOOTSTRAP_SUPPORTED_METRICS

    # Collect summary statistics for each method
    summary_records = []

    # Process one n_episodes at a time
    for n_ep in n_episodes_values:
        # Get metadata rows for this n_episodes
        rows_at_n_ep = {
            method: metadata_df[(metadata_df['method'] == method) & (metadata_df['n_episodes'] == n_ep)]
            for method in methods
        }
        available_methods = {m for m, r in rows_at_n_ep.items() if not r.empty}
        if not available_methods:
            continue

        any_row = rows_at_n_ep[next(iter(available_methods))].iloc[0]
        from pathlib import Path
        results_dir = str(Path(any_row['predictions_path']).parent.parent.parent)
        gamma = any_row.get('policy_gamma', 0.99)
        truncation_coefficient = any_row.get('truncation_coefficient', 5.0)

        # Compute ground truth stats once per n_episodes (for error/mse metrics)
        ground_truth_stats = None
        if metric_key in ('ground_truth_error', 'ground_truth_error_squared', 'mse', 'bias_variance_decomposition'):
            ground_truth_stats = compute_ground_truth_stats(results_dir, dataset_type=dataset_type,
                                                            s1_proportion=0.9, seed=42,
                                                            temporal_p=temporal_p,
                                                            gamma=gamma,
                                                            truncation_coefficient=truncation_coefficient,
                                                            filter_truncation=True)

        # Load stats for all methods at this n_episodes
        method_stats_dict = {}
        for method in available_methods:
            row = rows_at_n_ep[method].iloc[0]
            method_stats_dict[method] = compute_stats_from_predictions(
                row['predictions_path'], n_ep,
                dataset_type=dataset_type, temporal_p=temporal_p,
                adjust_constant=adjust_constant,
                gamma=row.get('policy_gamma'),
                truncation_coefficient=row.get('truncation_coefficient', 5.0)
            )

        # Compute bootstrap stderrs for all methods at once (shared batch resampling)
        bootstrap_stderrs = {}
        if use_bootstrap:
            method_paths = tuple(
                (m, rows_at_n_ep[m].iloc[0]['predictions_path'])
                for m in available_methods
            )
            bootstrap_stderrs = compute_bootstrap_stderr_evolution(
                method_paths, metric_key, results_dir,
                dataset_type, 0.9, 42, temporal_p,
                adjust_constant, gamma, truncation_coefficient,
                epsilon, baseline_method
            )

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
            if is_comparison and method == baseline_method:
                continue
            if method not in method_stats_dict:
                continue

            compute_fn = metric_info['compute_fn']
            metric_df = compute_fn(context, method)

            if not metric_df.empty:
                n_states = len(metric_df)
                if use_bootstrap and method in bootstrap_stderrs:
                    stderr = bootstrap_stderrs[method]
                else:
                    stderr = metric_df['metric_value'].std() / (n_states ** 0.5)
                summary_records.append({
                    'method': get_method_display_name(method),
                    'n_episodes': n_ep,
                    'mean': metric_df['metric_value'].mean(),
                    'stderr': stderr
                })

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
                error_y=dict(type='data', array=method_data['stderr']),
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
        yaxis_title=f"Mean {metric_info['name']} (± stderr)",
        height=500
    )

    st.plotly_chart(fig, width='stretch')

    with st.expander("Show data table"):
        st.dataframe(summary, width='stretch')
