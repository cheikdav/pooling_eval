"""Tab for paired state evaluation with ground truth CIs."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path

from common import get_method_display_name, sort_methods


def sort_predictions(predictions_data):
    """Sort predictions_data dict by METHOD_ORDER, unknown methods go last."""
    ordered = sort_methods(predictions_data.keys())
    return {m: predictions_data[m] for m in ordered}


def render_tab(filtered_metadata, methods, baseline_method):
    """Render the paired states tab.

    Args:
        filtered_metadata: DataFrame with experiment metadata
        methods: List of methods to display
        baseline_method: Baseline method name (not used here, passed for consistency)
    """
    st.header("🎯 Paired State Evaluation")

    # Mode selection
    mode = st.radio(
        "Evaluation Mode:",
        options=['full', 'difference'],
        format_func=lambda x: {
            'full': 'Full Dataset (Individual States)',
            'difference': 'Difference Dataset (V(s₁) - V(s₂))'
        }[x],
        help="Full: evaluate V(s) for each state independently | Difference: evaluate V(s₁) - V(s₂) for pairs"
    )

    st.markdown("---")

    # Training size selection
    n_episodes_values = sorted(filtered_metadata['n_episodes'].unique())
    selected_n_ep = st.selectbox(
        "Training data size:",
        n_episodes_values,
        format_func=lambda x: f"{x} episodes",
        key="paired_n_ep"
    )

    # Load paired state data
    filtered_for_n_ep = filtered_metadata[filtered_metadata['n_episodes'] == selected_n_ep]

    if filtered_for_n_ep.empty:
        st.error(f"No data for {selected_n_ep} episodes")
        return

    # Get results directory from predictions path
    first_row = filtered_for_n_ep.iloc[0]
    results_dir = Path(first_row['predictions_path']).parents[2]

    # Load paired states ground truth
    paired_states_file = results_dir.parent / "data" / "paired_states.npz"

    if not paired_states_file.exists():
        st.warning(f"No paired state data found. Generate it with: `uv run -m src.generate_data --config <config> --generate-paired`")
        return

    paired_data = np.load(paired_states_file, allow_pickle=True)

    # Load paired predictions from each method
    predictions_data = {}
    for _, row in filtered_for_n_ep.iterrows():
        if row['method'] in methods:
            method = row['method']
            predictions_path = Path(row['predictions_path'])

            # Try to load paired predictions
            paired_predictions_path = predictions_path.parent / "paired_predictions.parquet"

            if paired_predictions_path.exists():
                # Load paired predictions
                pred_df = pd.read_parquet(paired_predictions_path)
                predictions_data[method] = pred_df
            else:
                # No paired predictions available for this method
                pass

    if not predictions_data:
        st.warning("No paired prediction data available yet. Run evaluation to generate paired predictions: `uv run -m src.evaluate --config <config>`")
        st.info("Note: Regular evaluation batch predictions exist, but paired state predictions require a separate data generation step.")
        # Still show ground truth statistics even without predictions

    if mode == 'full':
        render_full_dataset_mode(paired_data, predictions_data, methods, selected_n_ep)
    else:
        render_difference_mode(paired_data, predictions_data, methods, selected_n_ep)


def render_full_dataset_mode(paired_data, predictions_data, methods, n_episodes):
    """Render full dataset mode: V(s) for each state independently."""

    st.subheader("Individual State Evaluation")
    st.markdown("Evaluates V(s) for each state independently against ground truth (mean ± CI)")

    # Extract ground truth for all states (s1 and s2 combined)
    n_pairs = len(paired_data['pair_indices'])

    # Combine s1 and s2 into single dataset
    all_gt_means = np.concatenate([paired_data['s1_mean'], paired_data['s2_mean']])
    all_gt_ci_lower = np.concatenate([paired_data['s1_ci_lower'], paired_data['s2_ci_lower']])
    all_gt_ci_upper = np.concatenate([paired_data['s1_ci_upper'], paired_data['s2_ci_upper']])

    # For now, show ground truth statistics
    st.markdown("### Ground Truth Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total States", len(all_gt_means))
    with col2:
        st.metric("Mean Value", f"{np.mean(all_gt_means):.2f}")
    with col3:
        avg_ci_width = np.mean(all_gt_ci_upper - all_gt_ci_lower)
        st.metric("Avg CI Width", f"{avg_ci_width:.2f}")

    # Histogram of ground truth values
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=all_gt_means,
        nbinsx=30,
        name="Ground Truth Values",
        marker_color='blue',
        opacity=0.7
    ))
    fig.update_layout(
        title=f"Distribution of Ground Truth Values ({n_episodes} episodes)",
        xaxis_title="V(s) - Ground Truth Mean",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # CI width distribution
    ci_widths = all_gt_ci_upper - all_gt_ci_lower
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ci_widths,
        nbinsx=30,
        name="CI Widths",
        marker_color='green',
        opacity=0.7
    ))
    fig.update_layout(
        title="Distribution of Confidence Interval Widths",
        xaxis_title="CI Width (95%)",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # If we have predictions, show evaluation metrics
    if predictions_data:
        predictions_data = sort_predictions(predictions_data)
        st.markdown("### Prediction Evaluation")

        # Compute metrics for each method
        metrics_data = []
        for method, pred_df in predictions_data.items():
            avg_pred = pred_df.groupby('pair_idx')[['s1_predicted', 's2_predicted']].mean()
            all_pred = np.concatenate([avg_pred['s1_predicted'].values, avg_pred['s2_predicted'].values])

            errors = all_pred - all_gt_means
            mse = np.mean(errors**2)
            mae = np.mean(np.abs(errors))

            metrics_data.append({
                'Method': get_method_display_name(method),
                'MSE': mse,
                'MAE': mae,
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.format({
            'MSE': '{:.4f}',
            'MAE': '{:.4f}',
        }), use_container_width=True)

        # Histograms: per-state MSE, squared bias, variance distributions
        def _per_state_metric(metric_name, pred_df, avg_pred):
            if metric_name == 'MSE':
                s1 = (avg_pred['s1_predicted'].values - paired_data['s1_mean']) ** 2
                s2 = (avg_pred['s2_predicted'].values - paired_data['s2_mean']) ** 2
            elif metric_name == 'Squared Bias':
                s1 = (avg_pred['s1_predicted'].values - paired_data['s1_mean']) ** 2
                s2 = (avg_pred['s2_predicted'].values - paired_data['s2_mean']) ** 2
            else:  # Variance
                s1 = pred_df.groupby('pair_idx')['s1_predicted'].var().fillna(0).values
                s2 = pred_df.groupby('pair_idx')['s2_predicted'].var().fillna(0).values
            return np.concatenate([s1, s2])

        for metric_name, title in [
            ('MSE', 'Per-State MSE Distribution'),
            ('Squared Bias', 'Per-State Squared Bias Distribution'),
            ('Variance', 'Per-State Variance Distribution'),
        ]:
            st.markdown(f"### {title}")
            hist_rows = []
            for method, pred_df in predictions_data.items():
                avg_pred = pred_df.groupby('pair_idx')[['s1_predicted', 's2_predicted']].mean()
                values = _per_state_metric(metric_name, pred_df, avg_pred)
                for v in values:
                    hist_rows.append({'Method': get_method_display_name(method), metric_name: v})
            hist_df = pd.DataFrame(hist_rows)
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.histogram(hist_df, x=metric_name, color='Method', nbins=40, opacity=0.7,
                                   barmode='overlay', title=f"{title} ({n_episodes} episodes)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Statistics**")
                summary = hist_df.groupby('Method')[metric_name].agg(mean='mean', std='std').reset_index()
                st.dataframe(summary.style.format({'mean': '{:.4f}', 'std': '{:.4f}'}),
                             use_container_width=True, hide_index=True)

        # Scatter plot: Predictions vs Ground Truth for each method
        st.markdown("### Predictions vs Ground Truth")

        for method, pred_df in predictions_data.items():
            avg_pred = pred_df.groupby('pair_idx')[['s1_predicted', 's2_predicted']].mean()
            all_pred = np.concatenate([avg_pred['s1_predicted'].values, avg_pred['s2_predicted'].values])

            fig = go.Figure()

            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=all_gt_means,
                y=all_pred,
                mode='markers',
                marker=dict(size=5, opacity=0.5),
                name=get_method_display_name(method)
            ))

            # Add diagonal line
            min_val = min(all_gt_means.min(), all_pred.min())
            max_val = max(all_gt_means.max(), all_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Prediction',
                showlegend=True
            ))

            fig.update_layout(
                title=f"{get_method_display_name(method)} - Predictions vs Ground Truth",
                xaxis_title="Ground Truth Mean",
                yaxis_title="Predicted Value",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def render_difference_mode(paired_data, predictions_data, methods, n_episodes):
    """Render difference mode: V(s₁) - V(s₂) for pairs."""

    st.subheader("Paired Difference Evaluation")
    st.markdown("Evaluates V(s₁) - V(s₂) for state pairs against ground truth (mean ± CI)")

    # Extract difference ground truth
    diff_means = paired_data['diff_mean']
    diff_ci_lower = paired_data['diff_ci_lower']
    diff_ci_upper = paired_data['diff_ci_upper']
    n_pairs = len(diff_means)

    # Ground truth statistics
    st.markdown("### Ground Truth Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pairs", n_pairs)
    with col2:
        st.metric("Mean Difference", f"{np.mean(diff_means):.2f}")
    with col3:
        st.metric("Std Difference", f"{np.std(diff_means):.2f}")
    with col4:
        avg_ci_width = np.mean(diff_ci_upper - diff_ci_lower)
        st.metric("Avg CI Width", f"{avg_ci_width:.2f}")

    # Histogram of ground truth differences
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=diff_means,
        nbinsx=30,
        name="Ground Truth Differences",
        marker_color='purple',
        opacity=0.7
    ))

    # Add reference line at 0
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Zero Difference"
    )

    fig.update_layout(
        title=f"Distribution of V(s₁) - V(s₂) Ground Truth ({n_episodes} episodes)",
        xaxis_title="V(s₁) - V(s₂) - Ground Truth Mean",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot: Individual state values
    st.markdown("### Individual State Values (s₁ vs s₂)")

    s1_means = paired_data['s1_mean']
    s2_means = paired_data['s2_mean']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s1_means,
        y=s2_means,
        mode='markers',
        marker=dict(
            size=8,
            color=diff_means,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="V(s₁)-V(s₂)")
        ),
        text=[f"Pair {i}<br>V(s₁)={s1:.2f}<br>V(s₂)={s2:.2f}<br>Diff={d:.2f}"
              for i, (s1, s2, d) in enumerate(zip(s1_means, s2_means, diff_means))],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Add diagonal line
    min_val = min(s1_means.min(), s2_means.min())
    max_val = max(s1_means.max(), s2_means.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='V(s₁) = V(s₂)',
        showlegend=True
    ))

    fig.update_layout(
        title="Ground Truth: V(s₁) vs V(s₂)",
        xaxis_title="V(s₁) - Ground Truth Mean",
        yaxis_title="V(s₂) - Ground Truth Mean",
        height=500,
        width=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # CI width distribution
    ci_widths = diff_ci_upper - diff_ci_lower
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ci_widths,
        nbinsx=30,
        name="CI Widths for Differences",
        marker_color='green',
        opacity=0.7
    ))
    fig.update_layout(
        title="Distribution of Confidence Interval Widths for Differences",
        xaxis_title="CI Width (95%) for V(s₁) - V(s₂)",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # If we have predictions, show evaluation metrics
    if predictions_data:
        predictions_data = sort_predictions(predictions_data)
        st.markdown("### Prediction Evaluation for Differences")

        # Compute metrics for each method
        metrics_data = []
        for method, pred_df in predictions_data.items():
            avg_pred = pred_df.groupby('pair_idx')['diff_predicted'].mean().values

            errors = avg_pred - diff_means
            mse = np.mean(errors**2)
            mae = np.mean(np.abs(errors))

            metrics_data.append({
                'Method': get_method_display_name(method),
                'MSE': mse,
                'MAE': mae,
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.format({
            'MSE': '{:.4f}',
            'MAE': '{:.4f}',
        }), use_container_width=True)

        # Histograms: per-pair MSE, squared bias, variance distributions
        def _per_pair_metric(metric_name, pred_df, avg_pred):
            if metric_name == 'MSE':
                return (avg_pred - diff_means) ** 2
            elif metric_name == 'Squared Bias':
                return (avg_pred - diff_means) ** 2
            else:  # Variance
                return pred_df.groupby('pair_idx')['diff_predicted'].var().fillna(0).values

        for metric_name, title in [
            ('MSE', 'Per-Pair MSE Distribution'),
            ('Squared Bias', 'Per-Pair Squared Bias Distribution'),
            ('Variance', 'Per-Pair Variance Distribution'),
        ]:
            st.markdown(f"### {title}")
            hist_rows = []
            for method, pred_df in predictions_data.items():
                avg_pred = pred_df.groupby('pair_idx')['diff_predicted'].mean().values
                values = _per_pair_metric(metric_name, pred_df, avg_pred)
                for v in values:
                    hist_rows.append({'Method': get_method_display_name(method), metric_name: v})
            hist_df = pd.DataFrame(hist_rows)
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.histogram(hist_df, x=metric_name, color='Method', nbins=40, opacity=0.7,
                                   barmode='overlay', title=f"{title} ({n_episodes} episodes)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Statistics**")
                summary = hist_df.groupby('Method')[metric_name].agg(mean='mean', std='std').reset_index()
                st.dataframe(summary.style.format({'mean': '{:.4f}', 'std': '{:.4f}'}),
                             use_container_width=True, hide_index=True)

        # Scatter plot: Predicted differences vs Ground Truth for each method
        st.markdown("### Predicted Differences vs Ground Truth")

        for method, pred_df in predictions_data.items():
            avg_pred = pred_df.groupby('pair_idx')['diff_predicted'].mean().values

            fig = go.Figure()

            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=diff_means,
                y=avg_pred,
                mode='markers',
                marker=dict(size=5, opacity=0.5),
                name=get_method_display_name(method)
            ))

            # Add diagonal line
            min_val = min(diff_means.min(), avg_pred.min())
            max_val = max(diff_means.max(), avg_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Prediction',
                showlegend=True
            ))

            # Add zero line
            fig.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
            fig.add_vline(x=0, line_dash="dot", line_color="red", opacity=0.5)

            fig.update_layout(
                title=f"{get_method_display_name(method)} - V(s₁) - V(s₂) Predictions vs Ground Truth",
                xaxis_title="Ground Truth Difference Mean",
                yaxis_title="Predicted Difference",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
