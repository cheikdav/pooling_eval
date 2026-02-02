"""Tab for paired state evaluation with ground truth CIs."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from common import get_method_display_name


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

    # Get results directory from first row
    first_row = filtered_for_n_ep.iloc[0]
    results_dir = first_row['results_dir']

    # Load paired states ground truth
    from pathlib import Path
    import numpy as np

    paired_states_file = Path(results_dir).parent / "data" / "paired_states.npz"

    if not paired_states_file.exists():
        st.warning(f"No paired state data found. Generate it with: `uv run -m src.generate_data --config <config> --generate-paired`")
        return

    paired_data = np.load(paired_states_file, allow_pickle=True)

    # Load predictions from each method
    predictions_data = {}
    for _, row in filtered_for_n_ep.iterrows():
        if row['method'] in methods:
            method = row['method']
            predictions_path = Path(row['predictions_path'])

            # Load predictions
            pred_df = pd.read_parquet(predictions_path)

            # Get unique states from the predictions (we need to match with paired states)
            # For now, we'll need to predict on paired states specifically
            # This requires modifying evaluate.py, but let's show what we can with current data

            predictions_data[method] = pred_df

    if not predictions_data:
        st.error("No prediction data available")
        return

    # Check if we have paired predictions (from evaluate.py update we'll do later)
    # For now, just show the ground truth statistics

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

    st.info("**Note:** Prediction evaluation will be available after implementing paired state predictions in evaluate.py")


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

    st.info("**Note:** Prediction evaluation will be available after implementing paired state predictions in evaluate.py")
