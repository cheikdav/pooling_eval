"""Tab for visualizing value predictions along episode trajectories."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from common import get_method_display_name, load_predictions_for_trajectory, load_ground_truth_returns
from pathlib import Path


def render_tab(filtered_metadata, methods, baseline_method, adjust_constant=False):
    """Render the trajectory visualization tab.

    Args:
        filtered_metadata: DataFrame with experiment metadata
        methods: List of methods to display
        baseline_method: Baseline method name (not used here, but kept for consistency)
        adjust_constant: If True, add constant so mean(predictions) = mean(ground_truth)
    """
    st.header("📈 Episode Trajectory Analysis")
    st.markdown("Visualize how value predictions change along episode trajectories")

    if filtered_metadata.empty:
        st.error("No data available")
        return

    # Load predictions from first method to get episode info
    first_row = filtered_metadata.iloc[0]
    first_predictions_path = first_row['predictions_path']
    first_pred_df = load_predictions_for_trajectory(first_predictions_path, adjust_constant=adjust_constant, gamma=first_row.get('policy_gamma'), truncation_coefficient=first_row.get('truncation_coefficient', 5.0))
    n_eval_episodes = first_pred_df['episode_idx'].nunique()

    st.markdown("---")

    # Selection controls in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        # n_episodes selector
        n_episodes_values = sorted(filtered_metadata['n_episodes'].unique())
        selected_n_ep = st.selectbox(
            "Training episodes:",
            n_episodes_values,
            format_func=lambda x: f"{x} episodes",
            key="traj_n_ep"
        )

    with col2:
        # Episode selector
        selected_episode = st.selectbox(
            "Episode index:",
            range(n_eval_episodes),
            format_func=lambda x: f"Episode {x}",
            key="traj_episode"
        )

    with col3:
        # Methods to plot
        selected_methods = st.multiselect(
            "Methods to plot:",
            methods,
            default=methods,
            key="traj_methods",
            format_func=get_method_display_name
        )

    if not selected_methods:
        st.warning("Please select at least one method")
        return

    st.markdown("---")

    # Get results path (go up from predictions path to results dir)
    first_predictions_path = filtered_metadata[
        (filtered_metadata['method'] == selected_methods[0]) &
        (filtered_metadata['n_episodes'] == selected_n_ep)
    ].iloc[0]['predictions_path']
    results_path = Path(first_predictions_path).parents[2]

    # Load ground truth returns
    ground_truth_df = load_ground_truth_returns(str(results_path))

    # Show info about ground truth
    if ground_truth_df is None:
        expected_path = Path(results_path) / "ground_truth" / "ground_truth_returns.parquet"
        st.info(f"Ground truth file not found at: {expected_path}. Run evaluation to generate it.")
    else:
        st.success(f"Loaded ground truth with {len(ground_truth_df)} states from {ground_truth_df['episode_idx'].nunique()} episodes")

    # Get gamma and truncation_coefficient from metadata
    first_method_row = filtered_metadata[
        (filtered_metadata['method'] == selected_methods[0]) &
        (filtered_metadata['n_episodes'] == selected_n_ep)
    ].iloc[0]
    gamma = first_method_row.get('policy_gamma', 0.99)
    truncation_coefficient = first_method_row.get('truncation_coefficient', 5.0)

    # Load predictions for each method
    fig = go.Figure()
    episode_length = None
    is_truncated = None

    for method in selected_methods:
        # Filter metadata for this method and n_episodes
        method_row = filtered_metadata[
            (filtered_metadata['method'] == method) &
            (filtered_metadata['n_episodes'] == selected_n_ep)
        ]

        if method_row.empty:
            continue

        # Load predictions using shared function
        row = method_row.iloc[0]
        pred_df = load_predictions_for_trajectory(row['predictions_path'], adjust_constant=adjust_constant, gamma=row.get('policy_gamma'), truncation_coefficient=row.get('truncation_coefficient', 5.0))

        # Filter to this episode
        episode_preds = pred_df[pred_df['episode_idx'] == selected_episode].copy()

        if episode_preds.empty:
            continue

        # Get episode metadata from first method
        if episode_length is None:
            episode_length = len(episode_preds)
            # Check if episode is truncated
            if 'is_truncated' in episode_preds.columns:
                is_truncated = episode_preds['is_truncated'].iloc[0]

        # Use step_in_episode from the dataframe
        steps = episode_preds['step_in_episode'].values
        values = episode_preds['mean_value'].values

        # Add trace
        fig.add_trace(go.Scatter(
            x=steps,
            y=values,
            mode='lines+markers',
            name=get_method_display_name(method),
            line=dict(width=2),
            marker=dict(size=6)
        ))

    # Add ground truth if available
    if ground_truth_df is not None:
        ground_truth_episode = ground_truth_df[ground_truth_df['episode_idx'] == selected_episode].copy()
        if not ground_truth_episode.empty:
            # Also get truncation info from ground truth if not already captured
            if is_truncated is None and 'is_truncated' in ground_truth_episode.columns:
                is_truncated = ground_truth_episode['is_truncated'].iloc[0]
            if episode_length is None and 'episode_length' in ground_truth_episode.columns:
                episode_length = ground_truth_episode['episode_length'].iloc[0]

            fig.add_trace(go.Scatter(
                x=ground_truth_episode['step_in_episode'].values,
                y=ground_truth_episode['ground_truth_return'].values,
                mode='lines',
                name='Ground Truth',
                line=dict(width=3, color='white', dash='dash')
            ))

            # Add critic value from trained policy if available
            if 'critic_value' in ground_truth_episode.columns:
                fig.add_trace(go.Scatter(
                    x=ground_truth_episode['step_in_episode'].values,
                    y=ground_truth_episode['critic_value'].values,
                    mode='lines',
                    name='Policy Critic',
                    line=dict(width=2, color='yellow', dash='dot')
                ))

    # Add vertical line showing truncation cutoff for truncated episodes
    if is_truncated and episode_length is not None:
        n_discard = int(truncation_coefficient / (1 - gamma))
        cutoff_step = episode_length - n_discard
        if cutoff_step > 0:
            fig.add_vline(
                x=cutoff_step,
                line_dash="dot",
                line_color="red",
                line_width=2,
                annotation_text=f"Truncation cutoff (C/(1-γ)={n_discard})",
                annotation_position="top"
            )

    # Update layout
    fig.update_layout(
        title=f"Value Predictions Along Episode {selected_episode} ({selected_n_ep} training episodes)",
        xaxis_title="Step in Episode",
        yaxis_title="Predicted Value",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, width='stretch')

    # Show episode info after plot
    col1, col2 = st.columns(2)
    with col1:
        if episode_length is not None:
            st.metric("Episode Length", episode_length)
    with col2:
        if is_truncated is not None:
            status = "Yes (time limit)" if is_truncated else "No (natural end)"
            st.metric("Truncated", status)

    st.markdown("---")

    # Show statistics
    with st.expander("Show value statistics"):
        stats_records = []

        for method in selected_methods:
            method_row = filtered_metadata[
                (filtered_metadata['method'] == method) &
                (filtered_metadata['n_episodes'] == selected_n_ep)
            ]

            if method_row.empty:
                continue

            row = method_row.iloc[0]
            pred_df = load_predictions_for_trajectory(row['predictions_path'], adjust_constant=adjust_constant, gamma=row.get('policy_gamma'), truncation_coefficient=row.get('truncation_coefficient', 5.0))
            episode_preds = pred_df[pred_df['episode_idx'] == selected_episode]

            if not episode_preds.empty:
                values = episode_preds['mean_value'].values
                stats_records.append({
                    'Method': get_method_display_name(method),
                    'Mean': f"{values.mean():.4f}",
                    'Std': f"{values.std():.4f}",
                    'Min': f"{values.min():.4f}",
                    'Max': f"{values.max():.4f}",
                    'Range': f"{values.max() - values.min():.4f}",
                    'Num Steps': len(values)
                })

        if stats_records:
            stats_df = pd.DataFrame(stats_records)
            st.dataframe(stats_df, width='stretch', hide_index=True)
