"""Tab for visualizing value predictions along episode trajectories."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from common import get_method_display_name, load_episode_data, load_predictions_for_trajectory


def render_tab(filtered_metadata, methods, baseline_method):
    """Render the trajectory visualization tab.

    Args:
        filtered_metadata: DataFrame with experiment metadata
        methods: List of methods to display
        baseline_method: Baseline method name (not used here, but kept for consistency)
    """
    st.header("📈 Episode Trajectory Analysis")
    st.markdown("Visualize how value predictions change along episode trajectories")

    if filtered_metadata.empty:
        st.error("No data available")
        return

    # Get experiment path (assume all rows have same experiment)
    experiment_path = Path(filtered_metadata.iloc[0]['predictions_path']).parents[2]

    # Load episode data using shared loader
    episode_data = load_episode_data(str(experiment_path))

    if episode_data is None:
        st.error(f"Evaluation batch not found at {experiment_path}/data/batch_eval.npz")
        return

    n_eval_episodes = len(episode_data['observations'])

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

    # Get episode info
    episode_obs = episode_data['observations'][selected_episode]
    episode_length = len(episode_obs)
    episode_rewards = episode_data['rewards'][selected_episode] if 'rewards' in episode_data else None
    episode_return = episode_data['episode_returns'][selected_episode] if 'episode_returns' in episode_data else None

    # Show episode info
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Episode Length", episode_length)
    with info_col2:
        if episode_return is not None:
            st.metric("Episode Return", f"{episode_return:.2f}")
    with info_col3:
        if episode_rewards is not None:
            st.metric("Mean Reward", f"{episode_rewards.mean():.2f}")

    st.markdown("---")

    # Load predictions for each method using shared loader
    fig = go.Figure()

    for method in selected_methods:
        # Filter metadata for this method and n_episodes
        method_row = filtered_metadata[
            (filtered_metadata['method'] == method) &
            (filtered_metadata['n_episodes'] == selected_n_ep)
        ]

        if method_row.empty:
            continue

        # Load predictions using shared function
        predictions_path = method_row.iloc[0]['predictions_path']
        pred_df = load_predictions_for_trajectory(predictions_path)

        # Filter to this episode
        episode_preds = pred_df[pred_df['episode_idx'] == selected_episode].copy()

        # Sort by state_idx to maintain order
        episode_preds = episode_preds.sort_values('state_idx')

        if len(episode_preds) != episode_length:
            st.warning(f"Warning: {method} has {len(episode_preds)} predictions but episode has {episode_length} states")
            continue

        # Create step indices (0, 1, 2, ...)
        steps = list(range(len(episode_preds)))
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

    st.plotly_chart(fig, use_container_width=True)

    # Optional: Show rewards overlay
    if episode_rewards is not None:
        with st.expander("Show rewards along trajectory"):
            fig_rewards = go.Figure()

            fig_rewards.add_trace(go.Bar(
                x=list(range(episode_length)),
                y=episode_rewards,
                name="Reward",
                marker_color='lightblue'
            ))

            fig_rewards.update_layout(
                title="Rewards Along Episode",
                xaxis_title="Step in Episode",
                yaxis_title="Reward",
                height=300
            )

            st.plotly_chart(fig_rewards, use_container_width=True)

    # Show statistics
    with st.expander("Show statistics"):
        stats_records = []

        for method in selected_methods:
            method_row = filtered_metadata[
                (filtered_metadata['method'] == method) &
                (filtered_metadata['n_episodes'] == selected_n_ep)
            ]

            if method_row.empty:
                continue

            predictions_path = method_row.iloc[0]['predictions_path']
            pred_df = load_predictions_for_trajectory(predictions_path)
            episode_preds = pred_df[pred_df['episode_idx'] == selected_episode]

            if not episode_preds.empty:
                values = episode_preds['mean_value'].values
                stats_records.append({
                    'Method': get_method_display_name(method),
                    'Mean': f"{values.mean():.4f}",
                    'Std': f"{values.std():.4f}",
                    'Min': f"{values.min():.4f}",
                    'Max': f"{values.max():.4f}",
                    'Range': f"{values.max() - values.min():.4f}"
                })

        if stats_records:
            stats_df = pd.DataFrame(stats_records)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
