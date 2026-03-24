"""Trajectory tab: visualize value predictions along episode timesteps."""

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.data import get_gt_store, load_method_predictions, apply_constant_adjustment
from src.dashboard.plotting import plot_trajectory


def render(entries_df: pd.DataFrame, methods: list, baseline_method: str, params: dict):
    gt_store = get_gt_store()

    n_episodes_values = sorted(entries_df["n_episodes"].unique())
    n_episodes = st.select_slider("Training episodes", options=n_episodes_values, value=n_episodes_values[-1], key="trajectory_n_episodes")

    # Load ground truth for episode info
    gt_key = entries_df["ground_truth_path"].dropna().iloc[0] if not entries_df.empty else None
    gt = gt_store.get(gt_key) if gt_key else None

    if gt is None:
        st.warning("No ground truth data available.")
        return

    # Episode selector
    unique_episodes = np.unique(gt.episode_idx)
    episode_idx = st.selectbox("Episode", unique_episodes)

    # Episode mask
    ep_mask = gt.episode_idx == episode_idx
    ep_timesteps = gt.timestep[ep_mask]
    ep_returns = gt.returns[ep_mask]
    ep_length = gt.episode_length[ep_mask][0]
    ep_truncated = gt.is_truncated[ep_mask][0]

    st.caption(f"Episode {episode_idx}: length={ep_length}, truncated={ep_truncated}")

    # Build ground truth series
    gt_series = pd.Series(ep_returns, index=ep_timesteps, name="Ground Truth")

    # Critic values
    critic_series = None
    if gt.critic_value is not None:
        critic_series = pd.Series(gt.critic_value[ep_mask], index=ep_timesteps)

    # Load predictions for each method
    traces = {}
    stds = {}
    ep_state_indices = np.where(ep_mask)[0]
    for method in methods:
        row = entries_df[
            (entries_df["method"] == method) & (entries_df["n_episodes"] == n_episodes)
        ]
        if row.empty:
            continue
        row = row.iloc[0]

        mp = load_method_predictions(
            row["predictions_path"], row["ground_truth_path"], method, n_episodes,
        )
        if params.get("adjust_constant", False):
            mp = apply_constant_adjustment(mp, gt_store)

        traces[method] = pd.Series(mp.mean[ep_state_indices], index=ep_timesteps)
        stds[method] = pd.Series(np.sqrt(mp.variance[ep_state_indices]), index=ep_timesteps)

    if not traces:
        st.warning("No predictions found for selected methods.")
        return

    fig = plot_trajectory(
        traces, ground_truth=gt_series, critic_values=critic_series,
        stds=stds,
        title=f"Episode {episode_idx} — {n_episodes} training episodes",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats table
    with st.expander("Statistics"):
        rows = []
        for method, series in traces.items():
            rows.append({
                "Method": method,
                "Mean": series.mean(),
                "Std": series.std(),
                "Min": series.min(),
                "Max": series.max(),
                "Range": series.max() - series.min(),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
