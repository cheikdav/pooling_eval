"""Paired states tab: evaluate estimators on state pairs with ground truth CIs."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.dashboard.data import (
    MethodPredictions, GroundTruthStore, PairedEval, get_gt_store,
    load_paired_predictions, load_paired_ground_truth,
)
from src.dashboard.metrics import bootstrap_metric_stderr, METRICS
from src.dashboard.plotting import plot_violin, plot_evolution, COLORS


def _load_paired_for_methods(
    entries_df: pd.DataFrame,
    n_episodes: int,
    methods: list,
    gt_store: GroundTruthStore,
    mode: str,
) -> dict:
    """Load paired MethodPredictions for each method at a given n_episodes."""
    predictions = {}
    for method in methods:
        row = entries_df[
            (entries_df["method"] == method) & (entries_df["n_episodes"] == n_episodes)
        ]
        if row.empty:
            continue
        row = row.iloc[0]
        paired_path = row.get("paired_predictions_path")
        if paired_path is None:
            continue
        mp = load_paired_predictions(
            paired_path, row["data_dir"], gt_store, method, n_episodes, mode=mode,
        )
        if mp is not None:
            predictions[method] = mp
    return predictions


def _plot_scatter_with_ci(
    gt: PairedEval,
    predictions: dict,
    mode: str,
    title: str = "",
) -> go.Figure:
    """Scatter plot: predicted value vs ground truth with CI error bars."""
    fig = go.Figure()

    if mode == "all":
        gt_vals = np.concatenate([gt.s1_mean, gt.s2_mean])
        ci_lower = np.concatenate([gt.s1_ci_lower, gt.s2_ci_lower])
        ci_upper = np.concatenate([gt.s1_ci_upper, gt.s2_ci_upper])
    else:
        gt_vals = gt.diff_mean
        ci_lower = gt.diff_ci_lower
        ci_upper = gt.diff_ci_upper

    # Perfect prediction line
    all_vals = gt_vals
    margin = (all_vals.max() - all_vals.min()) * 0.05
    line_range = [all_vals.min() - margin, all_vals.max() + margin]
    fig.add_trace(go.Scatter(
        x=line_range, y=line_range, mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Perfect", showlegend=False,
    ))

    for i, (method, mp) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(
            x=gt_vals, y=mp.mean,
            mode="markers", name=method,
            marker=dict(color=COLORS[i % len(COLORS)], size=5, opacity=0.7),
            error_x=dict(
                type="data",
                symmetric=False,
                array=ci_upper - gt_vals,
                arrayminus=gt_vals - ci_lower,
                color="rgba(255,255,255,0.2)",
                thickness=1,
            ),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Ground Truth (with 95% CI)",
        yaxis_title="Predicted (mean across batches)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def render(entries_df: pd.DataFrame, methods: list, baseline_method: str, params: dict):
    gt_store = get_gt_store()

    # Check if paired data exists
    sample_row = entries_df.iloc[0] if not entries_df.empty else None
    if sample_row is None:
        st.warning("No data available.")
        return

    has_paired = sample_row.get("paired_predictions_path") is not None
    if not has_paired:
        st.info("No paired predictions available for this experiment.")
        return

    # Mode selection
    mode = st.radio("Mode", ["All States", "Differences (s1-s2)"], horizontal=True, key="paired_mode")
    mode_key = {"All States": "all", "Differences (s1-s2)": "diff"}[mode]

    n_episodes_values = sorted(entries_df["n_episodes"].unique())
    n_episodes = st.select_slider(
        "Training episodes", options=n_episodes_values,
        value=n_episodes_values[-1], key="paired_n_episodes",
    )

    predictions = _load_paired_for_methods(entries_df, n_episodes, methods, gt_store, mode_key)
    if not predictions:
        st.warning("No paired predictions found.")
        return

    # Get ground truth (use base key for CI fields, mode key for returns)
    sample_mp = next(iter(predictions.values()))
    gt = gt_store.get(sample_mp.ground_truth_key)
    if not isinstance(gt, PairedEval):
        st.error("Ground truth is not PairedEval type.")
        return
    # Base GT has all CI fields
    data_dir = entries_df.iloc[0]["data_dir"]
    base_gt = gt_store.get(f"{data_dir}:paired")

    # --- Ground truth summary ---
    st.subheader("Ground Truth Statistics")
    if mode_key == "all":
        gt_vals = np.concatenate([base_gt.s1_mean, base_gt.s2_mean])
        ci_widths = np.concatenate([
            base_gt.s1_ci_upper - base_gt.s1_ci_lower,
            base_gt.s2_ci_upper - base_gt.s2_ci_lower,
        ])
    else:
        gt_vals = base_gt.diff_mean
        ci_widths = base_gt.diff_ci_upper - base_gt.diff_ci_lower

    n_label = "States" if mode_key == "all" else "Pairs"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(n_label, len(gt_vals))
    col2.metric("Mean", f"{gt_vals.mean():.2f}")
    col3.metric("Std", f"{gt_vals.std():.2f}")
    col4.metric("Avg CI Width", f"{ci_widths.mean():.2f}")

    # --- Scatter plot ---
    st.subheader("Predictions vs Ground Truth")
    fig = _plot_scatter_with_ci(base_gt, predictions, mode_key,
                                title=f"Predictions vs GT ({n_episodes} episodes, {mode})")
    st.plotly_chart(fig, use_container_width=True)

    # --- Per-pair metrics ---
    st.subheader("Per-Pair Metric Distributions")

    metric_key = st.selectbox("Metric", ["variance", "bias", "bias_squared", "mse"],
                              format_func=lambda k: METRICS[k].name, key="paired_metric")
    metric = METRICS[metric_key]

    metric_values = {}
    for method, mp in predictions.items():
        metric_values[method] = metric.fn(mp, gt_store)

    fig = plot_violin(metric_values, title=f"{metric.name} ({n_episodes} episodes)",
                      x_label=metric.name)
    st.plotly_chart(fig, use_container_width=True)

    # Stats table
    rows = []
    for method, vals in metric_values.items():
        rows.append({
            "Method": method,
            "Mean": np.nanmean(vals),
            "Median": np.nanmedian(vals),
            "Std": np.nanstd(vals),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Evolution ---
    st.subheader("Evolution Across Training Sizes")

    records = []
    for n_ep in n_episodes_values:
        preds = _load_paired_for_methods(entries_df, n_ep, methods, gt_store, mode_key)
        for method, mp in preds.items():
            vals = metric.fn(mp, gt_store)
            mean_val = np.nanmean(vals)
            stderr = bootstrap_metric_stderr(mp, gt_store, metric.fn)
            records.append({
                "method": method, "n_episodes": n_ep,
                "mean": mean_val, "stderr": stderr,
            })

    if records:
        fig = plot_evolution(records, title=f"{metric.name} vs Training Size",
                             y_label=metric.name)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Data table"):
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
