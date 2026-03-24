"""Metrics tab: absolute and comparison metrics analysis."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.dashboard.data import (
    MethodPredictions, GroundTruthStore, get_gt_store,
    load_method_predictions, apply_constant_adjustment, filter_states,
    to_temporal_diff,
)
from src.dashboard.metrics import METRICS, available_metrics, bootstrap_metric_stderr
from src.dashboard.plotting import plot_violin, plot_bar, plot_evolution, COLORS


def _load_predictions_for_methods(
    entries_df: pd.DataFrame,
    n_episodes: int,
    methods: list,
    gt_store: GroundTruthStore,
    params: dict,
) -> dict:
    """Load MethodPredictions for each method at a given n_episodes."""
    predictions = {}
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
        if params.get("dataset_type") == "temporal_diff":
            mp = to_temporal_diff(mp, gt_store, temporal_p=params.get("temporal_p", 0.2))
        predictions[method] = mp
    return predictions


def _compute_stats_table(metric_values: dict) -> pd.DataFrame:
    """Summary statistics table for metric values across methods."""
    rows = []
    for method, vals in metric_values.items():
        if isinstance(vals, pd.DataFrame):
            v = vals["metric_value"].values
        else:
            v = vals
        rows.append({
            "Method": method,
            "Mean": np.nanmean(v),
            "Median": np.nanmedian(v),
            "Std": np.nanstd(v),
            "Min": np.nanmin(v),
            "Max": np.nanmax(v),
        })
    return pd.DataFrame(rows)


def render(entries_df: pd.DataFrame, methods: list, baseline_method: str, params: dict):
    gt_store = get_gt_store()

    # Load one set of predictions to determine available metrics after transformation
    n_episodes_values = sorted(entries_df["n_episodes"].unique())
    sample_preds = _load_predictions_for_methods(
        entries_df, n_episodes_values[0], methods[:1], gt_store, params,
    )
    if not sample_preds:
        st.warning("No predictions found.")
        return
    sample_mp = next(iter(sample_preds.values()))
    avail = available_metrics(gt_store, sample_mp.ground_truth_key)

    # Separate absolute vs comparison
    absolute = {k: v for k, v in avail.items() if not v.comparison}
    comparison = {k: v for k, v in avail.items() if v.comparison}

    metric_type = st.radio("Metric type", ["Absolute", "Comparison"], horizontal=True)
    metric_pool = absolute if metric_type == "Absolute" else comparison

    if not metric_pool:
        st.warning("No metrics available for this configuration.")
        return

    metric_key = st.selectbox(
        "Metric",
        list(metric_pool.keys()),
        format_func=lambda k: f"{metric_pool[k].name} — {metric_pool[k].description}",
    )
    metric = metric_pool[metric_key]

    # --- Section 1: Single training size ---
    st.subheader("Analysis for Specific Training Size")
    n_episodes = st.select_slider("Training episodes", options=n_episodes_values, value=n_episodes_values[-1], key="metrics_n_episodes")

    predictions = _load_predictions_for_methods(
        entries_df, n_episodes, methods, gt_store, params,
    )
    if not predictions:
        st.warning("No predictions found for selected configuration.")
        return

    # Load baseline for comparison metrics
    baseline_mp = predictions.get(baseline_method) if metric.comparison else None
    if metric.comparison and baseline_mp is None:
        st.warning(f"Baseline method '{baseline_method}' not found at {n_episodes} episodes.")
        return

    # Compute metric for each method
    methods_to_show = [m for m in methods if m != baseline_method] if metric.comparison else methods
    metric_values = {}
    for method in methods_to_show:
        mp = predictions.get(method)
        if mp is None:
            continue

        # Apply state filtering
        mask = filter_states(
            mp.mean, mp.variance,
            params.get("filter_high_variance", 0),
            params.get("filter_extreme_mean", 0),
        )

        # Build filtered predictions
        filtered_mp = MethodPredictions(
            method=mp.method, n_episodes=mp.n_episodes,
            pivot=mp.pivot.iloc[mask], mean=mp.mean[mask], variance=mp.variance[mask],
            ground_truth_key=mp.ground_truth_key,
        )
        filtered_baseline = None
        if baseline_mp is not None:
            filtered_baseline = MethodPredictions(
                method=baseline_mp.method, n_episodes=baseline_mp.n_episodes,
                pivot=baseline_mp.pivot.iloc[mask], mean=baseline_mp.mean[mask],
                variance=baseline_mp.variance[mask],
                ground_truth_key=baseline_mp.ground_truth_key,
            )

        vals = metric.fn(
            filtered_mp, gt_store,
            baseline=filtered_baseline,
            epsilon=params.get("epsilon", 1e-2),
            n_buckets=params.get("n_buckets", 10),
        )
        metric_values[method] = vals

    if not metric_values:
        st.warning("No data to display.")
        return

    # Plot
    if metric.plot_type == "scatter_timestep":
        any_mp = next(iter(predictions.values()))
        gt_data = gt_store.get(any_mp.ground_truth_key)
        fig = go.Figure()
        for i, method in enumerate(methods_to_show):
            mp = predictions.get(method)
            if mp is None:
                continue
            color = COLORS[i % len(COLORS)]
            # Sort by timestep and compute rolling statistics
            order = np.argsort(gt_data.timestep)
            ts = gt_data.timestep[order]
            var = mp.variance[order]
            window = max(len(ts) // 100, 10)
            s = pd.Series(var, index=ts)
            rolling_median = s.rolling(window, center=True, min_periods=1).median()
            rolling_q25 = s.rolling(window, center=True, min_periods=1).quantile(0.25)
            rolling_q75 = s.rolling(window, center=True, min_periods=1).quantile(0.75)
            # 25th-75th percentile band
            fig.add_trace(go.Scatter(
                x=np.concatenate([ts, ts[::-1]]),
                y=np.concatenate([rolling_q75.values, rolling_q25.values[::-1]]),
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)",
                line=dict(width=0),
                showlegend=False, name=method,
            ))
            fig.add_trace(go.Scatter(
                x=ts, y=rolling_median.values,
                mode="lines", name=method,
                line=dict(color=color, width=2),
            ))
        fig.update_layout(
            title=f"{metric.name} ({n_episodes} episodes)",
            xaxis_title="Timestep in Episode", yaxis_title="Variance (log)",
            yaxis_type="log",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
    elif metric.plot_type == "bar":
        fig = plot_bar(metric_values, title=f"{metric.name} ({n_episodes} episodes)")
    else:
        fig = plot_violin(
            metric_values, title=f"{metric.name} ({n_episodes} episodes)",
            x_label=metric.name,
            reference_line=metric.reference_line,
            reference_label=metric.reference_label,
        )
    st.plotly_chart(fig, use_container_width=True)

    # Stats table
    stats = _compute_stats_table(metric_values)
    st.dataframe(stats, use_container_width=True, hide_index=True)

    # --- Section 2: Evolution across training sizes ---
    if metric.plot_type == "scatter_timestep":
        return
    st.subheader("Evolution Across Training Sizes")

    records = []
    for n_ep in n_episodes_values:
        preds = _load_predictions_for_methods(
            entries_df, n_ep, methods, gt_store, params,
        )
        bl = preds.get(baseline_method) if metric.comparison else None

        for method in methods_to_show:
            mp = preds.get(method)
            if mp is None:
                continue

            vals = metric.fn(
                mp, gt_store, baseline=bl,
                epsilon=params.get("epsilon", 1e-2),
                n_buckets=params.get("n_buckets", 10),
            )

            if isinstance(vals, pd.DataFrame):
                mean_val = vals["metric_value"].mean()
            else:
                mean_val = np.nanmean(vals)

            stderr = bootstrap_metric_stderr(
                mp, gt_store, metric.fn, baseline=bl,
                epsilon=params.get("epsilon", 1e-2),
                n_buckets=params.get("n_buckets", 10),
            )

            records.append({
                "method": method,
                "n_episodes": n_ep,
                "mean": mean_val,
                "stderr": stderr,
            })

    if records:
        fig = plot_evolution(
            records, title=f"{metric.name} vs Training Size",
            y_label=metric.name,
            reference_line=metric.reference_line,
            reference_label=metric.reference_label,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Data table"):
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
