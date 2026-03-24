"""Shared plotting functions. All return Plotly figures."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional, List


COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def plot_violin(
    data: Dict[str, np.ndarray],
    title: str = "",
    x_label: str = "Value",
    reference_line: Optional[float] = None,
    reference_label: Optional[str] = None,
) -> go.Figure:
    """Violin plots with one subplot per method (shared x-axis)."""
    from plotly.subplots import make_subplots

    methods = list(data.keys())
    n = len(methods)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=methods,
    )

    for i, method in enumerate(methods):
        values = data[method]
        fig.add_trace(
            go.Violin(
                x=values, name=method,
                fillcolor=COLORS[i % len(COLORS)],
                line_color=COLORS[i % len(COLORS)],
                opacity=0.7,
                meanline_visible=True,
                showlegend=False,
            ),
            row=i + 1, col=1,
        )
        if reference_line is not None:
            fig.add_vline(
                x=reference_line, line_dash="dash", line_color="white",
                annotation_text=reference_label if i == 0 else None,
                row=i + 1, col=1,
            )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=max(250 * n, 300),
    )
    fig.update_xaxes(title_text=x_label, row=n, col=1)

    return fig


def plot_bar(
    data: Dict[str, pd.DataFrame],
    title: str = "",
    x_label: str = "Bucket",
    y_label: str = "Value",
) -> go.Figure:
    """Grouped bar chart (e.g. for decile metrics).

    Each DataFrame should have 'bucket' and 'metric_value' columns.
    """
    fig = go.Figure()
    for i, (method, df) in enumerate(data.items()):
        fig.add_trace(go.Bar(
            x=df["bucket"], y=df["metric_value"], name=method,
            marker_color=COLORS[i % len(COLORS)],
        ))

    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label,
        barmode="group", template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_evolution(
    records: List[dict],
    title: str = "",
    y_label: str = "Value",
    reference_line: Optional[float] = None,
    reference_label: Optional[str] = None,
) -> go.Figure:
    """Line plot of metric mean (± stderr) vs n_episodes.

    Each record: {method, n_episodes, mean, stderr (optional)}
    """
    fig = go.Figure()
    df = pd.DataFrame(records)

    for i, method in enumerate(df["method"].unique()):
        mdf = df[df["method"] == method].sort_values("n_episodes")
        color = COLORS[i % len(COLORS)]

        error_y = None
        if "stderr" in mdf.columns and mdf["stderr"].notna().any():
            error_y = dict(type="data", array=mdf["stderr"].values, visible=True)

        fig.add_trace(go.Scatter(
            x=mdf["n_episodes"], y=mdf["mean"], name=method,
            mode="lines+markers", marker_color=color, error_y=error_y,
        ))

    if reference_line is not None:
        fig.add_hline(
            y=reference_line, line_dash="dash", line_color="gray",
            annotation_text=reference_label or "",
        )

    fig.update_layout(
        title=title, xaxis_title="Training Episodes", yaxis_title=y_label,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_trajectory(
    traces: Dict[str, pd.Series],
    ground_truth: Optional[pd.Series] = None,
    critic_values: Optional[pd.Series] = None,
    stds: Optional[Dict[str, pd.Series]] = None,
    title: str = "",
) -> go.Figure:
    """Line plot for episode trajectory: predicted values along timesteps.

    traces: {method_name: Series indexed by timestep}
    ground_truth: Series indexed by timestep
    stds: optional {method_name: Series of std} for ±1 std bands
    """
    fig = go.Figure()

    if ground_truth is not None:
        fig.add_trace(go.Scatter(
            x=ground_truth.index, y=ground_truth.values,
            name="Ground Truth", mode="lines",
            line=dict(dash="dash", color="white", width=2),
        ))

    if critic_values is not None:
        fig.add_trace(go.Scatter(
            x=critic_values.index, y=critic_values.values,
            name="Policy Critic", mode="lines",
            line=dict(dash="dot", color="yellow", width=1.5),
        ))

    for i, (method, series) in enumerate(traces.items()):
        color = COLORS[i % len(COLORS)]
        # ±1 std band
        if stds and method in stds:
            std = stds[method]
            upper = series + std
            lower = series - std
            fig.add_trace(go.Scatter(
                x=np.concatenate([series.index, series.index[::-1]]),
                y=np.concatenate([upper.values, lower.values[::-1]]),
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
                line=dict(width=0),
                showlegend=False, name=method,
            ))
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=method, mode="lines+markers",
            marker_color=color,
        ))

    fig.update_layout(
        title=title, xaxis_title="Timestep", yaxis_title="Value",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
