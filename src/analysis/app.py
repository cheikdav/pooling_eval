"""Streamlit dashboard for value estimator analysis."""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from metrics import METRICS, compute_metric
from metadata_discovery import discover_predictions


st.set_page_config(page_title="Value Estimator Analysis", layout="wide")


# Data Loading Functions

@st.cache_data
def load_predictions_data(experiments_dir):
    """Load all prediction files with metadata."""
    all_preds = discover_predictions(Path(experiments_dir))
    if not all_preds:
        return pd.DataFrame()
    return pd.DataFrame(all_preds)


@st.cache_data
def compute_stats_from_predictions(predictions_path, n_episodes):
    """Load predictions and compute statistics aggregated across batches.

    Memory-efficient: loads raw data, computes stats, then frees raw data.
    Only the aggregated stats (one row per state) are cached.

    Args:
        predictions_path: Path to predictions.parquet file
        n_episodes: Number of episodes (for metadata only)

    Returns:
        DataFrame with columns: state_idx, n_episodes, mean, variance, std, count
        where statistics are aggregated across all batches for each state
    """
    # Load raw predictions
    df = pd.read_parquet(predictions_path)

    # Compute statistics grouped by state_idx, aggregated across batches
    # Structure: (state_idx, episode_idx, batch_name, predicted_value) -> (state_idx, stats)
    stats = df.groupby('state_idx')['predicted_value'].agg(
        mean='mean',
        variance='var',
        std='std',
        count='count'
    ).reset_index()

    stats['n_episodes'] = n_episodes

    # Free raw data memory immediately
    del df

    return stats


# UI Helper Functions

def show_selection_filters(metadata_df):
    """Display sidebar filters and return filtered metadata."""
    st.sidebar.header("Filters")

    # Environment selection
    envs = sorted(metadata_df['policy_environment'].unique())
    selected_env = st.sidebar.selectbox("Environment", envs)

    # Policy selection
    env_df = metadata_df[metadata_df['policy_environment'] == selected_env]
    policies = sorted(env_df['policy_display_name'].unique())
    selected_policy = st.sidebar.selectbox("Policy", policies)

    # Filter to selected env/policy
    filtered = env_df[env_df['policy_display_name'] == selected_policy]

    # Method selection
    methods = sorted(filtered['method'].unique())
    selected_methods = st.sidebar.multiselect(
        "Methods to Compare", methods, default=methods
    )

    # Baseline method selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Comparison Settings**")
    baseline_method = st.sidebar.selectbox(
        "Baseline Method",
        methods,
        index=methods.index('monte_carlo') if 'monte_carlo' in methods else 0,
        help="Method to use as baseline for ratio metrics"
    )

    # Ensure baseline method is included in the data
    methods_to_load = list(set(selected_methods + [baseline_method]))
    filtered = filtered[filtered['method'].isin(methods_to_load)]

    return filtered, selected_env, selected_policy, selected_methods, baseline_method


def plot_metric_distribution(stats_df, metric_key, methods, n_episodes, baseline_method='monte_carlo'):
    """Create histogram of metric values."""
    metric_info = METRICS[metric_key]
    metric_data = compute_metric(stats_df, metric_key, baseline_method)
    filtered = metric_data[metric_data['method'].isin(methods)]

    if filtered.empty:
        st.warning("No data available")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.histogram(
            filtered, x='metric_value', color='method',
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
        st.markdown(f"**Statistics** (vs {baseline_method})")
        summary = filtered.groupby('method')['metric_value'].agg(
            mean='mean', std='std'
        ).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)


def plot_metric_evolution(all_stats_df, metric_key, methods, baseline_method='monte_carlo'):
    """Create evolution plot across n_episodes."""
    metric_info = METRICS[metric_key]
    metric_data = compute_metric(all_stats_df, metric_key, baseline_method)
    filtered = metric_data[metric_data['method'].isin(methods)]

    if filtered.empty:
        st.warning("No data available")
        return

    # Aggregate by method and n_episodes
    summary = filtered.groupby(['method', 'n_episodes'])['metric_value'].agg(
        mean='mean', std='std'
    ).reset_index()

    fig = go.Figure()

    for method in methods:
        method_data = summary[summary['method'] == method]
        fig.add_trace(go.Scatter(
            x=method_data['n_episodes'],
            y=method_data['mean'],
            error_y=dict(type='data', array=method_data['std']),
            mode='lines+markers',
            name=method,
            marker=dict(size=10)
        ))

    if metric_info['reference_line'] is not None:
        fig.add_hline(
            y=metric_info['reference_line'], line_dash="dash", line_color="red",
            annotation_text=metric_info['reference_label']
        )

    fig.update_layout(
        title=f"{metric_info['name']} vs Training Data Size (baseline: {baseline_method})",
        xaxis_title="Training Episodes",
        yaxis_title=f"Mean {metric_info['name']} (± std)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show data table"):
        st.dataframe(summary, use_container_width=True)


# Main App

st.title("Value Estimator Variance Analysis")
st.markdown("Compare variance and performance of different value estimation methods")

# Load data
experiments_dir = Path("experiments")
if not experiments_dir.exists():
    st.error(f"Experiments directory not found: {experiments_dir.absolute()}")
    st.stop()

metadata_df = load_predictions_data(str(experiments_dir))
if metadata_df.empty:
    st.error("No predictions found. Run: `python -m src.evaluate --config <config>.yaml`")
    st.stop()

# Sidebar filters
filtered_metadata, env, policy, methods, baseline_method = show_selection_filters(metadata_df)

if not methods:
    st.warning("Please select at least one method")
    st.stop()

# Show current selection
st.markdown("### Current Selection")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Environment", env)
with col2:
    st.metric("Policy", policy)
with col3:
    st.metric("Baseline", baseline_method)

st.markdown("---")

# Section 1: Single n_episodes analysis
st.header("📊 Analysis for Specific Training Size")

n_episodes_values = sorted(filtered_metadata['n_episodes'].unique())
selected_n_ep = st.selectbox(
    "Training data size:",
    n_episodes_values,
    format_func=lambda x: f"{x} episodes"
)

# Load stats for selected n_episodes (include baseline method)
methods_to_load = list(set(methods + [baseline_method]))
filtered_for_n_ep = filtered_metadata[filtered_metadata['n_episodes'] == selected_n_ep]

stats_list = []
for _, row in filtered_for_n_ep.iterrows():
    if row['method'] in methods_to_load:
        stats = compute_stats_from_predictions(row['predictions_path'], row['n_episodes'])
        stats['method'] = row['method']
        stats_list.append(stats)

stats_single = pd.concat(stats_list, ignore_index=True) if stats_list else pd.DataFrame()

if stats_single.empty:
    st.error(f"No data for {selected_n_ep} episodes")
    st.stop()

metric_key_single = st.selectbox(
    "Metric:",
    list(METRICS.keys()),
    format_func=lambda k: METRICS[k]['name'],
    key="metric_single"
)

st.markdown(f"**{METRICS[metric_key_single]['name']}**: {METRICS[metric_key_single]['description']}")
plot_metric_distribution(stats_single, metric_key_single, methods, selected_n_ep, baseline_method)

st.markdown("---")

# Section 2: Evolution across n_episodes
st.header("📈 Evolution Across Training Sizes")

metric_key_evolution = st.selectbox(
    "Metric:",
    list(METRICS.keys()),
    format_func=lambda k: METRICS[k]['name'],
    key="metric_evolution"
)

st.markdown(f"**{METRICS[metric_key_evolution]['name']}** evolution across training data sizes")

# Load stats for all n_episodes (include baseline method)
with st.spinner("Loading data..."):
    methods_to_load_all = list(set(methods + [baseline_method]))

    stats_list_all = []
    for _, row in filtered_metadata.iterrows():
        if row['method'] in methods_to_load_all and row['n_episodes'] in n_episodes_values:
            stats = compute_stats_from_predictions(row['predictions_path'], row['n_episodes'])
            stats['method'] = row['method']
            stats_list_all.append(stats)

    stats_all = pd.concat(stats_list_all, ignore_index=True) if stats_list_all else pd.DataFrame()

if stats_all.empty:
    st.error("No data available for evolution plot")
    st.stop()

plot_metric_evolution(stats_all, metric_key_evolution, methods, baseline_method)

st.markdown("---")
st.caption(f"Policy: {policy} | Methods: {', '.join(methods)}")
