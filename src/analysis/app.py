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
def compute_stats_from_predictions(predictions_path, method, n_episodes, s1_proportion=0.9, seed=42):
    """Load predictions and compute statistics.

    Returns: (stats_full, stats_s1, stats_s2, stats_differences)
    Each DataFrame has: state_idx, method, n_episodes, mean, variance, std, count
    """
    df = pd.read_parquet(predictions_path)

    def agg_stats(data, method_name, n_eps):
        stats = data.groupby('state_idx')['predicted_value'].agg(
            mean='mean', variance='var', std='std', count='count'
        ).reset_index()
        stats['method'] = method_name
        stats['n_episodes'] = n_eps
        return stats

    # Full dataset
    stats_full = agg_stats(df, method, n_episodes)

    # Split into S1/S2 partitions
    np.random.seed(seed)
    episodes = df['episode_idx'].unique()
    n_s1 = int(len(episodes) * s1_proportion)
    shuffled = np.random.permutation(episodes)
    s1_eps, s2_eps = set(shuffled[:n_s1]), set(shuffled[n_s1:])

    df_s1 = df[df['episode_idx'].isin(s1_eps)]
    df_s2 = df[df['episode_idx'].isin(s2_eps)]

    stats_s1 = agg_stats(df_s1, method, n_episodes)
    stats_s2 = agg_stats(df_s2, method, n_episodes)

    # Compute paired differences
    np.random.seed(seed + 1)
    s1_states = df_s1['state_idx'].unique()
    s2_states = df_s2['state_idx'].unique()
    pairings = {s: np.random.choice(s2_states) for s in s1_states}

    df_s1 = df_s1.copy()
    df_s1['paired_state_idx'] = df_s1['state_idx'].map(pairings)

    df_s2_paired = df_s2[['state_idx', 'batch_name', 'predicted_value']].copy()
    df_s2_paired.columns = ['paired_state_idx', 'batch_name', 'paired_value']

    df_merged = df_s1.merge(df_s2_paired, on=['paired_state_idx', 'batch_name'], how='inner')
    df_merged['value_difference'] = df_merged['predicted_value'] - df_merged['paired_value']

    stats_diff = df_merged.groupby('state_idx')['value_difference'].agg(
        mean='mean', variance='var', std='std', count='count'
    ).reset_index()
    stats_diff['method'] = method
    stats_diff['n_episodes'] = n_episodes

    return stats_full, stats_s1, stats_s2, stats_diff


@st.cache_data
def load_all_stats(metadata_df, methods, n_episodes_list, dataset_type):
    """Load stats for all method/n_episodes combinations.

    Args:
        metadata_df: DataFrame with columns [method, n_episodes, predictions_path, ...]
        methods: List of method names to load
        n_episodes_list: List of n_episodes values to load
        dataset_type: 'full', 's1', 's2', or 'differences'

    Returns:
        DataFrame with aggregated stats
    """
    dataset_idx = {'full': 0, 's1': 1, 's2': 2, 'differences': 3}[dataset_type]
    all_stats = []

    for _, row in metadata_df.iterrows():
        if row['method'] not in methods or row['n_episodes'] not in n_episodes_list:
            continue

        stats_tuple = compute_stats_from_predictions(
            row['predictions_path'], row['method'], row['n_episodes']
        )
        all_stats.append(stats_tuple[dataset_idx])

    return pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()


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

    # Dataset type
    st.sidebar.markdown("---")
    dataset_type = st.sidebar.selectbox(
        "Dataset",
        ['full', 's1', 's2', 'differences'],
        format_func=lambda x: {
            'full': 'Full Dataset',
            's1': 'S1 Partition (90%)',
            's2': 'S2 Partition (10%)',
            'differences': 'Differences (S1 - S2)'
        }[x]
    )

    # Ensure baseline method is included in the data
    methods_to_load = list(set(selected_methods + [baseline_method]))
    filtered = filtered[filtered['method'].isin(methods_to_load)]

    return filtered, selected_env, selected_policy, selected_methods, baseline_method, dataset_type


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
filtered_metadata, env, policy, methods, baseline_method, dataset_type = show_selection_filters(metadata_df)

if not methods:
    st.warning("Please select at least one method")
    st.stop()

# Show current selection
st.markdown("### Current Selection")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Environment", env)
with col2:
    st.metric("Policy", policy)
with col3:
    st.metric("Methods", len(methods))
with col4:
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
stats_single = load_all_stats(
    filtered_metadata[filtered_metadata['n_episodes'] == selected_n_ep],
    methods_to_load,
    [selected_n_ep],
    dataset_type
)

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
    stats_all = load_all_stats(
        filtered_metadata,
        methods_to_load_all,
        n_episodes_values,
        dataset_type
    )

if stats_all.empty:
    st.error("No data available for evolution plot")
    st.stop()

plot_metric_evolution(stats_all, metric_key_evolution, methods, baseline_method)

st.markdown("---")
st.caption(f"Dataset: {dataset_type} | Policy: {policy}")
