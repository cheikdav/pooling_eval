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
def compute_stats_from_predictions(predictions_path, n_episodes, dataset_type='full', s1_proportion=0.9, seed=42):
    """Load predictions and compute statistics aggregated across batches.

    Memory-efficient: loads raw data, computes stats, then frees raw data.
    Only the aggregated stats (one row per state) are cached.

    Args:
        predictions_path: Path to predictions.parquet file
        n_episodes: Number of episodes (for metadata only)
        dataset_type: 'full' for all data, 'differences' for paired differences
        s1_proportion: Proportion of episodes for S1 partition (used for differences)
        seed: Random seed for episode partitioning

    Returns:
        DataFrame with columns: state_idx, n_episodes, mean, variance, std, count
        where statistics are aggregated across all batches for each state
    """
    # Load raw predictions
    df = pd.read_parquet(predictions_path)

    if dataset_type == 'full':
        # Compute statistics grouped by state_idx, aggregated across batches
        stats = df.groupby('state_idx')['predicted_value'].agg(
            mean='mean',
            variance='var',
            std='std',
            count='count'
        ).reset_index()
        stats['n_episodes'] = n_episodes
        del df
        return stats

    elif dataset_type == 'differences':
        # Split episodes into S1 (90%) and S2 (10%)
        episodes = df['episode_idx'].unique()
        np.random.seed(seed)
        shuffled = np.random.permutation(episodes)
        n_s1 = int(len(episodes) * s1_proportion)
        s1_eps = set(shuffled[:n_s1])
        s2_eps = set(shuffled[n_s1:])

        df_s1 = df[df['episode_idx'].isin(s1_eps)].copy()
        df_s2 = df[df['episode_idx'].isin(s2_eps)].copy()

        # Create stable pairings: each S1 state gets paired with a random S2 state
        np.random.seed(seed + 1)
        s1_states = df_s1['state_idx'].unique()
        s2_states = df_s2['state_idx'].unique()
        pairings = {s: np.random.choice(s2_states) for s in s1_states}

        # Add paired_state_idx column to df_s1
        df_s1['paired_state_idx'] = df_s1['state_idx'].map(pairings)

        # Merge S1 with S2 on paired states
        df_merged = df_s1.merge(
            df_s2[['state_idx', 'batch_name', 'predicted_value']],
            left_on=['paired_state_idx', 'batch_name'],
            right_on=['state_idx', 'batch_name'],
            suffixes=('_s1', '_s2')
        )

        # Compute differences: V(s) - V(s')
        df_merged['difference'] = df_merged['predicted_value_s1'] - df_merged['predicted_value_s2']

        # Aggregate statistics on differences
        stats_diff = df_merged.groupby('state_idx_s1')['difference'].agg(
            mean='mean',
            variance='var',
            std='std',
            count='count'
        ).reset_index()
        stats_diff.rename(columns={'state_idx_s1': 'state_idx'}, inplace=True)
        stats_diff['n_episodes'] = n_episodes

        del df, df_s1, df_s2, df_merged
        return stats_diff

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


# UI Helper Functions

def show_selection_filters(metadata_df):
    """Display sidebar filters and return filtered metadata."""
    st.sidebar.header("Filters")

    # Dataset type selection
    dataset_type = st.sidebar.radio(
        "Dataset Type",
        options=['full', 'differences'],
        format_func=lambda x: 'Full Dataset' if x == 'full' else 'Paired Differences (S1 - S2)',
        help="Full: all states | Differences: V(s) - V(s') for 90% of episodes paired with 10%"
    )

    st.sidebar.markdown("---")

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

    # Epsilon parameter for log ratio metrics
    log_epsilon = st.sidebar.slider(
        "Log Epsilon (log₁₀ε)",
        min_value=-5.0,
        max_value=1.0,
        value=-2.0,
        step=0.5,
        help="Epsilon value for log ratio computations: log(x+ε) - log(y+ε). Adjust on log scale."
    )
    epsilon = 10 ** log_epsilon
    st.sidebar.caption(f"ε = {epsilon:.2e}")

    # Ensure baseline method is included in the data
    methods_to_load = list(set(selected_methods + [baseline_method]))
    filtered = filtered[filtered['method'].isin(methods_to_load)]

    return filtered, selected_env, selected_policy, selected_methods, baseline_method, epsilon, dataset_type


def plot_metric_distribution(stats_dict, metric_key, methods, n_episodes, baseline_method='monte_carlo', epsilon=1e-10):
    """Create histogram of metric values.

    Args:
        stats_dict: Dict mapping method names to DataFrames
        metric_key: Metric to compute
        methods: List of methods to display
        n_episodes: Number of episodes (for display)
        baseline_method: Baseline method name
        epsilon: Small value added before taking log
    """
    metric_info = METRICS[metric_key]

    if baseline_method not in stats_dict:
        st.error(f"Baseline method {baseline_method} not found")
        return

    baseline_stats = stats_dict[baseline_method]

    # Compute metrics for each method
    metric_list = []
    for method in methods:
        if method == baseline_method or method not in stats_dict:
            continue

        method_stats = stats_dict[method]
        metric_df = compute_metric(baseline_stats, method_stats, metric_key, epsilon=epsilon)
        metric_df['method'] = method
        metric_list.append(metric_df)

    if not metric_list:
        st.warning("No data available")
        return

    combined = pd.concat(metric_list, ignore_index=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.histogram(
            combined, x='metric_value', color='method',
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
        summary = combined.groupby('method')['metric_value'].agg(
            mean='mean', std='std'
        ).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)


def plot_metric_evolution(metadata_df, metric_key, methods, baseline_method, n_episodes_values, epsilon=1e-10, dataset_type='full'):
    """Create evolution plot across n_episodes.

    Memory-efficient: processes one n_episodes at a time, loads baseline once,
    then for each method, computes metric and discards immediately.

    Args:
        metadata_df: DataFrame with experiment metadata
        metric_key: Metric to compute
        methods: List of methods to display
        baseline_method: Baseline method name
        n_episodes_values: List of n_episodes values to plot
        epsilon: Small value added before taking log
        dataset_type: 'full' or 'differences'
    """
    metric_info = METRICS[metric_key]

    # Collect summary statistics for each method
    summary_records = []

    # Process one n_episodes at a time
    for n_ep in n_episodes_values:
        # Load baseline stats for this n_episodes
        baseline_row = metadata_df[(metadata_df['method'] == baseline_method) &
                                   (metadata_df['n_episodes'] == n_ep)]

        if baseline_row.empty:
            continue

        baseline_stats = compute_stats_from_predictions(
            baseline_row.iloc[0]['predictions_path'],
            n_ep,
            dataset_type=dataset_type
        )

        # Process each method for this n_episodes
        for method in methods:
            if method == baseline_method:
                continue

            method_row = metadata_df[(metadata_df['method'] == method) &
                                    (metadata_df['n_episodes'] == n_ep)]

            if method_row.empty:
                continue

            # Load method stats
            method_stats = compute_stats_from_predictions(
                method_row.iloc[0]['predictions_path'],
                n_ep,
                dataset_type=dataset_type
            )

            # Compute metric for this method vs baseline
            metric_df = compute_metric(baseline_stats, method_stats, metric_key, epsilon=epsilon)

            # Store summary statistics
            if not metric_df.empty:
                summary_records.append({
                    'method': method,
                    'n_episodes': n_ep,
                    'mean': metric_df['metric_value'].mean(),
                    'std': metric_df['metric_value'].std()
                })

            # Discard method stats and metric immediately
            del method_stats, metric_df

        # Discard baseline stats after processing all methods for this n_episodes
        del baseline_stats

    if not summary_records:
        st.warning("No data available")
        return

    summary = pd.DataFrame(summary_records)

    # Create plot
    fig = go.Figure()

    for method in methods:
        if method == baseline_method:
            continue
        method_data = summary[summary['method'] == method]
        if not method_data.empty:
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
filtered_metadata, env, policy, methods, baseline_method, epsilon, dataset_type = show_selection_filters(metadata_df)

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
    st.metric("Baseline", baseline_method)
with col4:
    dataset_label = "Full" if dataset_type == "full" else "Differences"
    st.metric("Dataset", dataset_label)

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

stats_dict_single = {}
for _, row in filtered_for_n_ep.iterrows():
    if row['method'] in methods_to_load:
        print(row['method'])
        stats = compute_stats_from_predictions(row['predictions_path'], row['n_episodes'], dataset_type=dataset_type)
        stats_dict_single[row['method']] = stats

if not stats_dict_single or baseline_method not in stats_dict_single:
    st.error(f"No data for {selected_n_ep} episodes")
    st.stop()

metric_key_single = st.selectbox(
    "Metric:",
    list(METRICS.keys()),
    format_func=lambda k: METRICS[k]['name'],
    key="metric_single"
)

st.markdown(f"**{METRICS[metric_key_single]['name']}**: {METRICS[metric_key_single]['description']}")
plot_metric_distribution(stats_dict_single, metric_key_single, methods, selected_n_ep, baseline_method, epsilon)

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

# Plot evolution (loads stats one method at a time to minimize memory)
plot_metric_evolution(filtered_metadata, metric_key_evolution, methods, baseline_method, n_episodes_values, epsilon, dataset_type)

st.markdown("---")
st.caption(f"Policy: {policy} | Methods: {', '.join(methods)}")
