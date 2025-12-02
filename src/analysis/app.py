"""Streamlit dashboard for visualizing value estimator predictions using metadata."""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import traceback

from metrics import METRICS, compute_metric
from metadata_discovery import (
    discover_predictions,
    get_selection_hierarchy,
    filter_estimators
)


PRIMARY_SELECTION_KEYS = [
    ('policy_environment', 'Environment'),
    ('policy_display_name', 'Policy'),
    ('n_episodes', 'Training Data Amount'),
]


st.set_page_config(page_title="Value Estimator Analysis", layout="wide")

st.title("Value Estimator Variance Analysis")
st.markdown("Compare variance and performance of different value estimation methods")


@st.cache_data
def load_and_compute_stats(predictions_path, method, n_episodes, s1_proportion=0.9, seed=42):
    """Load predictions file and compute statistics, discarding raw data.

    Args:
        predictions_path: Path to predictions parquet file
        method: Method name to add to stats
        n_episodes: Number of training episodes to add to stats
        s1_proportion: Proportion of episodes for S1 partition
        seed: Random seed for episode partitioning

    Returns:
        Tuple of 4 stats DataFrames: (stats_full, stats_s1, stats_s2, stats_merged)
        Each stats DataFrame has columns: state_idx, method, n_episodes, mean, variance, std, count
    """
    df = pd.read_parquet(predictions_path)

    def compute_stats(dataframe, method_name, n_eps):
        """Helper to compute stats from a predictions dataframe."""
        stats = dataframe.groupby(['state_idx'])['predicted_value'].agg(
            mean='mean',
            variance='var',
            std='std',
            count='count'
        ).reset_index()
        stats['method'] = method_name
        stats['n_episodes'] = n_eps
        return stats

    # Full dataset stats
    stats = compute_stats(df, method, n_episodes)

    # Partition episodes into S1 and S2
    np.random.seed(seed)
    all_episodes = df['episode_idx'].unique()
    n_s1 = int(len(all_episodes) * s1_proportion)

    shuffled_episodes = np.random.permutation(all_episodes)
    s1_episodes = set(shuffled_episodes[:n_s1])
    s2_episodes = set(shuffled_episodes[n_s1:])

    # Split dataframe by partition
    df_s1 = df[df['episode_idx'].isin(s1_episodes)].copy()
    df_s2 = df[df['episode_idx'].isin(s2_episodes)].copy()

    stats_s1 = compute_stats(df_s1, method, n_episodes)
    stats_s2 = compute_stats(df_s2, method, n_episodes)

    # Create stable random pairings (state_idx in S1 -> state_idx in S2)
    unique_s1_states = df_s1['state_idx'].unique()
    unique_s2_states = df_s2['state_idx'].unique()

    np.random.seed(seed + 1)
    pairings = {
        s: np.random.choice(unique_s2_states)
        for s in unique_s1_states
    }

    # Add paired state column to S1 dataframe
    df_s1['paired_state_idx'] = df_s1['state_idx'].map(pairings)

    # Prepare S2 for merge - only keep necessary columns
    df_s2_for_merge = df_s2[['state_idx', 'batch_idx', 'predicted_value']].copy()
    df_s2_for_merge.columns = ['paired_state_idx', 'batch_idx', 'paired_value']

    # Merge to get paired values
    df_merged = df_s1.merge(
        df_s2_for_merge,
        on=['paired_state_idx', 'batch_idx'],
        how='inner'
    )

    # Compute differences
    df_merged['value_difference'] = df_merged['predicted_value'] - df_merged['paired_value']

    # Compute stats on differences
    stats_merged = df_merged.groupby(['state_idx'])['value_difference'].agg(
        mean='mean',
        variance='var',
        std='std',
        count='count'
    ).reset_index()
    stats_merged['method'] = method
    stats_merged['n_episodes'] = n_episodes

    return stats, stats_s1, stats_s2, stats_merged


experiments_dir = Path("experiments")
if not experiments_dir.exists():
    st.error(f"Experiments directory not found at {experiments_dir.absolute()}")
    st.stop()

try:
    all_predictions = discover_predictions(experiments_dir)
    if not all_predictions:
        st.error("No predictions with metadata found in experiments directory")
        st.info("Run: `python -m src.evaluate --config <your_config>.yaml` to generate predictions.")
        st.stop()
except Exception as e:
    st.error(f"Failed to discover predictions: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

st.sidebar.header("Hierarchical Selection")

current_filters = {}
primary_keys = [key for key, _ in PRIMARY_SELECTION_KEYS]

for step_num, (key, display_name) in enumerate(PRIMARY_SELECTION_KEYS, 1):
    hierarchy = get_selection_hierarchy(all_predictions, primary_keys, current_filters)

    if hierarchy['next_key'] != key:
        continue

    available_values = hierarchy['available_values']

    if not available_values:
        st.error(f"No values available for {display_name}")
        st.stop()

    format_func = None
    if key == 'n_episodes':
        format_func = lambda x: f"{x} episodes"

    selected_value = st.sidebar.selectbox(
        f"{step_num}. {display_name}",
        available_values,
        format_func=format_func,
        help=f"Choose {display_name.lower()}"
    )

    current_filters[key] = selected_value

    if key == 'policy_display_name':
        filtered_predictions = hierarchy['current_estimators']
        filtered_predictions = filter_estimators(filtered_predictions, key, selected_value)

        if filtered_predictions:
            sample_pred = filtered_predictions[0]
            with st.sidebar.expander("Policy Details"):
                st.markdown(f"**Algorithm:** {sample_pred.get('policy_algorithm')}")
                st.markdown(f"**Seed:** {sample_pred.get('policy_seed')}")
                if sample_pred.get('policy_average_reward') is not None:
                    st.markdown(f"**Avg Reward:** {sample_pred.get('policy_average_reward'):.2f}")
                st.markdown(f"**Learning Rate:** {sample_pred.get('policy_learning_rate')}")
                st.markdown(f"**Gamma:** {sample_pred.get('policy_gamma')}")
                st.markdown(f"**Total Timesteps:** {sample_pred.get('policy_total_timesteps')}")

final_hierarchy = get_selection_hierarchy(all_predictions, primary_keys, current_filters)
selected_predictions = final_hierarchy['current_estimators']

if not selected_predictions:
    st.error("No predictions found for the selected criteria")
    st.stop()

additional_params = final_hierarchy['additional_params']

if additional_params:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Additional Parameters**")

    additional_params = [p for p in additional_params if p['key'] not in ['method', 'n_batches', 'n_states', 'n_eval_episodes', 'created_at']]

    if additional_params:
        for param_info in additional_params:
            param_key = param_info['key']
            param_values = param_info['values']

            display_key = param_key.replace('_', ' ').title()

            selected_param_value = st.sidebar.selectbox(
                display_key,
                param_values,
                help=f"Choose {display_key.lower()}"
            )

            selected_predictions = filter_estimators(selected_predictions, param_key, selected_param_value)

if not selected_predictions:
    st.error("No predictions remaining after filtering")
    st.stop()

sample_prediction = selected_predictions[0]
experiment_id = sample_prediction['experiment_id']
selected_env = sample_prediction['policy_environment']
selected_policy_display = sample_prediction['policy_display_name']
selected_n_episodes = sample_prediction['n_episodes']

# Get all available methods and n_episodes for this environment/policy
all_methods_for_policy = sorted(set(pred['method'] for pred in selected_predictions))
all_n_episodes_for_policy = sorted(set(pred['n_episodes'] for pred in selected_predictions))

st.sidebar.markdown("---")
st.sidebar.markdown("**Method Selection**")

selected_methods = st.sidebar.multiselect(
    "Select Methods to Compare",
    all_methods_for_policy,
    default=all_methods_for_policy,
    help="Choose which methods to compare against Monte Carlo"
)

if not selected_methods:
    st.warning("Please select at least one method")
    st.stop()

# Load stats for ALL n_episodes and selected methods (memory efficient - stats only)
try:
    all_stats = []
    all_stats_s1 = []
    all_stats_s2 = []
    all_stats_merged = []

    total_files = len(all_n_episodes_for_policy) * (len(selected_methods) + 1)  # +1 for monte_carlo
    progress_bar = st.sidebar.progress(0, text="Loading predictions...")

    file_count = 0
    for n_ep in all_n_episodes_for_policy:
        # Always load Monte Carlo for comparison
        methods_to_load = ['monte_carlo'] + selected_methods if 'monte_carlo' not in selected_methods else selected_methods

        for method in methods_to_load:
            method_prediction = next((p for p in selected_predictions
                                     if p['method'] == method and p['n_episodes'] == n_ep), None)
            if not method_prediction:
                continue

            predictions_file = Path(method_prediction['predictions_path'])
            if not predictions_file.exists():
                continue

            # Load and compute stats, discard raw dataframe immediately
            stats, stats_s1, stats_s2, stats_merged = load_and_compute_stats(
                str(predictions_file),
                method,
                n_ep
            )

            all_stats.append(stats)
            all_stats_s1.append(stats_s1)
            all_stats_s2.append(stats_s2)
            all_stats_merged.append(stats_merged)

            file_count += 1
            progress_bar.progress(file_count / total_files, text=f"Loading {method} ({n_ep} episodes)...")

    progress_bar.empty()

    # Combine all stats
    stats_full = pd.concat(all_stats, ignore_index=True)
    stats_s1_full = pd.concat(all_stats_s1, ignore_index=True)
    stats_s2_full = pd.concat(all_stats_s2, ignore_index=True)
    stats_merged_full = pd.concat(all_stats_merged, ignore_index=True)

    st.sidebar.success(f"✓ Loaded stats for {len(selected_methods)} method(s) across {len(all_n_episodes_for_policy)} n_episodes")
    st.sidebar.markdown(f"- Total stats records: {len(stats_full)}")
    st.sidebar.markdown(f"- Methods: {', '.join(stats_full['method'].unique())}")
    st.sidebar.markdown(f"- n_episodes: {', '.join(map(str, sorted(stats_full['n_episodes'].unique())))}")

except Exception as e:
    st.error(f"Failed to load predictions: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Visualization Options**")

dataset_key = st.sidebar.selectbox(
    "Select Dataset",
    options=['full', 's1', 's2', 'differences'],
    format_func=lambda k: {
        'full': 'Full Dataset',
        's1': 'S1 Partition (90%)',
        's2': 'S2 Partition (10%)',
        'differences': 'Differences (S1 - S2)'
    }[k],
    help="Choose which dataset partition to analyze"
)

# Map dataset selection to appropriate stats
dataset_map = {
    'full': stats_full,
    's1': stats_s1_full,
    's2': stats_s2_full,
    'differences': stats_merged_full
}
current_stats = dataset_map[dataset_key]

st.markdown("### Current Selection")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Environment", selected_env)
with col2:
    st.metric("Policy", selected_policy_display)
with col3:
    st.metric("Methods", len(selected_methods))

st.markdown("---")

# Section 1: Analysis for specific n_episodes
st.header("📊 Analysis for Specific Training Size")

selected_n_ep_single = st.selectbox(
    "Select training data size to analyze:",
    all_n_episodes_for_policy,
    format_func=lambda x: f"{x} episodes"
)

# Filter stats for selected n_episodes
stats_single_n_ep = current_stats[current_stats['n_episodes'] == selected_n_ep_single]

metric_key_single = st.selectbox(
    "Select metric:",
    list(METRICS.keys()),
    format_func=lambda k: METRICS[k]['name'],
    key="metric_single"
)

metric_info_single = METRICS[metric_key_single]

st.markdown(f"**{metric_info_single['name']}**: {metric_info_single['description']}")
st.markdown(f"_Using dataset: **{dataset_key}** | Training: **{selected_n_ep_single} episodes**_")

try:
    metric_data_single = compute_metric(stats_single_n_ep, metric_key_single)
    filtered_single = metric_data_single[metric_data_single['method'].isin(selected_methods)]

    if len(filtered_single) == 0:
        st.warning("No data available for selected filters")
    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            fig1 = px.histogram(
                filtered_single,
                x='metric_value',
                color='method',
                nbins=40,
                title=f"{metric_info_single['name']} Distribution ({selected_n_ep_single} episodes)",
                labels={'metric_value': metric_info_single['name'], 'count': 'Frequency'},
                opacity=0.7,
                barmode='overlay'
            )

            if metric_info_single['reference_line'] is not None:
                fig1.add_vline(x=metric_info_single['reference_line'], line_dash="dash", line_color="red",
                             annotation_text=metric_info_single['reference_label'])
            fig1.update_layout(height=500)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("**Statistics**")
            summary = filtered_single.groupby('method')['metric_value'].agg(
                mean='mean',
                std='std'
            ).reset_index()
            st.dataframe(summary, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Failed to compute metric: {str(e)}")
    st.code(traceback.format_exc())

st.markdown("---")

# Section 2: Evolution across n_episodes
st.header("📈 Evolution Across Training Sizes")

metric_key_evolution = st.selectbox(
    "Select metric for evolution:",
    list(METRICS.keys()),
    format_func=lambda k: METRICS[k]['name'],
    key="metric_evolution"
)

metric_info_evolution = METRICS[metric_key_evolution]

st.markdown(f"**{metric_info_evolution['name']}** evolution across training data sizes")
st.markdown(f"_Using dataset: **{dataset_key}**_")

try:
    # Compute metric for all n_episodes
    metric_data_all = compute_metric(current_stats, metric_key_evolution)
    filtered_all = metric_data_all[metric_data_all['method'].isin(selected_methods)]

    # Aggregate by method and n_episodes
    evolution_summary = filtered_all.groupby(['method', 'n_episodes'])['metric_value'].agg(
        mean='mean',
        std='std'
    ).reset_index()

    fig_evolution = go.Figure()

    for method in selected_methods:
        method_data = evolution_summary[evolution_summary['method'] == method]
        fig_evolution.add_trace(go.Scatter(
            x=method_data['n_episodes'],
            y=method_data['mean'],
            error_y=dict(type='data', array=method_data['std']),
            mode='lines+markers',
            name=method,
            marker=dict(size=10)
        ))

    if metric_info_evolution['reference_line'] is not None:
        fig_evolution.add_hline(
            y=metric_info_evolution['reference_line'],
            line_dash="dash",
            line_color="red",
            annotation_text=metric_info_evolution['reference_label']
        )

    fig_evolution.update_layout(
        title=f"Mean {metric_info_evolution['name']} vs Training Data Size",
        xaxis_title="Training Episodes",
        yaxis_title=f"Mean {metric_info_evolution['name']} (± std)",
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig_evolution, use_container_width=True)

    # Show data table
    with st.expander("Show evolution data table"):
        st.dataframe(evolution_summary, use_container_width=True)

except Exception as e:
    st.error(f"Failed to create evolution plot: {str(e)}")
    st.code(traceback.format_exc())

st.markdown("---")
st.caption(f"Experiment: {experiment_id} | Policy: {selected_policy_display}")
