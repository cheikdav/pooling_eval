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
    build_selection_tree,
    SelectionTreeNode
)


PRIMARY_SELECTION_KEYS = [
    ('policy_environment', 'Environment'),
    ('policy_display_name', 'Policy'),
    ('n_episodes', 'Training Data Amount'),
]

# Mapping of keys to display names
KEY_DISPLAY_NAMES = {
    'policy_environment': 'Environment',
    'policy_display_name': 'Policy',
    'n_episodes': 'Training Data Amount',
}


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

# Build selection tree
primary_keys = [key for key, _ in PRIMARY_SELECTION_KEYS]
selection_tree = build_selection_tree(all_predictions, primary_keys)

# Initialize session state for selections
if 'selections' not in st.session_state:
    st.session_state.selections = {}
if 'selection_depth' not in st.session_state:
    st.session_state.selection_depth = 0

st.sidebar.header("Hierarchical Selection")

# Navigate down the tree based on current selections
current_node = selection_tree
selection_path = []
selector_counter = 0

# Keys to exclude from additional selectors
exclude_keys = {'method', 'n_batches', 'n_states', 'n_eval_episodes', 'created_at', 'predictions_path', 'metadata_path'}

while not current_node.is_leaf():
    # Automatically navigate single branches
    if current_node.has_single_branch():
        value, child_node = current_node.get_single_child()
        selection_path.append((current_node.branch_key, value))
        current_node = child_node
        continue

    # Need user selection
    branch_key = current_node.branch_key
    available_values = sorted(current_node.children.keys())

    # Get display name for this key
    display_name = KEY_DISPLAY_NAMES.get(branch_key, branch_key.replace('_', ' ').title())

    # Create unique key for this selector
    selector_key = f"selector_{selector_counter}_{branch_key}"
    selector_counter += 1

    # Check if we need to reset selections (upstream change)
    if selector_key not in st.session_state.selections:
        # New selector - clear all downstream selections
        keys_to_remove = [k for k in st.session_state.selections.keys()
                         if k.startswith(f"selector_{selector_counter}")]
        for k in keys_to_remove:
            del st.session_state.selections[k]

    # Get current selection or default to first value
    current_selection = st.session_state.selections.get(selector_key, available_values[0])

    # Handle special formatting
    format_func = None
    if branch_key == 'n_episodes':
        format_func = lambda x: f"{x} episodes"

    # Create selector
    selectbox_kwargs = {
        'label': f"{len(selection_path) + 1}. {display_name}",
        'options': available_values,
        'key': selector_key,
        'help': f"Choose {display_name.lower()}"
    }
    if format_func:
        selectbox_kwargs['format_func'] = format_func

    selected_value = st.sidebar.selectbox(**selectbox_kwargs)

    # Detect if selection changed
    if selected_value != current_selection:
        # Clear downstream selections
        keys_to_remove = [k for k in st.session_state.selections.keys()
                         if k.startswith(f"selector_{selector_counter}")]
        for k in keys_to_remove:
            del st.session_state.selections[k]

    # Update session state
    st.session_state.selections[selector_key] = selected_value

    # Show policy details after policy selection
    if branch_key == 'policy_display_name':
        child_node = current_node.children[selected_value]
        sample_idx = child_node.indices[0]
        sample_pred = all_predictions[sample_idx]
        with st.sidebar.expander("Policy Details"):
            st.markdown(f"**Algorithm:** {sample_pred.get('policy_algorithm')}")
            st.markdown(f"**Seed:** {sample_pred.get('policy_seed')}")
            if sample_pred.get('policy_average_reward') is not None:
                st.markdown(f"**Avg Reward:** {sample_pred.get('policy_average_reward'):.2f}")
            st.markdown(f"**Learning Rate:** {sample_pred.get('policy_learning_rate')}")
            st.markdown(f"**Gamma:** {sample_pred.get('policy_gamma')}")
            st.markdown(f"**Total Timesteps:** {sample_pred.get('policy_total_timesteps')}")

    # Move to next node
    selection_path.append((branch_key, selected_value))
    current_node = current_node.children[selected_value]

# Get final predictions
selected_predictions = [all_predictions[i] for i in current_node.indices]

if not selected_predictions:
    st.error("No predictions found for the selected criteria")
    st.stop()

# Extract selected values from selection path
selected_values = {key: value for key, value in selection_path}

sample_prediction = selected_predictions[0]
experiment_id = sample_prediction['experiment_id']
selected_env = selected_values.get('policy_environment', sample_prediction.get('policy_environment'))
selected_policy_display = selected_values.get('policy_display_name', sample_prediction.get('policy_display_name'))
selected_n_episodes = selected_values.get('n_episodes', sample_prediction.get('n_episodes'))

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


@st.cache_data
def load_stats_for_n_episodes(selected_predictions_json, n_ep, methods):
    """Load stats for a specific n_episodes value.

    Args:
        selected_predictions_json: JSON string of selected predictions (for cache key)
        n_ep: Number of episodes to load
        methods: List of methods to load

    Returns:
        Tuple of (stats, stats_s1, stats_s2, stats_merged) for this n_episodes
    """
    import json
    selected_preds = json.loads(selected_predictions_json)

    all_stats = []
    all_stats_s1 = []
    all_stats_s2 = []
    all_stats_merged = []

    for method in methods:
        method_prediction = next((p for p in selected_preds
                                 if p['method'] == method and p['n_episodes'] == n_ep), None)
        if not method_prediction:
            continue

        predictions_file = Path(method_prediction['predictions_path'])
        if not predictions_file.exists():
            continue

        stats, stats_s1, stats_s2, stats_merged = load_and_compute_stats(
            str(predictions_file),
            method,
            n_ep
        )

        all_stats.append(stats)
        all_stats_s1.append(stats_s1)
        all_stats_s2.append(stats_s2)
        all_stats_merged.append(stats_merged)

    if not all_stats:
        return None, None, None, None

    return (
        pd.concat(all_stats, ignore_index=True),
        pd.concat(all_stats_s1, ignore_index=True),
        pd.concat(all_stats_s2, ignore_index=True),
        pd.concat(all_stats_merged, ignore_index=True)
    )

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

# Load stats for selected n_episodes only
import json
try:
    methods_to_load = ['monte_carlo'] + selected_methods if 'monte_carlo' not in selected_methods else selected_methods
    selected_predictions_json = json.dumps(selected_predictions)

    stats_single, stats_s1_single, stats_s2_single, stats_merged_single = load_stats_for_n_episodes(
        selected_predictions_json,
        selected_n_ep_single,
        methods_to_load
    )

    if stats_single is None:
        st.error(f"No data available for {selected_n_ep_single} episodes")
        st.stop()

    # Map dataset selection to appropriate stats
    dataset_map_single = {
        'full': stats_single,
        's1': stats_s1_single,
        's2': stats_s2_single,
        'differences': stats_merged_single
    }
    stats_single_n_ep = dataset_map_single[dataset_key]

except Exception as e:
    st.error(f"Failed to load stats for {selected_n_ep_single} episodes: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

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
    # Load stats incrementally for each n_episodes and compute evolution
    evolution_data = []

    progress_bar = st.progress(0, text="Loading data for evolution plot...")

    for idx, n_ep in enumerate(all_n_episodes_for_policy):
        stats_full_ep, stats_s1_ep, stats_s2_ep, stats_merged_ep = load_stats_for_n_episodes(
            selected_predictions_json,
            n_ep,
            methods_to_load
        )

        if stats_full_ep is None:
            continue

        dataset_map_ep = {
            'full': stats_full_ep,
            's1': stats_s1_ep,
            's2': stats_s2_ep,
            'differences': stats_merged_ep
        }
        current_stats_ep = dataset_map_ep[dataset_key]

        # Compute metric for this n_episodes
        metric_data_ep = compute_metric(current_stats_ep, metric_key_evolution)
        filtered_ep = metric_data_ep[metric_data_ep['method'].isin(selected_methods)]

        # Aggregate by method
        summary_ep = filtered_ep.groupby(['method'])['metric_value'].agg(
            mean='mean',
            std='std'
        ).reset_index()
        summary_ep['n_episodes'] = n_ep

        evolution_data.append(summary_ep)

        progress_bar.progress((idx + 1) / len(all_n_episodes_for_policy),
                            text=f"Loading {n_ep} episodes...")

    progress_bar.empty()

    if not evolution_data:
        st.warning("No data available for evolution plot")
        st.stop()

    # Combine all evolution data
    evolution_summary = pd.concat(evolution_data, ignore_index=True)

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
