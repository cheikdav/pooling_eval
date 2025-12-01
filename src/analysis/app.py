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
def load_predictions(predictions_path, s1_proportion=0.9, seed=42):
    """Load predictions parquet file and compute statistics.

    Returns 4 (df, stats) pairs:
    1. Full dataset
    2. S1 partition (90% of episodes)
    3. S2 partition (10% of episodes)
    4. Differences (S1 states with paired S2 values)
    """
    df = pd.read_parquet(predictions_path)

    def compute_stats(dataframe):
        """Helper to compute stats from a predictions dataframe."""
        return dataframe.groupby(['state_idx', 'method'])['predicted_value'].agg(
            mean='mean',
            variance='var',
            std='std',
            count='count'
        ).reset_index()

    # Full dataset stats
    stats = compute_stats(df)

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

    stats_s1 = compute_stats(df_s1)
    stats_s2 = compute_stats(df_s2)

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

    # Prepare S2 for merge
    df_s2_renamed = df_s2.rename(columns={
        'state_idx': 'paired_state_idx',
        'predicted_value': 'paired_value'
    })[['paired_state_idx', 'method', 'batch_idx', 'n_episodes', 'paired_value']]

    # Merge to get paired values
    df_merged = df_s1.merge(
        df_s2_renamed,
        on=['paired_state_idx', 'method', 'batch_idx', 'n_episodes'],
        how='inner'
    )

    # Compute differences and rename to match other dataframes
    df_merged['value_difference'] = df_merged['predicted_value'] - df_merged['paired_value']
    df_merged['predicted_value'] = df_merged['value_difference']

    stats_merged = compute_stats(df_merged)

    return (df, stats), (df_s1, stats_s1), (df_s2, stats_s2), (df_merged, stats_merged)


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

# Get all available methods for this environment/policy/n_episodes combination
available_methods = sorted(set(pred['method'] for pred in selected_predictions))

st.sidebar.markdown("---")
st.sidebar.markdown("**Method Selection**")

selected_methods = st.sidebar.multiselect(
    "Select Methods to Compare",
    available_methods,
    default=available_methods,
    help="Choose which methods to compare"
)

if not selected_methods:
    st.warning("Please select at least one method")
    st.stop()

# Load and combine predictions from selected methods
try:
    all_dfs = []
    all_dfs_s1 = []
    all_dfs_s2 = []
    all_dfs_merged = []

    for method in selected_methods:
        # Find prediction file for this method
        method_prediction = next((p for p in selected_predictions if p['method'] == method), None)
        if not method_prediction:
            st.warning(f"No predictions found for method: {method}")
            continue

        predictions_file = Path(method_prediction['predictions_path'])
        if not predictions_file.exists():
            st.warning(f"Predictions file not found: {predictions_file}")
            continue

        (df_method, stats_method), (df_s1_method, stats_s1_method), (df_s2_method, stats_s2_method), (df_merged_method, stats_merged_method) = load_predictions(str(predictions_file))

        # Add method column
        df_method['method'] = method
        df_s1_method['method'] = method
        df_s2_method['method'] = method
        df_merged_method['method'] = method

        all_dfs.append(df_method)
        all_dfs_s1.append(df_s1_method)
        all_dfs_s2.append(df_s2_method)
        all_dfs_merged.append(df_merged_method)

    # Combine all methods
    df = pd.concat(all_dfs, ignore_index=True)
    df_s1 = pd.concat(all_dfs_s1, ignore_index=True)
    df_s2 = pd.concat(all_dfs_s2, ignore_index=True)
    df_merged = pd.concat(all_dfs_merged, ignore_index=True)

    # Recompute stats across all methods
    stats = df.groupby(['state_idx', 'method'])['predicted_value'].agg(
        mean='mean',
        variance='var',
        std='std',
        count='count'
    ).reset_index()

    stats_s1 = df_s1.groupby(['state_idx', 'method'])['predicted_value'].agg(
        mean='mean',
        variance='var',
        std='std',
        count='count'
    ).reset_index()

    stats_s2 = df_s2.groupby(['state_idx', 'method'])['predicted_value'].agg(
        mean='mean',
        variance='var',
        std='std',
        count='count'
    ).reset_index()

    stats_merged = df_merged.groupby(['state_idx', 'method'])['predicted_value'].agg(
        mean='mean',
        variance='var',
        std='std',
        count='count'
    ).reset_index()

    st.sidebar.success(f"✓ Loaded predictions for {len(selected_methods)} method(s)")
    st.sidebar.markdown(f"- Total predictions: {len(df)}")
    st.sidebar.markdown(f"- Batches per method: {df.groupby('method')['batch_idx'].nunique().iloc[0]}")
    st.sidebar.markdown(f"- S1 episodes: {df_s1['episode_idx'].nunique()} ({len(df_s1)} states)")
    st.sidebar.markdown(f"- S2 episodes: {df_s2['episode_idx'].nunique()} ({len(df_s2)} states)")

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

metric_key = st.sidebar.selectbox(
    "Select Metric",
    options=list(METRICS.keys()),
    format_func=lambda k: METRICS[k]['name'],
    help="Choose which metric to visualize"
)

if not selected_methods:
    st.warning("Please select at least one method")
    st.stop()

dataset_map = {
    'full': (df, stats),
    's1': (df_s1, stats_s1),
    's2': (df_s2, stats_s2),
    'differences': (df_merged, stats_merged)
}
current_df, current_stats = dataset_map[dataset_key]

metric_info = METRICS[metric_key]

st.markdown("### Current Selection")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Environment", selected_env)
with col2:
    st.metric("Policy", selected_policy_display)
with col3:
    st.metric("Training Episodes", selected_n_episodes)
with col4:
    st.metric("Methods", len(selected_methods))

st.markdown("---")

st.header("1. Metric Analysis")
st.markdown(f"**{metric_info['name']}**: {metric_info['description']}")
st.markdown(f"_Using dataset: **{dataset_key}** ({len(current_df)} predictions)_")

try:
    metric_data = compute_metric(current_df, current_stats, metric_key)
    st.success(f"✓ Computed {metric_info['name']} for {len(metric_data)} state-method-episode combinations")
except Exception as e:
    st.error(f"Failed to compute metric: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

filtered_data = metric_data[metric_data['method'].isin(selected_methods)]

if len(filtered_data) == 0:
    st.warning("No data available for selected filters")
    st.stop()

col1, col2 = st.columns([3, 1])

with col1:
    try:
        st.markdown(f"_Rendering histogram with {len(filtered_data)} data points..._")
        fig = px.histogram(
            filtered_data,
            x='metric_value',
            color='method',
            nbins=40,
            title=f"{metric_info['name']} Distribution by Method ({selected_policy_display}, {selected_n_episodes} episodes)",
            labels={'metric_value': metric_info['name'], 'count': 'Frequency'},
            opacity=0.7,
            barmode='overlay'
        )

        if metric_info['reference_line'] is not None:
            fig.add_vline(x=metric_info['reference_line'], line_dash="dash", line_color="red",
                         annotation_text=metric_info['reference_label'])
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering histogram: {str(e)}")
        st.code(traceback.format_exc())
        st.write("Debug: First few rows of data being plotted:")
        st.write(filtered_data.head(10))

with col2:
    st.markdown("**Statistics**")
    try:
        summary = filtered_data.groupby('method')['metric_value'].agg(
            mean='mean',
            std='std'
        ).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error computing statistics: {str(e)}")
        st.code(traceback.format_exc())

st.header("2. Performance Comparison")
st.markdown(f"Average prediction variance across methods ({selected_policy_display}, {selected_n_episodes} episodes)")

try:
    perf_stats = current_stats[current_stats['method'].isin(selected_methods + ['monte_carlo'])]

    perf_summary = perf_stats.groupby('method')['variance'].mean().reset_index()
    perf_summary.columns = ['method', 'avg_variance']

    st.markdown(f"_Rendering bar chart with {len(perf_summary)} methods..._")

    fig2 = px.bar(
        perf_summary,
        x='method',
        y='avg_variance',
        title=f"Average Variance by Method ({selected_policy_display}, {selected_n_episodes} episodes)",
        labels={'avg_variance': 'Average Variance', 'method': 'Method'},
        text_auto='.3f'
    )
    fig2.update_layout(height=500)

    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering performance comparison: {str(e)}")
    st.code(traceback.format_exc())

st.header(f"3. Comparison Across Training Data Sizes")
st.markdown(f"Compare performance across different training data sizes for {selected_policy_display}")

try:
    policy_predictions = filter_estimators(all_predictions, 'policy_display_name', selected_policy_display)
    available_n_eps = sorted(set(pred['n_episodes'] for pred in policy_predictions))

    if len(available_n_eps) > 1:
        comparison_data = []
        for n_ep in available_n_eps:
            # Load predictions for each n_episodes and selected methods
            for method in selected_methods:
                method_pred = next((p for p in policy_predictions if p['n_episodes'] == n_ep and p['method'] == method), None)
                if method_pred:
                    temp_file = Path(method_pred['predictions_path'])
                    if temp_file.exists():
                        temp_df = pd.read_parquet(temp_file)
                        temp_df['n_episodes'] = n_ep
                        temp_df['method'] = method
                        comparison_data.append(temp_df)

        if comparison_data:
            combined_df = pd.concat(comparison_data, ignore_index=True)

            combined_stats = combined_df.groupby(['state_idx', 'method', 'n_episodes'])['predicted_value'].agg(
                mean='mean',
                variance='var',
                std='std',
                count='count'
            ).reset_index()

            combined_metric = compute_metric(combined_df, combined_stats, metric_key)

            combined_filtered = combined_metric[combined_metric['method'].isin(selected_methods)]

            metric_summary = combined_filtered.groupby(['method', 'n_episodes'])['metric_value'].agg(
                mean='mean',
                std='std'
            ).reset_index()

            st.markdown(f"_Rendering line plot with {len(metric_summary)} data points across {len(selected_methods)} methods..._")

            fig3 = go.Figure()

            for method in selected_methods:
                method_data = metric_summary[metric_summary['method'] == method]
                fig3.add_trace(go.Scatter(
                    x=method_data['n_episodes'],
                    y=method_data['mean'],
                    error_y=dict(type='data', array=method_data['std']),
                    mode='lines+markers',
                    name=method,
                    marker=dict(size=10)
                ))

            if metric_info['reference_line'] is not None:
                fig3.add_hline(y=metric_info['reference_line'], line_dash="dash", line_color="red",
                              annotation_text=metric_info['reference_label'])
            fig3.update_layout(
                title=f"Mean {metric_info['name']} vs Training Episodes ({selected_policy_display})",
                xaxis_title="Training Episodes",
                yaxis_title=f"Mean {metric_info['name']}",
                height=500
            )

            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No comparison data available")
    else:
        st.info("Only one training data size available for this policy")
except Exception as e:
    st.error(f"Error rendering metric vs training data plot: {str(e)}")
    st.code(traceback.format_exc())

st.markdown("---")
st.caption(f"Experiment: {experiment_id} | Policy: {selected_policy_display} | Training: {selected_n_episodes} episodes")
