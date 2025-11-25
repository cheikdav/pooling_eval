"""Minimal Streamlit dashboard for visualizing value estimator predictions."""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import traceback

from metrics import METRICS, compute_metric


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

experiment_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])
experiment_names = [d.name for d in experiment_dirs]

if not experiment_names:
    st.error("No experiments found in experiments directory")
    st.stop()

st.sidebar.header("Filters")

selected_experiment = st.sidebar.selectbox(
    "Select Experiment",
    experiment_names,
    help="Choose which experiment to analyze"
)

# Discover available n_episodes files
results_dir = experiments_dir / selected_experiment / "results"
if not results_dir.exists():
    st.error(f"Results directory not found at {results_dir}")
    st.info("Run: `python -m src.evaluate --config <your_config>.yaml`")
    st.stop()

prediction_files = sorted(results_dir.glob("predictions_*.parquet"))
if not prediction_files:
    st.error(f"No prediction files found in {results_dir}")
    st.info("Run: `python -m src.evaluate --config <your_config>.yaml`")
    st.stop()

# Extract n_episodes from filenames
available_n_episodes = []
for f in prediction_files:
    try:
        n_ep = int(f.stem.replace("predictions_", ""))
        available_n_episodes.append(n_ep)
    except ValueError:
        continue

if not available_n_episodes:
    st.error("No valid prediction files found")
    st.stop()

selected_n_episodes_file = st.sidebar.selectbox(
    "Select Training Data Size",
    sorted(available_n_episodes),
    help="Choose which training data size to analyze"
)

predictions_file = results_dir / f"predictions_{selected_n_episodes_file}.parquet"

try:
    (df, stats), (df_s1, stats_s1), (df_s2, stats_s2), (df_merged, stats_merged) = load_predictions(str(predictions_file))
    st.sidebar.success(f"✓ Loaded {len(df)} predictions from {df['batch_idx'].nunique()} batches")
    st.sidebar.markdown(f"- S1 episodes: {df_s1['episode_idx'].nunique()} ({len(df_s1)} states)")
    st.sidebar.markdown(f"- S2 episodes: {df_s2['episode_idx'].nunique()} ({len(df_s2)} states)")
    st.sidebar.markdown(f"- Paired states: {len(df_merged)}")
except Exception as e:
    st.error(f"Failed to load predictions file: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

st.sidebar.markdown(f"**Data Summary**")
st.sidebar.markdown(f"- Training episodes: {selected_n_episodes_file}")
st.sidebar.markdown(f"- States: {df['state_idx'].nunique()}")
st.sidebar.markdown(f"- Methods: {df['method'].nunique()}")
st.sidebar.markdown(f"- Batches: {df['batch_idx'].nunique()}")

all_methods = sorted([m for m in df['method'].unique() if m != 'monte_carlo' and pd.notna(m)])

selected_methods = st.sidebar.multiselect(
    "Select Methods",
    all_methods,
    default=all_methods,
    help="Choose which methods to compare"
)

st.sidebar.markdown("---")

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

# Select the appropriate dataset based on user choice
dataset_map = {
    'full': (df, stats),
    's1': (df_s1, stats_s1),
    's2': (df_s2, stats_s2),
    'differences': (df_merged, stats_merged)
}
current_df, current_stats = dataset_map[dataset_key]

metric_info = METRICS[metric_key]

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
            title=f"{metric_info['name']} Distribution by Method (Training: {selected_n_episodes_file} episodes)",
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
st.markdown(f"Average prediction variance across methods (Training: {selected_n_episodes_file} episodes)")

try:
    perf_stats = current_stats[current_stats['method'].isin(selected_methods + ['monte_carlo'])]

    perf_summary = perf_stats.groupby('method')['variance'].mean().reset_index()
    perf_summary.columns = ['method', 'avg_variance']

    st.markdown(f"_Rendering bar chart with {len(perf_summary)} methods..._")

    fig2 = px.bar(
        perf_summary,
        x='method',
        y='avg_variance',
        title=f"Average Variance by Method (Training: {selected_n_episodes_file} episodes)",
        labels={'avg_variance': 'Average Variance', 'method': 'Method'},
        text_auto='.3f'
    )
    fig2.update_layout(height=500)

    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering performance comparison: {str(e)}")
    st.code(traceback.format_exc())

st.header(f"3. Comparison Across Training Data Sizes")
st.markdown(f"Compare performance across different training data sizes")

try:
    # Load all available n_episodes files for comparison
    comparison_data = []
    for n_ep in available_n_episodes:
        temp_file = results_dir / f"predictions_{n_ep}.parquet"
        temp_df = pd.read_parquet(temp_file)
        temp_df['n_episodes'] = n_ep
        comparison_data.append(temp_df)

    combined_df = pd.concat(comparison_data, ignore_index=True)

    # Compute stats for combined data
    combined_stats = combined_df.groupby(['state_idx', 'method', 'n_episodes'])['predicted_value'].agg(
        mean='mean',
        variance='var',
        std='std',
        count='count'
    ).reset_index()

    # Compute metric across all n_episodes
    combined_metric = compute_metric(combined_df, combined_stats, metric_key)

    # Filter by selected methods
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
        title=f"Mean {metric_info['name']} vs Training Episodes",
        xaxis_title="Training Episodes",
        yaxis_title=f"Mean {metric_info['name']}",
        height=500
    )

    st.plotly_chart(fig3, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering metric vs training data plot: {str(e)}")
    st.code(traceback.format_exc())

st.markdown("---")
st.caption(f"Experiment: {selected_experiment} | Data: {predictions_file}")
