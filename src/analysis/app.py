"""Minimal Streamlit dashboard for visualizing value estimator predictions."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import traceback

from src.analysis.metrics import METRICS, compute_metric


st.set_page_config(page_title="Value Estimator Analysis", layout="wide")

st.title("Value Estimator Variance Analysis")
st.markdown("Compare variance and performance of different value estimation methods")


@st.cache_data
def load_predictions(predictions_path, s1_proportion=0.9, seed=42):
    """Load predictions CSV and compute statistics.

    Returns 4 (df, stats) pairs:
    1. Full dataset
    2. S1 partition (90% of episodes)
    3. S2 partition (10% of episodes)
    4. Differences (S1 states with paired S2 values)
    """
    df = pd.read_csv(predictions_path)

    def compute_stats(dataframe):
        """Helper to compute stats from a predictions dataframe."""
        return dataframe.groupby(['state_idx', 'method', 'n_episodes'])['predicted_value'].agg(
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

predictions_file = experiments_dir / selected_experiment / "results" / "predictions.csv"

if not predictions_file.exists():
    st.error(f"Predictions file not found at {predictions_file}")
    st.info("Run: `python -m src.evaluate --config <your_config>.yaml`")
    st.stop()

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
st.sidebar.markdown(f"- States: {df['state_idx'].nunique()}")
st.sidebar.markdown(f"- Methods: {df['method'].nunique()}")
st.sidebar.markdown(f"- Batches: {df['batch_idx'].nunique()}")
st.sidebar.markdown(f"- Episode subsets: {df['n_episodes'].nunique()}")

all_methods = sorted([m for m in df['method'].unique() if m != 'monte_carlo'])
all_n_episodes = sorted(df['n_episodes'].unique())

selected_methods = st.sidebar.multiselect(
    "Select Methods",
    all_methods,
    default=all_methods,
    help="Choose which methods to compare"
)

selected_n_episodes = st.sidebar.multiselect(
    "Select Episode Counts",
    all_n_episodes,
    default=all_n_episodes,
    help="Choose which training data sizes to compare"
)

st.sidebar.markdown("---")

metric_key = st.sidebar.selectbox(
    "Select Metric",
    options=list(METRICS.keys()),
    format_func=lambda k: METRICS[k]['name'],
    help="Choose which metric to visualize"
)

if not selected_methods:
    st.warning("Please select at least one method")
    st.stop()

if not selected_n_episodes:
    st.warning("Please select at least one episode count")
    st.stop()

metric_info = METRICS[metric_key]

st.header("1. Metric Analysis")
st.markdown(f"**{metric_info['name']}**: {metric_info['description']}")

try:
    metric_data = compute_metric(df, stats, metric_key)
    st.success(f"✓ Computed {metric_info['name']} for {len(metric_data)} state-method-episode combinations")
except Exception as e:
    st.error(f"Failed to compute metric: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

filtered_data = metric_data[
    (metric_data['method'].isin(selected_methods)) &
    (metric_data['n_episodes'].isin(selected_n_episodes))
]

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
            facet_col='n_episodes',
            nbins=40,
            title=f"{metric_info['name']} Distribution by Method and Episode Count",
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
        summary = filtered_data.groupby(['method', 'n_episodes'])['metric_value'].agg(
            mean='mean',
            std='std'
        ).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error computing statistics: {str(e)}")
        st.code(traceback.format_exc())

st.header("2. Performance Comparison")
st.markdown("Average prediction variance across methods and episode counts")

try:
    perf_stats = stats[
        (stats['method'].isin(selected_methods + ['monte_carlo'])) &
        (stats['n_episodes'].isin(selected_n_episodes))
    ]

    perf_summary = perf_stats.groupby(['method', 'n_episodes'])['variance'].mean().reset_index()
    perf_summary.columns = ['method', 'n_episodes', 'avg_variance']

    st.markdown(f"_Rendering bar chart with {len(perf_summary)} method-episode combinations..._")

    fig2 = px.bar(
        perf_summary,
        x='method',
        y='avg_variance',
        color='n_episodes',
        barmode='group',
        title="Average Variance by Method and Training Data Size",
        labels={'avg_variance': 'Average Variance', 'method': 'Method', 'n_episodes': 'Training Episodes'},
        text_auto='.3f'
    )
    fig2.update_layout(height=500)

    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering performance comparison: {str(e)}")
    st.code(traceback.format_exc())

st.header(f"3. {metric_info['name']} vs Training Data Size")
st.markdown(f"How does the {metric_info['name'].lower()} change with more training data?")

try:
    metric_summary = filtered_data.groupby(['method', 'n_episodes'])['metric_value'].agg(
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
