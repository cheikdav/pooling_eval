"""Minimal Streamlit dashboard for visualizing value estimator predictions."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path


st.set_page_config(page_title="Value Estimator Analysis", layout="wide")

st.title("Value Estimator Variance Analysis")
st.markdown("Compare variance and performance of different value estimation methods")


@st.cache_data
def load_predictions(predictions_path):
    """Load predictions CSV and compute statistics."""
    df = pd.read_csv(predictions_path)

    stats = df.groupby(['state_idx', 'method', 'n_episodes'])['predicted_value'].agg(
        mean='mean',
        variance='var',
        std='std',
        count='count'
    ).reset_index()

    return df, stats


@st.cache_data
def compute_variance_ratios(stats):
    """Compute variance ratios relative to Monte Carlo."""
    mc_values = stats[stats['method'] == 'monte_carlo'][['state_idx', 'n_episodes', 'variance']]
    mc_values.columns = ['state_idx', 'n_episodes', 'mc_variance']

    merged = stats.merge(mc_values, on=['state_idx', 'n_episodes'])
    merged['variance_ratio'] = merged['variance'] / merged['mc_variance']
    merged['log_variance_ratio'] = np.log(merged['variance_ratio'])

    return merged[merged['method'] != 'monte_carlo'].copy()


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

df, stats = load_predictions(predictions_file)

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

if not selected_methods:
    st.warning("Please select at least one method")
    st.stop()

if not selected_n_episodes:
    st.warning("Please select at least one episode count")
    st.stop()

st.header("1. Variance Ratio Analysis")
st.markdown("Histogram of log variance ratios: log(Method Variance / Monte Carlo Variance)")

variance_ratios = compute_variance_ratios(stats)

filtered_ratios = variance_ratios[
    (variance_ratios['method'].isin(selected_methods)) &
    (variance_ratios['n_episodes'].isin(selected_n_episodes))
]

if len(filtered_ratios) == 0:
    st.warning("No data available for selected filters")
    st.stop()

col1, col2 = st.columns([3, 1])

with col1:
    fig = px.histogram(
        filtered_ratios,
        x='log_variance_ratio',
        color='method',
        facet_col='n_episodes',
        nbins=40,
        title="Log Variance Ratio Distribution by Method and Episode Count",
        labels={'log_variance_ratio': 'Log Variance Ratio: log(Method / MC)', 'count': 'Frequency'},
        opacity=0.7,
        barmode='overlay'
    )

    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Equal to MC")
    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Statistics**")
    summary = filtered_ratios.groupby(['method', 'n_episodes'])['log_variance_ratio'].agg(
        mean='mean',
        std='std'
    ).reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.header("2. Performance Comparison")
st.markdown("Average prediction variance across methods and episode counts")

perf_stats = stats[
    (stats['method'].isin(selected_methods + ['monte_carlo'])) &
    (stats['n_episodes'].isin(selected_n_episodes))
]

perf_summary = perf_stats.groupby(['method', 'n_episodes'])['variance'].mean().reset_index()
perf_summary.columns = ['method', 'n_episodes', 'avg_variance']

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

st.header("3. Variance Ratio vs Training Data Size")
st.markdown("How does the variance ratio change with more training data?")

ratio_summary = filtered_ratios.groupby(['method', 'n_episodes'])['log_variance_ratio'].agg(
    mean='mean',
    std='std'
).reset_index()

fig3 = go.Figure()

for method in selected_methods:
    method_data = ratio_summary[ratio_summary['method'] == method]
    fig3.add_trace(go.Scatter(
        x=method_data['n_episodes'],
        y=method_data['mean'],
        error_y=dict(type='data', array=method_data['std']),
        mode='lines+markers',
        name=method,
        marker=dict(size=10)
    ))

fig3.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Equal to MC")
fig3.update_layout(
    title="Mean Log Variance Ratio vs Training Episodes",
    xaxis_title="Training Episodes",
    yaxis_title="Mean Log Variance Ratio",
    height=500
)

st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption(f"Experiment: {selected_experiment} | Data: {predictions_file}")
