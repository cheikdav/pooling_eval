"""Streamlit dashboard for value estimator analysis."""

from pathlib import Path
import streamlit as st
import pandas as pd

from common import load_predictions_data, get_method_display_name
import tab_absolute
import tab_comparison
import tab_trajectory


st.set_page_config(page_title="Value Estimator Analysis", layout="wide")


def show_selection_filters(metadata_df):
    """Display sidebar filters and return filtered metadata."""
    st.sidebar.header("Filters")

    # Dataset type selection
    dataset_type = st.sidebar.radio(
        "Dataset Type",
        options=['full', 'differences', 'temporal'],
        format_func=lambda x: {
            'full': 'Full Dataset',
            'differences': 'Paired Differences (S1 - S2)',
            'temporal': 'Temporal Differences (Within Episode)'
        }[x],
        help="Full: all states | Differences: V(s) - V(s') paired across episodes | Temporal: V(s_t) - V(s_{t+δ}) within episodes, δ ~ Geometric(p)"
    )

    # Temporal difference parameter (only show if temporal mode selected)
    if dataset_type == 'temporal':
        temporal_p = st.sidebar.slider(
            "Geometric Parameter (p)",
            min_value=0.05,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Parameter for geometric distribution: smaller p = larger average gaps"
        )
        st.sidebar.caption(f"Average δ ≈ {1/temporal_p:.1f} steps")
    else:
        temporal_p = 0.2  # Default value when not in temporal mode

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

    # Number of buckets for decile-based metrics
    n_buckets = st.sidebar.slider(
        "Number of Buckets",
        min_value=2,
        max_value=20,
        value=10,
        step=1,
        help="Number of buckets for decile-based metrics (variance by value decile)"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Filtering**")

    # Filtering options
    filter_high_variance = st.sidebar.slider(
        "Exclude Top % Variance",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
        help="Exclude states with top x% variance (outliers)"
    )

    filter_extreme_mean = st.sidebar.slider(
        "Exclude Top/Bottom % Mean",
        min_value=0,
        max_value=25,
        value=0,
        step=1,
        help="Exclude states with extreme mean values (top x% and bottom x%)"
    )

    # Ensure baseline method is included in the data
    methods_to_load = list(set(selected_methods + [baseline_method]))
    filtered = filtered[filtered['method'].isin(methods_to_load)]

    return filtered, selected_env, selected_policy, selected_methods, baseline_method, epsilon, dataset_type, n_buckets, filter_high_variance, filter_extreme_mean, temporal_p


# Main App

st.title("Value Estimator Variance Analysis")
st.markdown("Compare variance and performance of different value estimation methods")

# Load data - support multiple search paths
default_paths = [
    "experiments",  # Local experiments
    "/scratch/dc3430/pooling_eval/experiments"  # Scratch filesystem
]

# Try each path and collect all predictions
all_metadata = []
for exp_path in default_paths:
    experiments_dir = Path(exp_path)
    if experiments_dir.exists():
        st.sidebar.info(f"Searching: {experiments_dir}")
        df = load_predictions_data(str(experiments_dir))
        if not df.empty:
            all_metadata.append(df)

if not all_metadata:
    st.error(f"No predictions found in any search paths: {default_paths}")
    st.stop()

metadata_df = pd.concat(all_metadata, ignore_index=True)
if metadata_df.empty:
    st.error("No predictions found. Run: `uv run -m src.evaluate --config <config>.yaml`")
    st.stop()

# Sidebar filters
filtered_metadata, env, policy, methods, baseline_method, epsilon, dataset_type, n_buckets, filter_high_variance, filter_extreme_mean, temporal_p = show_selection_filters(metadata_df)

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
    st.metric("Baseline", get_method_display_name(baseline_method))
with col4:
    dataset_label = {
        'full': 'Full',
        'differences': 'Differences',
        'temporal': 'Temporal'
    }[dataset_type]
    st.metric("Dataset", dataset_label)

if filter_high_variance > 0 or filter_extreme_mean > 0:
    filters_active = []
    if filter_high_variance > 0:
        filters_active.append(f"Excluding top {filter_high_variance}% variance")
    if filter_extreme_mean > 0:
        filters_active.append(f"Excluding top/bottom {filter_extreme_mean}% mean")
    st.info(f"**Active Filters:** {', '.join(filters_active)}")

st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Absolute Metrics", "Comparison Metrics", "Episode Trajectories"])

with tab1:
    tab_absolute.render_tab(
        filtered_metadata, methods, baseline_method, epsilon, dataset_type,
        n_buckets, filter_high_variance, filter_extreme_mean, temporal_p
    )

with tab2:
    tab_comparison.render_tab(
        filtered_metadata, methods, baseline_method, epsilon, dataset_type,
        n_buckets, filter_high_variance, filter_extreme_mean, temporal_p
    )

with tab3:
    tab_trajectory.render_tab(
        filtered_metadata, methods, baseline_method
    )

st.markdown("---")
display_methods = [get_method_display_name(m) for m in methods]
st.caption(f"Policy: {policy} | Methods: {', '.join(display_methods)}")
