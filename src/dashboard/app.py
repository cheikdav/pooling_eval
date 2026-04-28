"""Dashboard entry point. Sidebar selection and tab routing."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' is importable
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import streamlit as st

from src.dashboard.discovery import discover_experiments
from src.dashboard import tab_metrics, tab_trajectory, tab_paired


def _load_search_paths() -> list:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return [Path(p) for p in cfg.get("search_paths", [])]


SEARCH_PATHS = _load_search_paths()


def main():
    st.set_page_config(page_title="Value Estimation Dashboard", layout="wide")
    st.title("Value Estimation Dashboard")

    # Discover experiments
    entries_df = discover_experiments(SEARCH_PATHS)
    if entries_df.empty:
        st.error("No experiments found. Check search paths.")
        return

    # --- Sidebar ---
    with st.sidebar:
        st.header("Selection")

        # Environment
        envs = sorted(entries_df["env_name"].unique())
        env = st.selectbox("Environment", envs)
        filtered = entries_df[entries_df["env_name"] == env]

        # Policy
        policies = sorted(filtered["policy_display_name"].unique())
        policy = st.selectbox("Policy", policies)
        filtered = filtered[filtered["policy_display_name"] == policy]

        # Methods
        available_methods = sorted(filtered["method"].unique())
        methods = st.multiselect("Methods", available_methods, default=available_methods)
        if not methods:
            st.warning("Select at least one method.")
            return
        filtered = filtered[filtered["method"].isin(methods)]

        # Baseline
        baseline = st.selectbox("Baseline method", methods)

        st.divider()
        st.header("Parameters")

        dataset_type = st.radio("Dataset", ["Full", "Temporal Diff"], horizontal=True,
                               help="Full: per-state metrics. Temporal Diff: within-episode paired differences.")
        temporal_p = None
        if dataset_type == "Temporal Diff":
            temporal_p = st.slider("Temporal p (geometric)", 0.05, 0.8, 0.2, 0.05,
                                   help="Geometric distribution parameter for temporal gaps")

        epsilon = st.number_input("Epsilon (log smoothing)", value=1e-2, format="%.1e")
        n_buckets = st.slider("Decile buckets", 5, 20, 10)
        adjust_constant = st.checkbox("Constant adjustment", value=False,
                                      help="Remove state-independent bias per batch")

        st.divider()
        st.header("Filters")

        filter_high_variance = st.slider("Filter top % variance", 0, 50, 0)
        filter_extreme_mean = st.slider("Filter extreme % mean", 0, 50, 0)

    params = {
        "epsilon": epsilon,
        "n_buckets": n_buckets,
        "adjust_constant": adjust_constant,
        "filter_high_variance": filter_high_variance,
        "filter_extreme_mean": filter_extreme_mean,
        "dataset_type": dataset_type.lower().replace(" ", "_"),
        "temporal_p": temporal_p,
    }

    # --- Tabs ---
    tab_names = ["Metrics", "Trajectories", "Paired States"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_metrics.render(filtered, methods, baseline, params)

    with tabs[1]:
        tab_trajectory.render(filtered, methods, baseline, params)

    with tabs[2]:
        tab_paired.render(filtered, methods, baseline, params)


if __name__ == "__main__":
    main()
