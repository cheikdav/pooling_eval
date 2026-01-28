"""Common utilities for the Streamlit dashboard."""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from metadata_discovery import discover_predictions


METHOD_DISPLAY_NAMES = {
    'dqn': 'td',
    'monte_carlo': 'monte_carlo'
}


def get_method_display_name(method):
    """Get display name for a method."""
    return METHOD_DISPLAY_NAMES.get(method, method)


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


def apply_data_filters(stats, filter_high_variance=0, filter_extreme_mean=0):
    """Apply filtering to remove outliers from statistics.

    Args:
        stats: DataFrame with variance and mean columns
        filter_high_variance: Percentage of top variance states to exclude (0-50)
        filter_extreme_mean: Percentage of top and bottom mean states to exclude (0-25)

    Returns:
        Filtered DataFrame
    """
    result = stats.copy()

    # Filter high variance outliers
    if filter_high_variance > 0:
        variance_threshold = np.percentile(result['variance'], 100 - filter_high_variance)
        result = result[result['variance'] <= variance_threshold]

    # Filter extreme mean values
    if filter_extreme_mean > 0:
        mean_lower = np.percentile(result['mean'], filter_extreme_mean)
        mean_upper = np.percentile(result['mean'], 100 - filter_extreme_mean)
        result = result[(result['mean'] >= mean_lower) & (result['mean'] <= mean_upper)]

    return result


@st.cache_data
def load_episode_data(experiment_path):
    """Load evaluation batch to get episode structure.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        Dictionary with episode data or None if not found
    """
    batch_path = Path(experiment_path) / "data" / "batch_eval.npz"
    if not batch_path.exists():
        return None

    batch = np.load(batch_path, allow_pickle=True)
    return dict(batch)


@st.cache_data
def load_predictions_for_trajectory(predictions_path):
    """Load predictions and compute mean across batches for each state.

    Args:
        predictions_path: Path to predictions.parquet file

    Returns:
        DataFrame with state_idx, episode_idx, mean_value
    """
    df = pd.read_parquet(predictions_path)

    # Compute mean across batches for each state
    mean_df = df.groupby(['state_idx', 'episode_idx'])['predicted_value'].mean().reset_index()
    mean_df.rename(columns={'predicted_value': 'mean_value'}, inplace=True)

    return mean_df
