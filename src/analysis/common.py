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
def compute_stats_from_predictions(predictions_path, n_episodes, dataset_type='full', s1_proportion=0.9, seed=42, temporal_p=0.2):
    """Load predictions and compute statistics aggregated across batches.

    Memory-efficient: loads raw data, computes stats, then frees raw data.
    Only the aggregated stats (one row per state) are cached.

    Args:
        predictions_path: Path to predictions.parquet file
        n_episodes: Number of episodes (for metadata only)
        dataset_type: 'full' for all data, 'differences' for paired differences,
                     'temporal' for within-episode temporal differences
        s1_proportion: Proportion of episodes for S1 partition (used for differences)
        seed: Random seed for episode partitioning
        temporal_p: Geometric distribution parameter for temporal gaps (used for temporal mode)

    Returns:
        DataFrame with columns: state_idx, n_episodes, mean, variance, std, count
        where statistics are aggregated across all batches for each state
        Also includes metadata: predictions_path, results_dir (for ground truth access)
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

        # Add metadata for metrics that need access to raw predictions or ground truth
        # predictions_path format: experiments/<exp_id>/results/<method>/<n_episodes>/predictions.parquet
        # results_dir: experiments/<exp_id>/results
        predictions_path_obj = Path(predictions_path)
        results_dir = str(predictions_path_obj.parent.parent.parent)
        stats['predictions_path'] = predictions_path
        stats['results_dir'] = results_dir

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

        # Add metadata for metrics that need access to raw predictions or ground truth
        predictions_path_obj = Path(predictions_path)
        results_dir = str(predictions_path_obj.parent.parent.parent)
        stats_diff['predictions_path'] = predictions_path
        stats_diff['results_dir'] = results_dir

        del df, df_s1, df_s2, df_merged
        return stats_diff

    elif dataset_type == 'temporal':
        # Within-episode temporal differences: V(s_t) - V(s_{t+δ}) where δ ~ Geometric(p)
        # Sample multiple non-overlapping intervals per episode with buffer between them
        np.random.seed(seed)

        temporal_diffs = []

        for (ep_idx, batch_name), ep_group in df.groupby(['episode_idx', 'batch_name']):
            # Sort by state_idx to get temporal order within episode
            ep_group = ep_group.sort_values('state_idx').reset_index(drop=True)
            episode_length = len(ep_group)

            # Sequentially sample non-overlapping intervals
            t = 0
            while t < episode_length:
                # Sample gap from geometric distribution (returns values >= 1)
                delta = np.random.geometric(temporal_p)

                # Check if we can create a valid pair
                if t + delta < episode_length:
                    state_t = ep_group.iloc[t]['state_idx']
                    state_t_delta = ep_group.iloc[t + delta]['state_idx']
                    value_t = ep_group.iloc[t]['predicted_value']
                    value_t_delta = ep_group.iloc[t + delta]['predicted_value']

                    # Compute difference: V(s_t) - V(s_{t+δ})
                    difference = value_t - value_t_delta

                    temporal_diffs.append({
                        'state_idx': state_t,
                        'paired_state_idx': state_t_delta,
                        'batch_name': batch_name,
                        'episode_idx': ep_idx,
                        'difference': difference,
                        'delta': delta
                    })

                    # Move past this interval plus a small buffer
                    # Buffer is also sampled from geometric to avoid fixed patterns
                    buffer = np.random.geometric(temporal_p)
                    t = t + delta + buffer
                else:
                    # Can't fit another interval, move to next episode
                    break

        if not temporal_diffs:
            # No valid pairs found
            stats_temporal = pd.DataFrame(columns=['state_idx', 'n_episodes', 'mean', 'variance', 'std', 'count'])
        else:
            df_temporal = pd.DataFrame(temporal_diffs)

            # Aggregate statistics on differences by state_idx
            stats_temporal = df_temporal.groupby('state_idx')['difference'].agg(
                mean='mean',
                variance='var',
                std='std',
                count='count'
            ).reset_index()

        stats_temporal['n_episodes'] = n_episodes

        # Add metadata
        predictions_path_obj = Path(predictions_path)
        results_dir = str(predictions_path_obj.parent.parent.parent)
        stats_temporal['predictions_path'] = predictions_path
        stats_temporal['results_dir'] = results_dir

        del df, temporal_diffs
        return stats_temporal

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
def load_predictions_for_trajectory(predictions_path):
    """Load predictions and compute mean across batches for each state.

    Args:
        predictions_path: Path to predictions.parquet file

    Returns:
        DataFrame with episode_idx, state_idx, mean_value, step_in_episode
    """
    df = pd.read_parquet(predictions_path)

    # Compute mean across batches for each state
    mean_df = df.groupby(['state_idx', 'episode_idx'])['predicted_value'].mean().reset_index()
    mean_df.rename(columns={'predicted_value': 'mean_value'}, inplace=True)

    # For each episode, create step_in_episode (0, 1, 2, ...) by sorting by state_idx
    def add_step_index(group):
        group = group.sort_values('state_idx')
        group['step_in_episode'] = range(len(group))
        return group

    mean_df = mean_df.groupby('episode_idx', group_keys=False).apply(add_step_index)

    return mean_df


@st.cache_data
def load_ground_truth_returns(results_path):
    """Load ground truth returns from results directory.

    Args:
        results_path: Path to results directory (e.g., experiments/exp_id/results)

    Returns:
        DataFrame with episode_idx, state_idx, ground_truth_return, step_in_episode
        or None if ground truth file doesn't exist
    """
    ground_truth_file = Path(results_path) / "ground_truth" / "ground_truth_returns.parquet"

    if not ground_truth_file.exists():
        return None

    df = pd.read_parquet(ground_truth_file)

    # For each episode, create step_in_episode (0, 1, 2, ...) by sorting by state_idx
    def add_step_index(group):
        group = group.sort_values('state_idx')
        group['step_in_episode'] = range(len(group))
        return group

    df = df.groupby('episode_idx', group_keys=False).apply(add_step_index)

    return df
