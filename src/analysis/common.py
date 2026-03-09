"""Common utilities for the Streamlit dashboard."""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
from metadata_discovery import discover_predictions


@dataclass
class MetricContext:
    """All data and parameters needed to compute and plot metrics.

    This centralizes all the data and settings in one place, making it easier
    to pass around and audit what data is being used for metric computation.
    """
    # Method data - stats for each method being compared
    method_stats: Dict[str, pd.DataFrame]  # method_name -> stats DataFrame

    # Baseline method for comparison metrics
    baseline_method: str

    # Methods to display in plots
    methods_to_display: List[str]

    # Ground truth data (loaded once, shared by all metrics that need it)
    ground_truth_stats: Optional[pd.DataFrame] = None

    # Display settings
    n_episodes: int = 0  # Number of training episodes for current view

    # Computation parameters
    epsilon: float = 1e-10  # Small value for log computations
    n_buckets: int = 10  # Number of buckets for decile-based metrics

    # Dataset configuration
    dataset_type: str = 'full'  # 'full', 'differences', or 'temporal'
    temporal_p: float = 0.2  # Geometric parameter for temporal differences
    seed: int = 42  # Random seed for partitioning
    s1_proportion: float = 0.9  # Proportion for S1 partition in differences mode

    def get_baseline_stats(self) -> Optional[pd.DataFrame]:
        """Get baseline method stats if available."""
        return self.method_stats.get(self.baseline_method)


METHOD_DISPLAY_NAMES = {
    'dqn': 'td',
    'monte_carlo': 'monte_carlo'
}

METHOD_ORDER = [
    'monte_carlo', 'dqn',
    'least_squares_mc', 'least_squares_td',
    'least_squares_mc_rbf', 'least_squares_td_rbf',
    'nnls_mc', 'nnls_td',
]


def sort_methods(methods):
    """Sort methods by METHOD_ORDER; unknown methods go last."""
    def key(m):
        try:
            return (METHOD_ORDER.index(m), m)
        except ValueError:
            return (len(METHOD_ORDER), m)
    return sorted(methods, key=key)


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
def _get_ground_truth_mean(results_dir):
    """Load ground truth and return mean value (cached).

    Args:
        results_dir: Path to results directory

    Returns:
        Mean ground truth value, or None if file doesn't exist
    """
    ground_truth_file = Path(results_dir) / "ground_truth" / "ground_truth_returns.parquet"

    if not ground_truth_file.exists():
        return None

    ground_truth_df = pd.read_parquet(ground_truth_file)
    return ground_truth_df['ground_truth_return'].mean()


@st.cache_data
def compute_ground_truth_stats(results_dir, dataset_type='full', s1_proportion=0.9, seed=42, temporal_p=0.2):
    """Load ground truth and compute statistics matching the dataset type.

    Args:
        results_dir: Path to results directory
        dataset_type: 'full' for all data, 'differences' for paired differences,
                     'temporal' for within-episode temporal differences
        s1_proportion: Proportion of episodes for S1 partition (used for differences)
        seed: Random seed for episode partitioning
        temporal_p: Geometric distribution parameter for temporal gaps (used for temporal mode)

    Returns:
        DataFrame with columns: state_idx, ground_truth_value
        For differences/temporal modes, returns difference values
    """
    ground_truth_file = Path(results_dir) / "ground_truth" / "ground_truth_returns.parquet"

    if not ground_truth_file.exists():
        return pd.DataFrame(columns=['state_idx', 'ground_truth_value'])

    df = pd.read_parquet(ground_truth_file)

    if dataset_type == 'full':
        # Simple case: just return ground truth values
        result = df[['state_idx', 'ground_truth_return']].copy()
        result.rename(columns={'ground_truth_return': 'ground_truth_value'}, inplace=True)
        return result

    elif dataset_type == 'differences':
        # Split episodes into S1 (90%) and S2 (10%) - same logic as predictions
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

        # Compute differences for each pairing
        differences = []
        for s1_idx, s2_idx in pairings.items():
            s1_value = df_s1[df_s1['state_idx'] == s1_idx]['ground_truth_return'].iloc[0]
            s2_value = df_s2[df_s2['state_idx'] == s2_idx]['ground_truth_return'].iloc[0]
            differences.append({
                'state_idx': s1_idx,
                'ground_truth_value': s1_value - s2_value
            })

        return pd.DataFrame(differences)

    elif dataset_type == 'temporal':
        # Within-episode temporal differences - same logic as predictions
        np.random.seed(seed)

        # Get episode info
        episode_info = df.groupby('episode_idx').agg({
            'state_idx': ['min', 'count']
        }).reset_index()
        episode_info.columns = ['episode_idx', 'min_state_idx', 'episode_length']

        # Sample temporal pairs
        n_episodes = len(episode_info)
        max_episode_length = int(episode_info['episode_length'].max())
        max_pairs = max_episode_length // 2 + 1

        deltas = np.random.geometric(temporal_p, size=(n_episodes, max_pairs))
        buffers = np.random.geometric(temporal_p, size=(n_episodes, max_pairs))

        steps = deltas + buffers
        positions = np.concatenate([np.zeros((n_episodes, 1), dtype=int), np.cumsum(steps[:, :-1], axis=1)], axis=1)

        episode_lengths = episode_info['episode_length'].values[:, np.newaxis]
        valid_mask = (positions + deltas) < episode_lengths

        valid_episode_idx, valid_pair_idx = np.where(valid_mask)

        if len(valid_episode_idx) == 0:
            return pd.DataFrame(columns=['state_idx', 'ground_truth_value'])

        valid_positions = positions[valid_episode_idx, valid_pair_idx]
        valid_deltas = deltas[valid_episode_idx, valid_pair_idx]

        actual_episode_idx = episode_info.iloc[valid_episode_idx]['episode_idx'].values
        min_state_idx = episode_info.iloc[valid_episode_idx]['min_state_idx'].values

        state_t_idx = min_state_idx + valid_positions
        state_t_delta_idx = min_state_idx + valid_positions + valid_deltas

        # Get ground truth values
        differences = []
        for ep_idx, st_idx, st_delta_idx in zip(actual_episode_idx, state_t_idx, state_t_delta_idx):
            ep_df = df[df['episode_idx'] == ep_idx]
            value_t = ep_df[ep_df['state_idx'] == st_idx]['ground_truth_return'].iloc[0]
            value_t_delta = ep_df[ep_df['state_idx'] == st_delta_idx]['ground_truth_return'].iloc[0]
            differences.append({
                'state_idx': st_idx,
                'ground_truth_value': value_t - value_t_delta
            })

        return pd.DataFrame(differences)

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


@st.cache_data
def compute_all_batch_constants(filtered_metadata, methods, n_episodes):
    """Compute per-batch adjustment constants for all selected methods.

    Args:
        filtered_metadata: DataFrame with experiment metadata
        methods: List of method names to compute constants for
        n_episodes: Number of training episodes to filter by

    Returns:
        DataFrame with columns: batch_name, constant, method
        Returns empty DataFrame if no data available
    """
    all_constants = []

    # Filter to selected n_episodes
    filtered_for_n_ep = filtered_metadata[filtered_metadata['n_episodes'] == n_episodes]

    for _, row in filtered_for_n_ep.iterrows():
        if row['method'] not in methods:
            continue

        predictions_path = row['predictions_path']
        method_name = row['method']

        # Get ground truth mean
        predictions_path_obj = Path(predictions_path)
        results_dir = str(predictions_path_obj.parent.parent.parent)
        mean_ground_truth = _get_ground_truth_mean(results_dir)

        if mean_ground_truth is None:
            continue

        # Load predictions and compute constant for each batch
        df = pd.read_parquet(predictions_path)

        for batch_name, batch_df in df.groupby('batch_name'):
            mean_batch = batch_df['predicted_value'].mean()
            constant = mean_ground_truth - mean_batch
            all_constants.append({
                'batch_name': batch_name,
                'constant': constant,
                'method': method_name
            })

    return pd.DataFrame(all_constants)


@st.cache_data
def compute_stats_from_predictions(predictions_path, n_episodes, dataset_type='full', s1_proportion=0.9, seed=42, temporal_p=0.2, adjust_constant=False):
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
        adjust_constant: If True, add constant so mean(predictions) = mean(ground_truth)

    Returns:
        DataFrame with columns: state_idx, n_episodes, mean, variance, std, count
        where statistics are aggregated across all batches for each state
        Also includes metadata: predictions_path, results_dir (for ground truth access)
    """
    # Load raw predictions
    df = pd.read_parquet(predictions_path)

    # Apply constant adjustment if requested
    # Each batch gets its own constant: c_batch = mean(ground_truth) - mean(predictions_batch)
    if adjust_constant:
        # Get cached ground truth mean
        predictions_path_obj = Path(predictions_path)
        results_dir = str(predictions_path_obj.parent.parent.parent)
        mean_ground_truth = _get_ground_truth_mean(results_dir)

        if mean_ground_truth is not None:
            # Compute and apply constant for each batch separately
            def adjust_batch(batch_df):
                mean_batch_predictions = batch_df['predicted_value'].mean()
                constant = mean_ground_truth - mean_batch_predictions
                batch_df = batch_df.copy()
                batch_df['predicted_value'] = batch_df['predicted_value'] + constant
                return batch_df

            df = df.groupby('batch_name', group_keys=False).apply(adjust_batch)

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
        # Strategy: Sample pairs ONCE per episode, then compute differences across all batches
        np.random.seed(seed)

        # Step 1: Get episode lengths efficiently
        first_batch = df['batch_name'].unique()[0]
        episode_info = df[df['batch_name'] == first_batch].groupby('episode_idx').agg({
            'state_idx': ['min', 'count']
        }).reset_index()
        episode_info.columns = ['episode_idx', 'min_state_idx', 'episode_length']

        # Step 2: Sample temporal pairs for all episodes (fully vectorized)
        n_episodes = len(episode_info)
        max_episode_length = int(episode_info['episode_length'].max())
        max_pairs = max_episode_length // 2 + 1

        # Sample deltas and buffers for all episodes at once (2D arrays)
        deltas = np.random.geometric(temporal_p, size=(n_episodes, max_pairs))
        buffers = np.random.geometric(temporal_p, size=(n_episodes, max_pairs))

        # Compute starting positions for all episodes
        steps = deltas + buffers
        positions = np.concatenate([np.zeros((n_episodes, 1), dtype=int), np.cumsum(steps[:, :-1], axis=1)], axis=1)

        # Create mask for valid pairs (both t and t+delta must be within episode)
        episode_lengths = episode_info['episode_length'].values[:, np.newaxis]  # Shape: (n_episodes, 1)
        valid_mask = (positions + deltas) < episode_lengths

        # Extract valid pairs
        valid_episode_idx, valid_pair_idx = np.where(valid_mask)

        if len(valid_episode_idx) == 0:
            stats_temporal = pd.DataFrame(columns=['state_idx', 'n_episodes', 'mean', 'variance', 'std', 'count'])
        else:
            # Get corresponding values
            valid_positions = positions[valid_episode_idx, valid_pair_idx]
            valid_deltas = deltas[valid_episode_idx, valid_pair_idx]

            # Map to actual episode indices and state indices
            actual_episode_idx = episode_info.iloc[valid_episode_idx]['episode_idx'].values
            min_state_idx = episode_info.iloc[valid_episode_idx]['min_state_idx'].values

            state_t_idx = min_state_idx + valid_positions
            state_t_delta_idx = min_state_idx + valid_positions + valid_deltas

            # Step 3: Create pairs DataFrame
            pairs_df = pd.DataFrame({
                'episode_idx': actual_episode_idx,
                'state_t_idx': state_t_idx,
                'state_t_delta_idx': state_t_delta_idx
            })

            # Step 4: Merge with predictions to get value_t for all batches
            pairs_with_t = pairs_df.merge(
                df[['episode_idx', 'state_idx', 'batch_name', 'predicted_value']],
                left_on=['episode_idx', 'state_t_idx'],
                right_on=['episode_idx', 'state_idx'],
                how='inner'
            )

            # Step 5: Merge again to get value_t_delta for all batches
            pairs_complete = pairs_with_t.merge(
                df[['episode_idx', 'state_idx', 'batch_name', 'predicted_value']],
                left_on=['episode_idx', 'state_t_delta_idx', 'batch_name'],
                right_on=['episode_idx', 'state_idx', 'batch_name'],
                how='inner',
                suffixes=('_t', '_t_delta')
            )

            # Step 6: Compute differences vectorized
            pairs_complete['difference'] = pairs_complete['predicted_value_t'] - pairs_complete['predicted_value_t_delta']

            # Step 7: Aggregate by state_t_idx (variance across batches for same state in pairs)
            stats_temporal = pairs_complete.groupby('state_idx_t')['difference'].agg(
                mean='mean',
                variance='var',
                std='std',
                count='count'
            ).reset_index()
            stats_temporal.rename(columns={'state_idx_t': 'state_idx'}, inplace=True)

        stats_temporal['n_episodes'] = n_episodes

        # Add metadata
        predictions_path_obj = Path(predictions_path)
        results_dir = str(predictions_path_obj.parent.parent.parent)
        stats_temporal['predictions_path'] = predictions_path
        stats_temporal['results_dir'] = results_dir

        del df
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
def load_predictions_for_trajectory(predictions_path, adjust_constant=False):
    """Load predictions and compute mean across batches for each state.

    Args:
        predictions_path: Path to predictions.parquet file
        adjust_constant: If True, apply per-batch constant adjustment before averaging

    Returns:
        DataFrame with episode_idx, state_idx, mean_value, step_in_episode
    """
    df = pd.read_parquet(predictions_path)

    # Apply constant adjustment if requested
    if adjust_constant:
        predictions_path_obj = Path(predictions_path)
        results_dir = str(predictions_path_obj.parent.parent.parent)
        mean_ground_truth = _get_ground_truth_mean(results_dir)

        if mean_ground_truth is not None:
            def adjust_batch(batch_df):
                mean_batch_predictions = batch_df['predicted_value'].mean()
                constant = mean_ground_truth - mean_batch_predictions
                batch_df = batch_df.copy()
                batch_df['predicted_value'] = batch_df['predicted_value'] + constant
                return batch_df

            df = df.groupby('batch_name', group_keys=False).apply(adjust_batch)

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
