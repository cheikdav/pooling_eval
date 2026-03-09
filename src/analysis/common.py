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
def get_differences_split(results_dir, seed, s1_proportion):
    """Generate episode partition and state pairings for differences mode.

    This function is cached so the same split is used across all methods and n_episodes
    that share the same ground truth dataset.

    Args:
        results_dir: Path to results directory (identifies the ground truth)
        seed: Random seed for reproducible splits
        s1_proportion: Proportion of episodes in S1 partition

    Returns:
        Dictionary with:
            - 's1_episodes': set of episode indices in S1
            - 's2_episodes': set of episode indices in S2
            - 'pairings': dict mapping s1_state_idx -> s2_state_idx
    """
    ground_truth_file = Path(results_dir) / "ground_truth" / "ground_truth_returns.parquet"

    if not ground_truth_file.exists():
        return {'s1_episodes': set(), 's2_episodes': set(), 'pairings': {}}

    df = pd.read_parquet(ground_truth_file)
    episodes = df['episode_idx'].unique()

    # Partition episodes into S1 and S2
    np.random.seed(seed)
    shuffled = np.random.permutation(episodes)
    n_s1 = int(len(episodes) * s1_proportion)
    s1_eps = set(shuffled[:n_s1])
    s2_eps = set(shuffled[n_s1:])

    # Get available states in each partition
    df_s1 = df[df['episode_idx'].isin(s1_eps)]
    df_s2 = df[df['episode_idx'].isin(s2_eps)]

    s1_states = df_s1['state_idx'].unique()
    s2_states = df_s2['state_idx'].unique()

    # Create pairings: each S1 state paired with a random S2 state
    np.random.seed(seed + 1)
    pairings = {int(s): int(np.random.choice(s2_states)) for s in s1_states}

    return {
        's1_episodes': s1_eps,
        's2_episodes': s2_eps,
        'pairings': pairings
    }


@st.cache_data
def get_temporal_split(results_dir, seed, temporal_p):
    """Generate temporal pairs for temporal differences mode.

    This function is cached so the same split is used across all methods and n_episodes
    that share the same ground truth dataset.

    Args:
        results_dir: Path to results directory (identifies the ground truth)
        seed: Random seed for reproducible splits
        temporal_p: Geometric distribution parameter for temporal gaps

    Returns:
        Dictionary with:
            - 'deltas': array of temporal gaps
            - 'buffers': array of buffer sizes
            - 'positions': array of starting positions
            - 'episode_info': DataFrame with episode metadata
    """
    ground_truth_file = Path(results_dir) / "ground_truth" / "ground_truth_returns.parquet"

    if not ground_truth_file.exists():
        return {
            'deltas': np.array([]),
            'buffers': np.array([]),
            'positions': np.array([]),
            'episode_info': pd.DataFrame()
        }

    df = pd.read_parquet(ground_truth_file)

    # Get episode info
    episode_info = df.groupby('episode_idx').agg({
        'state_idx': ['min', 'count']
    }).reset_index()
    episode_info.columns = ['episode_idx', 'min_state_idx', 'episode_length']

    # Sample temporal pairs
    np.random.seed(seed)
    n_episodes = len(episode_info)
    max_episode_length = int(episode_info['episode_length'].max())
    max_pairs = max_episode_length // 2 + 1

    deltas = np.random.geometric(temporal_p, size=(n_episodes, max_pairs))
    buffers = np.random.geometric(temporal_p, size=(n_episodes, max_pairs))

    steps = deltas + buffers
    positions = np.concatenate([
        np.zeros((n_episodes, 1), dtype=int),
        np.cumsum(steps[:, :-1], axis=1)
    ], axis=1)

    return {
        'deltas': deltas,
        'buffers': buffers,
        'positions': positions,
        'episode_info': episode_info
    }


def _compute_differences(df, split, value_column):
    """Compute paired state differences using a cached split.

    Args:
        df: DataFrame with columns [episode_idx, state_idx, value_column, ...]
        split: Split dict from get_differences_split() with s1_episodes, s2_episodes, pairings
        value_column: Name of the column containing values to difference

    Returns:
        DataFrame with columns [state_idx, difference_value]
        For predictions with batches: also includes batch_name column
    """
    # Split into S1 and S2 using cached partition
    df_s1 = df[df['episode_idx'].isin(split['s1_episodes'])].copy()
    df_s2 = df[df['episode_idx'].isin(split['s2_episodes'])].copy()

    # Add pairing information to S1
    df_s1['paired_state_idx'] = df_s1['state_idx'].map(split['pairings'])

    # Determine merge keys (handle both ground truth and predictions)
    merge_keys = ['paired_state_idx']
    if 'batch_name' in df_s1.columns:
        merge_keys.append('batch_name')

    # Merge S1 with S2 on paired states (and batch if present)
    df_merged = df_s1.merge(
        df_s2[['state_idx'] + merge_keys[1:] + [value_column]],
        left_on=merge_keys,
        right_on=['state_idx'] + merge_keys[1:],
        suffixes=('_s1', '_s2')
    )

    # Compute differences: V(s) - V(s')
    df_merged['difference_value'] = df_merged[f'{value_column}_s1'] - df_merged[f'{value_column}_s2']

    # Keep relevant columns
    result_cols = ['state_idx_s1', 'difference_value']
    if 'batch_name' in df_merged.columns:
        result_cols.append('batch_name')

    result = df_merged[result_cols].copy()
    result.rename(columns={'state_idx_s1': 'state_idx'}, inplace=True)

    return result


def _compute_temporal_differences(df, split, value_column):
    """Compute within-episode temporal differences using a cached split.

    Args:
        df: DataFrame with columns [episode_idx, state_idx, value_column, ...]
        split: Split dict from get_temporal_split() with deltas, positions, episode_info
        value_column: Name of the column containing values to difference

    Returns:
        DataFrame with columns [state_idx, difference_value]
        For predictions with batches: also includes batch_name column
    """
    deltas = split['deltas']
    positions = split['positions']
    episode_info = split['episode_info']

    if episode_info.empty:
        return pd.DataFrame(columns=['state_idx', 'difference_value'])

    # Find valid pairs (both t and t+delta within episode)
    episode_lengths = episode_info['episode_length'].values[:, np.newaxis]
    valid_mask = (positions + deltas) < episode_lengths

    valid_episode_idx, valid_pair_idx = np.where(valid_mask)

    if len(valid_episode_idx) == 0:
        return pd.DataFrame(columns=['state_idx', 'difference_value'])

    # Get valid positions and deltas
    valid_positions = positions[valid_episode_idx, valid_pair_idx]
    valid_deltas = deltas[valid_episode_idx, valid_pair_idx]

    # Map to actual episode indices and state indices
    actual_episode_idx = episode_info.iloc[valid_episode_idx]['episode_idx'].values
    min_state_idx = episode_info.iloc[valid_episode_idx]['min_state_idx'].values

    state_t_idx = min_state_idx + valid_positions
    state_t_delta_idx = min_state_idx + valid_positions + valid_deltas

    # Create pairs DataFrame
    pairs_df = pd.DataFrame({
        'episode_idx': actual_episode_idx,
        'state_t_idx': state_t_idx,
        'state_t_delta_idx': state_t_delta_idx
    })

    # Determine merge keys
    merge_cols = ['episode_idx', 'state_idx']
    if 'batch_name' in df.columns:
        merge_cols.append('batch_name')

    # Merge to get value_t
    pairs_with_t = pairs_df.merge(
        df[merge_cols + [value_column]],
        left_on=['episode_idx', 'state_t_idx'],
        right_on=['episode_idx', 'state_idx'],
        how='inner'
    )

    # Merge to get value_t_delta
    merge_on_left = ['episode_idx', 'state_t_delta_idx']
    merge_on_right = ['episode_idx', 'state_idx']
    if 'batch_name' in df.columns:
        merge_on_left.append('batch_name')
        merge_on_right.append('batch_name')

    pairs_complete = pairs_with_t.merge(
        df[merge_cols + [value_column]],
        left_on=merge_on_left,
        right_on=merge_on_right,
        how='inner',
        suffixes=('_t', '_t_delta')
    )

    # Compute differences: V(s_t) - V(s_{t+delta})
    pairs_complete['difference_value'] = pairs_complete[f'{value_column}_t'] - pairs_complete[f'{value_column}_t_delta']

    # Keep relevant columns
    result_cols = ['state_idx_t', 'difference_value']
    if 'batch_name' in pairs_complete.columns:
        result_cols.append('batch_name')

    result = pairs_complete[result_cols].copy()
    result.rename(columns={'state_idx_t': 'state_idx'}, inplace=True)

    return result


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
        # Use cached split and helper to compute differences
        split = get_differences_split(results_dir, seed, s1_proportion)
        result = _compute_differences(df, split, 'ground_truth_return')
        result.rename(columns={'difference_value': 'ground_truth_value'}, inplace=True)
        return result

    elif dataset_type == 'temporal':
        # Use cached split and helper to compute temporal differences
        split = get_temporal_split(results_dir, seed, temporal_p)
        result = _compute_temporal_differences(df, split, 'ground_truth_return')
        result.rename(columns={'difference_value': 'ground_truth_value'}, inplace=True)
        return result

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
    # Get results_dir once (needed for constant adjustment and splits)
    predictions_path_obj = Path(predictions_path)
    results_dir = str(predictions_path_obj.parent.parent.parent)

    # Load raw predictions
    df = pd.read_parquet(predictions_path)

    # PHASE 1: Transform dataset
    # Apply constant adjustment if requested
    if adjust_constant:
        mean_ground_truth = _get_ground_truth_mean(results_dir)
        if mean_ground_truth is not None:
            def adjust_batch(batch_df):
                mean_batch_predictions = batch_df['predicted_value'].mean()
                constant = mean_ground_truth - mean_batch_predictions
                batch_df = batch_df.copy()
                batch_df['predicted_value'] = batch_df['predicted_value'] + constant
                return batch_df
            df = df.groupby('batch_name', group_keys=False).apply(adjust_batch)

    # Apply dataset-specific transformations
    if dataset_type == 'full':
        # No transformation needed
        transformed_df = df
        value_column = 'predicted_value'
    elif dataset_type == 'differences':
        # Transform to paired state differences
        split = get_differences_split(results_dir, seed, s1_proportion)
        transformed_df = _compute_differences(df, split, 'predicted_value')
        value_column = 'difference_value'
    elif dataset_type == 'temporal':
        # Transform to temporal differences
        split = get_temporal_split(results_dir, seed, temporal_p)
        transformed_df = _compute_temporal_differences(df, split, 'predicted_value')
        value_column = 'difference_value'
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # PHASE 2: Aggregate statistics and add metadata
    if transformed_df.empty:
        stats = pd.DataFrame(columns=['state_idx', 'n_episodes', 'mean', 'variance', 'std', 'count'])
    else:
        stats = transformed_df.groupby('state_idx')[value_column].agg(
            mean='mean',
            variance='var',
            std='std',
            count='count'
        ).reset_index()

    # Add metadata
    stats['n_episodes'] = n_episodes
    stats['predictions_path'] = predictions_path
    stats['results_dir'] = results_dir

    # Clean up
    del df, transformed_df
    return stats


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
