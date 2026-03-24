"""Data loading, caching, and core data abstractions."""

import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Union


# --- Eval dataset types ---

@dataclass
class TrajectoryEval:
    """Ground truth with one value per state, ordered within episodes."""
    dataset_type: str = field(default="trajectory", init=False)
    returns: np.ndarray           # (n_states,) ground truth discounted returns
    episode_idx: np.ndarray       # (n_states,)
    timestep: np.ndarray          # (n_states,)
    episode_length: np.ndarray    # (n_states,)
    is_truncated: np.ndarray      # (n_states,) bool
    critic_value: Optional[np.ndarray] = None  # (n_states,) policy critic estimates


@dataclass
class TemporalDiffEval:
    """Ground truth for temporal difference pairs within episodes.

    Each pair compares state at position t with state at position t+delta.
    """
    dataset_type: str = field(default="temporal_diff", init=False)
    returns: np.ndarray       # (n_pairs,) GT(s_t) - GT(s_{t+delta})
    deltas: np.ndarray        # (n_pairs,) temporal gap for each pair
    episode_idx: np.ndarray   # (n_pairs,) which episode each pair belongs to


@dataclass
class PairedEval:
    """Ground truth for paired state comparisons from Monte Carlo rollouts.

    Each pair has two states with ground truth values estimated via many trajectories.
    """
    dataset_type: str = field(default="paired", init=False)
    s1_mean: np.ndarray        # (n_pairs,) GT mean return from state 1
    s2_mean: np.ndarray        # (n_pairs,) GT mean return from state 2
    s1_ci_lower: np.ndarray    # (n_pairs,) 95% CI lower bound for s1
    s1_ci_upper: np.ndarray    # (n_pairs,) 95% CI upper bound for s1
    s2_ci_lower: np.ndarray    # (n_pairs,)
    s2_ci_upper: np.ndarray    # (n_pairs,)
    diff_mean: np.ndarray      # (n_pairs,) s1_mean - s2_mean
    diff_ci_lower: np.ndarray  # (n_pairs,)
    diff_ci_upper: np.ndarray  # (n_pairs,)
    returns: np.ndarray = field(init=False)  # alias for diff_mean, for metric compatibility

    def __post_init__(self):
        self.returns = self.diff_mean


EvalDataset = Union[TrajectoryEval, TemporalDiffEval, PairedEval]


# --- Ground truth store ---

class GroundTruthStore:
    """Manual cache for ground truth data, keyed by string key. One instance per session."""

    def __init__(self):
        self._cache: Dict[str, EvalDataset] = {}

    def get(self, key: str) -> Optional[EvalDataset]:
        if key is None:
            return None
        if key not in self._cache:
            self._cache[key] = self._load(key)
        return self._cache[key]

    def put(self, key: str, dataset: EvalDataset):
        """Manually store a dataset (used for derived datasets like temporal diffs)."""
        self._cache[key] = dataset

    def _load(self, path: str) -> TrajectoryEval:
        df = pd.read_parquet(path)
        return TrajectoryEval(
            returns=df["ground_truth_return"].values,
            episode_idx=df["episode_idx"].values,
            timestep=df["timestep_in_episode"].values,
            episode_length=df["episode_length"].values,
            is_truncated=df["is_truncated"].values,
            critic_value=df["critic_value"].values if "critic_value" in df.columns else None,
        )


def get_gt_store() -> GroundTruthStore:
    """Get or create the session-level GroundTruthStore."""
    if "_gt_store" not in st.session_state:
        st.session_state._gt_store = GroundTruthStore()
    return st.session_state._gt_store


# --- Method predictions ---

@dataclass
class MethodPredictions:
    """Predictions for one (method, n_episodes) combination."""
    method: str
    n_episodes: int
    pivot: pd.DataFrame       # (n_items x n_batches) predicted values
    mean: np.ndarray          # (n_items,) mean across batches
    variance: np.ndarray      # (n_items,) variance across batches
    ground_truth_key: Optional[str] = None  # key into GroundTruthStore


# --- Loading functions ---

@st.cache_data
def _load_pivot(predictions_path: str) -> pd.DataFrame:
    """Load predictions parquet and pivot to (state_idx x batch_name)."""
    df = pd.read_parquet(predictions_path)
    return df.pivot_table(index="state_idx", columns="batch_name", values="predicted_value")


def load_method_predictions(
    predictions_path: str,
    ground_truth_key: Optional[str],
    method: str,
    n_episodes: int,
) -> MethodPredictions:
    """Load predictions and build MethodPredictions.

    The pivot table loading is cached; mean/variance are computed on the fly (fast).
    """
    pivot = _load_pivot(predictions_path)
    return MethodPredictions(
        method=method,
        n_episodes=n_episodes,
        pivot=pivot,
        mean=pivot.mean(axis=1).values,
        variance=pivot.var(axis=1, ddof=1).values,
        ground_truth_key=ground_truth_key,
    )


def apply_constant_adjustment(mp: MethodPredictions, gt_store: GroundTruthStore) -> MethodPredictions:
    """Adjust each batch by a constant so its mean matches the ground truth mean.

    Removes state-independent bias while preserving relative differences.
    """
    gt = gt_store.get(mp.ground_truth_key)
    if gt is None:
        return mp
    gt_mean = np.mean(gt.returns)
    batch_means = mp.pivot.mean(axis=0)
    adjusted_pivot = mp.pivot - batch_means + gt_mean
    return MethodPredictions(
        method=mp.method,
        n_episodes=mp.n_episodes,
        pivot=adjusted_pivot,
        mean=adjusted_pivot.mean(axis=1).values,
        variance=adjusted_pivot.var(axis=1, ddof=1).values,
        ground_truth_key=mp.ground_truth_key,
    )


def filter_states(
    mean: np.ndarray,
    variance: np.ndarray,
    filter_high_variance: int = 0,
    filter_extreme_mean: int = 0,
) -> np.ndarray:
    """Return boolean mask keeping states after filtering outliers.

    Args:
        filter_high_variance: Remove top N% by variance
        filter_extreme_mean: Remove top+bottom N% by mean
    """
    mask = np.ones(len(mean), dtype=bool)
    if filter_high_variance > 0:
        threshold = np.percentile(variance, 100 - filter_high_variance)
        mask &= variance <= threshold
    if filter_extreme_mean > 0:
        lo = np.percentile(mean, filter_extreme_mean)
        hi = np.percentile(mean, 100 - filter_extreme_mean)
        mask &= (mean >= lo) & (mean <= hi)
    return mask


@st.cache_data
def load_observations(data_dir: str) -> Optional[np.ndarray]:
    """Load observations from eval batch. Only called when metrics need state vectors."""
    eval_path = Path(data_dir) / "batch_eval.npz"
    if not eval_path.exists():
        return None
    batch = np.load(eval_path, allow_pickle=True)
    return np.concatenate(batch["observations"])


# --- Temporal difference transformation ---

def _generate_temporal_pairs(gt: TrajectoryEval, temporal_p: float, seed: int):
    """Generate within-episode temporal pairs using geometric gaps.

    For each episode, generates non-overlapping pairs (t, t+delta) where:
    - delta ~ Geometric(temporal_p): the gap between paired states
    - buffer ~ Geometric(temporal_p): spacing before the next pair
    - Pairs that fall outside the episode are discarded

    Returns:
        state_t_indices: (n_pairs,) flat indices into the state arrays for s_t
        state_td_indices: (n_pairs,) flat indices for s_{t+delta}
        deltas: (n_pairs,) temporal gap for each pair
        pair_episode_idx: (n_pairs,) episode each pair belongs to
    """
    rng = np.random.RandomState(seed)

    # Group states by episode
    unique_episodes = np.unique(gt.episode_idx)
    episode_starts = {}
    episode_lengths = {}
    for ep in unique_episodes:
        mask = gt.episode_idx == ep
        indices = np.where(mask)[0]
        episode_starts[ep] = indices[0]
        episode_lengths[ep] = len(indices)

    max_length = max(episode_lengths.values())
    n_episodes = len(unique_episodes)
    max_pairs = max_length // 2 + 1

    # Sample gaps and buffers
    deltas = rng.geometric(temporal_p, size=(n_episodes, max_pairs))
    buffers = rng.geometric(temporal_p, size=(n_episodes, max_pairs))

    # Compute positions: cumulative sum of (delta + buffer)
    steps = deltas + buffers
    positions = np.zeros((n_episodes, max_pairs), dtype=int)
    positions[:, 1:] = np.cumsum(steps[:, :-1], axis=1)

    # Find valid pairs: both t and t+delta within episode
    ep_lengths = np.array([episode_lengths[ep] for ep in unique_episodes])
    valid_mask = (positions + deltas) < ep_lengths[:, np.newaxis]

    valid_ep_idx, valid_pair_idx = np.where(valid_mask)
    if len(valid_ep_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    valid_positions = positions[valid_ep_idx, valid_pair_idx]
    valid_deltas = deltas[valid_ep_idx, valid_pair_idx]

    ep_starts = np.array([episode_starts[unique_episodes[i]] for i in valid_ep_idx])
    state_t = ep_starts + valid_positions
    state_td = ep_starts + valid_positions + valid_deltas
    pair_episodes = np.array([unique_episodes[i] for i in valid_ep_idx])

    return state_t, state_td, valid_deltas, pair_episodes


def to_temporal_diff(
    mp: MethodPredictions,
    gt_store: GroundTruthStore,
    temporal_p: float = 0.2,
    seed: int = 42,
) -> MethodPredictions:
    """Transform trajectory predictions into temporal difference predictions.

    For each within-episode pair (s_t, s_{t+delta}), computes V̂(s_t) - V̂(s_{t+delta})
    per batch. Also creates and stores a TemporalDiffEval ground truth.

    Returns a new MethodPredictions whose pivot rows are pairs (not states).
    """
    gt = gt_store.get(mp.ground_truth_key)
    if gt is None or not isinstance(gt, TrajectoryEval):
        return mp

    # Generate pairs (cached via gt_store key)
    td_gt_key = f"{mp.ground_truth_key}:temporal:{temporal_p}:{seed}"
    td_gt = gt_store.get(td_gt_key) if td_gt_key in gt_store._cache else None

    if td_gt is None:
        state_t, state_td, deltas, pair_episodes = _generate_temporal_pairs(gt, temporal_p, seed)
        if len(state_t) == 0:
            return mp

        gt_diffs = gt.returns[state_t] - gt.returns[state_td]
        td_gt = TemporalDiffEval(
            returns=gt_diffs,
            deltas=deltas,
            episode_idx=pair_episodes,
        )
        gt_store.put(td_gt_key, td_gt)
        # Stash pair indices for reuse by other methods
        gt_store.put(f"{td_gt_key}:indices", (state_t, state_td))
    else:
        state_t, state_td = gt_store.get(f"{td_gt_key}:indices")

    # Compute prediction differences per batch
    pivot_values = mp.pivot.values  # (n_states, n_batches)
    diff_pivot = pivot_values[state_t, :] - pivot_values[state_td, :]
    diff_pivot_df = pd.DataFrame(diff_pivot, columns=mp.pivot.columns)

    return MethodPredictions(
        method=mp.method,
        n_episodes=mp.n_episodes,
        pivot=diff_pivot_df,
        mean=diff_pivot_df.mean(axis=1).values,
        variance=diff_pivot_df.var(axis=1, ddof=1).values,
        ground_truth_key=td_gt_key,
    )


# --- Paired states ---

@st.cache_data
def load_paired_ground_truth(data_dir: str) -> Optional[PairedEval]:
    """Load paired_states.npz and return PairedEval ground truth."""
    path = Path(data_dir) / "paired_states.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return PairedEval(
        s1_mean=data["s1_mean"].astype(float),
        s2_mean=data["s2_mean"].astype(float),
        s1_ci_lower=data["s1_ci_lower"].astype(float),
        s1_ci_upper=data["s1_ci_upper"].astype(float),
        s2_ci_lower=data["s2_ci_lower"].astype(float),
        s2_ci_upper=data["s2_ci_upper"].astype(float),
        diff_mean=data["diff_mean"].astype(float),
        diff_ci_lower=data["diff_ci_lower"].astype(float),
        diff_ci_upper=data["diff_ci_upper"].astype(float),
    )


@st.cache_data
def _load_paired_pivot(paired_path: str, value_col: str) -> Optional[pd.DataFrame]:
    """Load paired predictions and pivot to (pair_idx x batch_name)."""
    path = Path(paired_path)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df.pivot_table(index="pair_idx", columns="batch_name", values=value_col)


def load_paired_predictions(
    paired_path: str,
    data_dir: str,
    gt_store: GroundTruthStore,
    method: str,
    n_episodes: int,
    mode: str = "diff",
) -> Optional[MethodPredictions]:
    """Load paired predictions as MethodPredictions.

    Args:
        mode: "diff" for V(s1)-V(s2) differences, "all" for s1 and s2 concatenated
    """
    if mode == "all":
        pivot_s1 = _load_paired_pivot(paired_path, "s1_predicted")
        pivot_s2 = _load_paired_pivot(paired_path, "s2_predicted")
        if pivot_s1 is None or pivot_s2 is None:
            return None
        pivot_s2 = pivot_s2.set_index(pivot_s2.index + len(pivot_s1))
        pivot = pd.concat([pivot_s1, pivot_s2], axis=0)
    else:
        pivot = _load_paired_pivot(paired_path, "diff_predicted")
        if pivot is None:
            return None

    # Ensure base PairedEval is loaded
    base_key = f"{data_dir}:paired"
    if base_key not in gt_store._cache:
        paired_gt = load_paired_ground_truth(data_dir)
        if paired_gt is None:
            return None
        gt_store.put(base_key, paired_gt)

    # Use mode-specific key so gt.returns matches the prediction mode
    gt_key = f"{data_dir}:paired:{mode}"
    if gt_key not in gt_store._cache:
        base_gt = gt_store.get(base_key)
        import copy
        mode_gt = copy.copy(base_gt)
        if mode == "all":
            mode_gt.returns = np.concatenate([base_gt.s1_mean, base_gt.s2_mean])
        else:
            mode_gt.returns = base_gt.diff_mean
        gt_store.put(gt_key, mode_gt)

    return MethodPredictions(
        method=method,
        n_episodes=n_episodes,
        pivot=pivot,
        mean=pivot.mean(axis=1).values,
        variance=pivot.var(axis=1, ddof=1).values,
        ground_truth_key=gt_key,
    )
