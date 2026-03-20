"""Data preprocessing utilities for flattening episodes and computing targets."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict


def compute_monte_carlo_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted Monte Carlo returns from rewards.

    Args:
        rewards: Array of rewards of shape (T,)
        gamma: Discount factor

    Returns:
        Discounted returns of shape (T,)
    """
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0

    # Compute returns backward
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def preprocess_episodes(batch: Dict[str, np.ndarray], gamma: float, truncation_coefficient: float = 5.0) -> Dict[str, np.ndarray]:
    """Preprocess episode-based batch into transition-based format with MC returns.

    Takes episode data (lists of variable-length arrays) and flattens into individual
    transitions, computing Monte Carlo returns for each state. Discards the last
    truncation_coefficient/(1-gamma) states from truncated episodes to handle truncation errors.

    Args:
        batch: Dictionary containing episode data:
            - observations: List of (T_i, obs_dim) arrays
            - actions: List of (T_i, act_dim) or (T_i,) arrays
            - rewards: List of (T_i,) arrays
            - dones: List of (T_i,) arrays
            - next_observations: List of (T_i, obs_dim) arrays
            - truncated: (n_episodes,) boolean array (optional, defaults to all False)
        gamma: Discount factor for computing returns
        truncation_coefficient: Coefficient for computing how many states to discard
            from the end of truncated episodes (discard last truncation_coefficient/(1-gamma) states)

    Returns:
        Dictionary containing flattened transition data:
            - observations: (total_transitions, obs_dim) array
            - actions: (total_transitions, act_dim) or (total_transitions,) array
            - rewards: (total_transitions,) array
            - dones: (total_transitions,) array
            - next_observations: (total_transitions, obs_dim) array
            - mc_returns: (total_transitions,) array - Monte Carlo returns for each state
    """
    # Check if data is already preprocessed
    if not isinstance(batch['observations'], list):
        # Already flattened, assume mc_returns is present
        return batch

    # Compute number of states to discard from the end of truncated episodes
    n_discard = int(truncation_coefficient / (1 - gamma))

    # Lists to collect flattened data
    all_observations = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_next_observations = []
    all_mc_returns = []

    # Process each episode
    for i in range(len(batch['observations'])):
        obs = batch['observations'][i]
        actions = batch['actions'][i]
        rewards = batch['rewards'][i]
        dones = batch['dones'][i]
        next_obs = batch['next_observations'][i]

        # Compute MC returns for this episode
        mc_returns = compute_monte_carlo_returns(rewards, gamma)

        # Check if this episode was truncated
        truncated = batch.get('truncated', np.zeros(len(batch['observations']), dtype=bool))[i]

        # Only discard states from truncated episodes
        episode_length = len(rewards)
        if truncated:
            keep_length = max(0, episode_length - n_discard)
            if keep_length == 0:
                # Skip episodes that are too short after discarding
                continue
        else:
            # Keep all states for naturally terminated episodes
            keep_length = episode_length

        # Add to lists (only keeping first keep_length states)
        all_observations.append(obs[:keep_length])
        all_actions.append(actions[:keep_length])
        all_rewards.append(rewards[:keep_length])
        all_dones.append(dones[:keep_length])
        all_next_observations.append(next_obs[:keep_length])
        all_mc_returns.append(mc_returns[:keep_length])

    # Concatenate all episodes
    preprocessed = {
        'observations': np.concatenate(all_observations, axis=0).astype(np.float32),
        'actions': np.concatenate(all_actions, axis=0),
        'rewards': np.concatenate(all_rewards, axis=0).astype(np.float32),
        'dones': np.concatenate(all_dones, axis=0).astype(np.float32),
        'next_observations': np.concatenate(all_next_observations, axis=0).astype(np.float32),
        'mc_returns': np.concatenate(all_mc_returns, axis=0).astype(np.float32),
    }

    return preprocessed


def split_episodes_for_preprocessing(batch: Dict[str, np.ndarray], preprocess_fraction: float, seed: int = 42):
    """Split episodes into preprocessing and training sets.

    Args:
        batch: Raw episode batch (lists of arrays):
            - observations: List of (T_i, obs_dim) arrays
            - actions: List of (T_i, act_dim) or (T_i,) arrays
            - rewards: List of (T_i,) arrays
            - dones: List of (T_i,) arrays
            - next_observations: List of (T_i, obs_dim) arrays
        preprocess_fraction: Fraction of episodes for preprocessing (0.0-1.0)
            - If 0.0, returns (None, full_batch) - no preprocessing split
            - If > 0.0, splits episodes into preprocessing and training sets
        seed: Random seed for reproducible splitting

    Returns:
        (preprocess_batch, train_batch):
            - preprocess_batch: None if preprocess_fraction=0.0, otherwise preprocessing episodes
            - train_batch: Full batch if preprocess_fraction=0.0, otherwise training episodes
    """
    if preprocess_fraction == 0.0:
        return None, batch

    total_episodes = len(batch['observations'])
    n_preprocess = int(total_episodes * preprocess_fraction)

    if n_preprocess == 0:
        raise ValueError(f"preprocess_fraction={preprocess_fraction} results in 0 preprocessing episodes. Use 0.0 to disable preprocessing.")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(total_episodes)

    preprocess_indices = indices[:n_preprocess]
    train_indices = indices[n_preprocess:]

    preprocess_batch = {
        'observations': [batch['observations'][i] for i in preprocess_indices],
        'actions': [batch['actions'][i] for i in preprocess_indices],
        'rewards': [batch['rewards'][i] for i in preprocess_indices],
        'dones': [batch['dones'][i] for i in preprocess_indices],
        'next_observations': [batch['next_observations'][i] for i in preprocess_indices],
    }

    train_batch = {
        'observations': [batch['observations'][i] for i in train_indices],
        'actions': [batch['actions'][i] for i in train_indices],
        'rewards': [batch['rewards'][i] for i in train_indices],
        'dones': [batch['dones'][i] for i in train_indices],
        'next_observations': [batch['next_observations'][i] for i in train_indices],
    }

    return preprocess_batch, train_batch


class TransitionDataset(Dataset):
    """PyTorch Dataset for transition data with optional feature caching."""

    def __init__(self, batch: Dict[str, np.ndarray]):
        """Initialize dataset from preprocessed batch.

        Args:
            batch: Dictionary containing flattened transition data:
                - observations: (n_transitions, obs_dim) array
                - actions: (n_transitions, act_dim) or (n_transitions,) array
                - rewards: (n_transitions,) array
                - dones: (n_transitions,) array
                - next_observations: (n_transitions, obs_dim) array
                - mc_returns: (n_transitions,) array
        """
        self.observations = torch.FloatTensor(batch['observations'])
        self.next_observations = torch.FloatTensor(batch['next_observations'])
        self.rewards = torch.FloatTensor(batch['rewards'])
        self.dones = torch.FloatTensor(batch['dones'])
        self.mc_returns = torch.FloatTensor(batch['mc_returns'])

        # Optional cached features (set via set_features())
        self.features = None
        self.next_features = None

    def set_features(self, features: torch.Tensor, next_features: torch.Tensor):
        """Cache extracted features to avoid recomputation during training.

        Args:
            features: Extracted features from observations (n_transitions, feature_dim)
            next_features: Extracted features from next_observations (n_transitions, feature_dim)
        """
        self.features = features
        self.next_features = next_features

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        item = {
            'observations': self.observations[idx],
            'next_observations': self.next_observations[idx],
            'rewards': self.rewards[idx],
            'dones': self.dones[idx],
            'mc_returns': self.mc_returns[idx],
        }

        # Add cached features if available
        if self.features is not None:
            item['features'] = self.features[idx]
        if self.next_features is not None:
            item['next_features'] = self.next_features[idx]

        return item
