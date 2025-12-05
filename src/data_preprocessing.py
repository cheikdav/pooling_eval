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


def preprocess_episodes(batch: Dict[str, np.ndarray], gamma: float) -> Dict[str, np.ndarray]:
    """Preprocess episode-based batch into transition-based format with MC returns.

    Takes episode data (lists of variable-length arrays) and flattens into individual
    transitions, computing Monte Carlo returns for each state.

    Args:
        batch: Dictionary containing episode data:
            - observations: List of (T_i, obs_dim) arrays
            - actions: List of (T_i, act_dim) or (T_i,) arrays
            - rewards: List of (T_i,) arrays
            - dones: List of (T_i,) arrays
            - next_observations: List of (T_i, obs_dim) arrays
        gamma: Discount factor for computing returns

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

        # Add to lists
        all_observations.append(obs)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_dones.append(dones)
        all_next_observations.append(next_obs)
        all_mc_returns.append(mc_returns)

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


def sample_episodes(batch: Dict[str, np.ndarray], n_episodes: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """Sample a subset of episodes from a batch.

    Args:
        batch: Dictionary containing episode data (lists of variable-length arrays):
            - observations: List of (T_i, obs_dim) arrays
            - actions: List of (T_i, act_dim) or (T_i,) arrays
            - rewards: List of (T_i,) arrays
            - dones: List of (T_i,) arrays
            - next_observations: List of (T_i, obs_dim) arrays
        n_episodes: Number of episodes to sample
        seed: Random seed for reproducible sampling

    Returns:
        Dictionary with sampled episodes (same structure as input)
    """
    total_episodes = len(batch['observations'])
    if n_episodes > total_episodes:
        raise ValueError(f"Cannot sample {n_episodes} episodes from batch with {total_episodes} episodes")

    # Sample random episode indices
    rng = np.random.RandomState(seed)
    indices = rng.choice(total_episodes, size=n_episodes, replace=False)

    # Extract sampled episodes
    sampled_batch = {
        'observations': [batch['observations'][i] for i in indices],
        'actions': [batch['actions'][i] for i in indices],
        'rewards': [batch['rewards'][i] for i in indices],
        'dones': [batch['dones'][i] for i in indices],
        'next_observations': [batch['next_observations'][i] for i in indices],
    }

    return sampled_batch


class TransitionDataset(Dataset):
    """PyTorch Dataset for transition data."""

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

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            'observations': self.observations[idx],
            'next_observations': self.next_observations[idx],
            'rewards': self.rewards[idx],
            'dones': self.dones[idx],
            'mc_returns': self.mc_returns[idx],
        }
