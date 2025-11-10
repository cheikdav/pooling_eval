"""Monte Carlo value estimator."""

import torch
import numpy as np
from typing import Dict

from src.estimators.base import ValueEstimator


class MonteCarloEstimator(ValueEstimator):
    """Monte Carlo value estimator using full episode returns."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float = 0.99,
        activation: str = "relu",
        learning_rate: float = 0.001,
        device: str = "auto"
    ):
        """Initialize Monte Carlo estimator.

        Args:
            obs_dim: Observation dimension
            hidden_sizes: List of hidden layer sizes
            discount_factor: Discount factor (gamma)
            activation: Activation function
            learning_rate: Learning rate
            device: Device to use
        """
        super().__init__(obs_dim, hidden_sizes, activation, learning_rate, device)
        self.discount_factor = discount_factor

    def compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted returns from rewards.

        Args:
            rewards: Array of rewards of shape (T,)

        Returns:
            Discounted returns of shape (T,)
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = np.zeros_like(rewards[0], dtype=np.float32)

        # Compute returns backward
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.discount_factor * running_return
            returns[t] = running_return

        return returns

    def compute_targets(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute Monte Carlo targets (discounted returns).

        Args:
            batch: Dictionary containing:
                - observations: List of (T_i, obs_dim) arrays
                - rewards: List of (T_i,) arrays
                - ... (other fields not used here)

        Returns:
            Target values as torch tensor
        """
        all_returns = []

        # Handle both list format (from full episodes) and array format (from batched data)
        if isinstance(batch['rewards'], list):
            # List of episodes
            for rewards in batch['rewards']:
                returns = self.compute_returns(rewards)
                all_returns.append(returns)
            all_returns = np.concatenate(all_returns)
        else:
            # Single flattened array (assume it's one episode or pre-flattened)
            all_returns = self.compute_returns(batch['rewards'])

        return torch.FloatTensor(all_returns).to(self.device)

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing episode data

        Returns:
            Dictionary of training metrics
        """
        # Flatten observations if needed
        if isinstance(batch['observations'], list):
            obs_array = np.concatenate(batch['observations'])
        else:
            obs_array = batch['observations']

        # Create flattened batch for parent class
        flat_batch = {
            'observations': obs_array,
            'rewards': batch['rewards'],  # Keep original format for compute_targets
        }

        metrics = super().train_step(flat_batch)
        return metrics

 

    def get_config(self) -> Dict:
        """Get estimator configuration."""
        config = super().get_config()
        config['discount_factor'] = self.discount_factor
        config['estimator_type'] = 'monte_carlo'
        return config
