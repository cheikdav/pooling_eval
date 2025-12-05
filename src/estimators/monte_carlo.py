"""Monte Carlo value estimator."""

import torch
import numpy as np
from typing import Dict, Any

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
        super().__init__(obs_dim, hidden_sizes, discount_factor, activation, learning_rate, device)

    @classmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        """Get method-specific parameters from config.

        Monte Carlo estimator has no additional parameters beyond the base class.

        Args:
            method_config: MonteCarloConfig instance

        Returns:
            Empty dictionary (no method-specific params)
        """
        return {}

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
                - mc_returns: (n_transitions,) array of precomputed Monte Carlo returns

        Returns:
            Target values as torch tensor
        """
        # Use precomputed Monte Carlo returns from preprocessing
        return torch.FloatTensor(batch['mc_returns']).to(self.device)

    def train_step(self, batch: Dict[str, np.ndarray], batch_size: int = None) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing preprocessed transition data
            batch_size: Size of mini-batches. If None, use full batch.

        Returns:
            Dictionary of training metrics
        """
        # Data is already preprocessed (flattened), pass directly to parent
        return super().train_step(batch, batch_size=batch_size)

    def get_config(self) -> Dict:
        """Get estimator configuration."""
        config = super().get_config()
        config['discount_factor'] = self.discount_factor
        config['estimator_type'] = 'monte_carlo'
        return config
