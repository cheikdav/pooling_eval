"""Base class for value estimators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class RunningNormalizer:
    """Tracks running mean and std for online normalization."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, values: torch.Tensor):
        """Update statistics with new batch of values."""
        batch_mean = values.mean().item()
        batch_var = values.var().item()
        batch_count = values.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Welford's online algorithm for running statistics
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize values using running statistics."""
        std = np.sqrt(self.var + self.epsilon)
        return (values - self.mean) / std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize values back to original scale."""
        std = np.sqrt(self.var + self.epsilon)
        return values * std + self.mean

    def state_dict(self) -> Dict[str, Any]:
        """Get state for saving."""
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count,
            'epsilon': self.epsilon,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']
        self.epsilon = state['epsilon']


class ValueNetwork(nn.Module):
    """Simple MLP for value function approximation."""

    def __init__(self, input_dim: int, hidden_sizes: list, activation: str = "relu", normalize_observations: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.normalize_observations = normalize_observations

        if self.normalize_observations:
            self.obs_normalizer = RunningNormalizer()
        else:
            self.obs_normalizer = None

        # Build network layers
        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_size = hidden_size

        # Output layer (single value)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Value estimates of shape (batch_size, 1)
        """
        if self.normalize_observations:
            if self.training:
                self.obs_normalizer.update(x)
            x = self.obs_normalizer.normalize(x)

        return self.network(x)


class ValueEstimator(ABC):
    """Base class for all value function estimators."""

    def __init__(
        self,
        obs_dim: int,
        discount_factor: float,
        device: str = "auto"
    ):
        """Initialize value estimator.

        Args:
            obs_dim: Observation dimension
            discount_factor: Discount factor (gamma) - common to all estimators
            device: Device to use ('auto', 'cpu', or 'cuda')
        """
        self.obs_dim = obs_dim
        self.discount_factor = discount_factor

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.training_step = 0

    @classmethod
    @abstractmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        """Get method-specific parameters from config.

        Args:
            method_config: Method-specific configuration

        Returns:
            Dictionary of method-specific parameters to pass to __init__
        """
        pass

    @classmethod
    def from_config(cls, method_config, network_config, obs_dim: int, gamma: float):
        """Create estimator from configuration.

        Args:
            method_config: Method-specific configuration (BaseEstimatorConfig subclass)
            network_config: Network configuration
            obs_dim: Observation dimension
            gamma: Discount factor (from training config, shared by all methods)

        Returns:
            Estimator instance
        """
        raise NotImplementedError("Subclasses must implement from_config")

    @abstractmethod
    def compute_targets(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute target values for the batch.

        Args:
            batch: Dictionary containing preprocessed transition data with fields:
                - observations: (n_transitions, obs_dim) array
                - next_observations: (n_transitions, obs_dim) array
                - rewards: (n_transitions,) array
                - dones: (n_transitions,) array
                - mc_returns: (n_transitions,) array - precomputed Monte Carlo returns

        Returns:
            Target values as torch tensor of shape (n_transitions,)
        """
        pass

    @abstractmethod
    def train(self):
        """Set estimator to training mode."""
        pass

    @abstractmethod
    def eval(self):
        """Set estimator to evaluation mode."""
        pass

    @abstractmethod
    def train_step(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step on a mini-batch.

        Args:
            mini_batch: Dictionary containing mini-batch data (torch tensors):
                - observations: (batch_size, obs_dim)
                - next_observations: (batch_size, obs_dim)
                - rewards: (batch_size,)
                - dones: (batch_size,)
                - mc_returns: (batch_size,)

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict values for given observations.

        Args:
            observations: Array of shape (n, obs_dim)

        Returns:
            Predicted values of shape (n,)
        """
        pass

    @abstractmethod
    def save(self, path: Path):
        """Save estimator to disk.

        Args:
            path: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load estimator from disk.

        Args:
            path: Path to checkpoint
        """
        pass

    @classmethod
    @abstractmethod
    def load_from_checkpoint(cls, path: Path, device: str = "auto"):
        """Load estimator from checkpoint file.

        Args:
            path: Path to checkpoint file
            device: Device to load model on ('auto', 'cpu', or 'cuda')

        Returns:
            Estimator instance with loaded weights
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        pass


