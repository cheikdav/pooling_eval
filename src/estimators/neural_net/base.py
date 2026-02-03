"""Base class for neural network-based value estimators."""

from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
import numpy as np

from ..base import ValueEstimator, ValueNetwork, RunningNormalizer

class NeuralNetEstimator(ValueEstimator):
    """Base class for neural network-based value estimators."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float,
        activation: str = "relu",
        learning_rate: float = 0.001,
        device: str = "auto",
        normalize_observations: bool = True
    ):
        """Initialize neural network estimator.

        Args:
            obs_dim: Observation dimension
            hidden_sizes: List of hidden layer sizes
            discount_factor: Discount factor (gamma)
            activation: Activation function ('relu' or 'tanh')
            learning_rate: Learning rate for optimizer
            device: Device to use ('auto', 'cpu', or 'cuda')
            normalize_observations: Whether to normalize observations using running statistics
        """
        super().__init__(obs_dim, discount_factor, device)

        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.normalize_observations = normalize_observations

        self.value_net = ValueNetwork(obs_dim, hidden_sizes, activation, normalize_observations).to(self.device)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)

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
        # Common parameters
        common_params = {
            'obs_dim': obs_dim,
            'hidden_sizes': network_config.hidden_sizes,
            'discount_factor': gamma,
            'activation': network_config.activation,
            'learning_rate': method_config.learning_rate,
            'device': network_config.device,
        }

        # Get method-specific parameters
        specific_params = cls._get_method_specific_params(method_config)

        # Instantiate with all parameters
        return cls(**common_params, **specific_params)

    def train(self):
        """Set estimator to training mode."""
        self.value_net.train()

    def eval(self):
        """Set estimator to evaluation mode."""
        self.value_net.eval()

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
        self.train()

        # Move tensors to device
        obs = mini_batch['observations'].to(self.device)
        mc_returns = mini_batch['mc_returns'].to(self.device)

        # Forward pass
        targets = self.compute_targets(mini_batch)
        values = self.value_net(obs).squeeze(-1)

        # Compute loss and backprop
        loss = nn.functional.mse_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            mae = torch.abs(values - targets).mean().item()
            mc_loss = nn.functional.mse_loss(values, mc_returns).item()

        self.training_step += 1

        return {
            'loss': loss.item(),
            'mae': mae,
            'mean_value': values.mean().item(),
            'mean_target': targets.mean().item(),
            'mc_loss': mc_loss,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict values for given observations.

        Args:
            observations: Array of shape (n, obs_dim)

        Returns:
            Predicted values of shape (n,)
        """
        self.eval()

        with torch.no_grad():
            obs = torch.FloatTensor(observations).to(self.device)
            values = self.value_net(obs).squeeze(-1)
            return values.cpu().numpy()

    def save(self, path: Path):
        """Save estimator to disk.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'normalize_observations': self.normalize_observations,
            'discount_factor': self.discount_factor,
        }

        # Save observation normalizer state if it exists
        if self.value_net.obs_normalizer is not None:
            checkpoint['obs_normalizer_state'] = self.value_net.obs_normalizer.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: Path):
        """Load estimator from disk.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

        # Restore observation normalizer state if it exists
        if 'obs_normalizer_state' in checkpoint and self.value_net.obs_normalizer is not None:
            self.value_net.obs_normalizer.load_state_dict(checkpoint['obs_normalizer_state'])

    @classmethod
    def load_from_checkpoint(cls, path: Path, device: str = "auto"):
        """Load estimator from checkpoint file.

        Args:
            path: Path to checkpoint file
            device: Device to load model on ('auto', 'cpu', or 'cuda')

        Returns:
            Estimator instance with loaded weights
        """
        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        checkpoint = torch.load(path, map_location=device_obj)

        # Create estimator instance with saved parameters
        estimator = cls(
            obs_dim=checkpoint['obs_dim'],
            hidden_sizes=checkpoint['hidden_sizes'],
            discount_factor=checkpoint.get('discount_factor', 0.99),  # Default if not saved
            activation=checkpoint['activation'],
            learning_rate=checkpoint['learning_rate'],
            device=device,
            normalize_observations=checkpoint.get('normalize_observations', True)
        )

        # Load state
        estimator.load(path)

        return estimator

    def get_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        return {
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'training_step': self.training_step,
        }


