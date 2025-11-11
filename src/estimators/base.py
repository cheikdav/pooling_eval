"""Base class for value estimators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from functools import wraps


class ValueNetwork(nn.Module):
    """Simple MLP for value function approximation."""

    def __init__(self, input_dim: int, hidden_sizes: list, activation: str = "relu"):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes

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
        return self.network(x)

def format_batch(func):
    @wraps(func)
    def wrapper(self, batch: Dict[str, np.ndarray], *args, **kwargs):
        # Convert numpy arrays to torch tensors
        batch = self._format_batch(batch)
        return func(self, batch, *args, **kwargs)
    return wrapper

class ValueEstimator(ABC):
    """Base class for value function estimators."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        activation: str = "relu",
        learning_rate: float = 0.001,
        device: str = "auto"
    ):
        """Initialize value estimator.

        Args:
            obs_dim: Observation dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('relu' or 'tanh')
            learning_rate: Learning rate for optimizer
            device: Device to use ('auto', 'cpu', or 'cuda')
        """
        self.obs_dim = obs_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create value network
        self.value_net = ValueNetwork(obs_dim, hidden_sizes, activation).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)

        # Training statistics
        self.training_step = 0

    def _format_batch(self, batch: Dict[str, np.ndarray]):
        return batch

    @abstractmethod
    def compute_targets(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute target values for the batch.

        Args:
            batch: Dictionary containing episode data

        Returns:
            Target values as torch tensor
        """
        pass

    @format_batch
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing episode data

        Returns:
            Dictionary of training metrics
        """
        self.value_net.train()

        # Compute targets (method-specific)
        targets = self.compute_targets(batch)

        # Get observations
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        # Forward pass
        values = self.value_net(obs).squeeze(-1)

        # Compute loss
        loss = nn.functional.mse_loss(values, targets)
        with torch.no_grad():
            mae = torch.abs(values - targets).mean().item()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_step += 1

        return {
            'loss': loss.item(),
            'mae': mae,
            'mean_value': values.mean().item(),
            'mean_target': targets.mean().item(),
        }


    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict values for given observations.

        Args:
            observations: Array of shape (n, obs_dim)

        Returns:
            Predicted values of shape (n,)
        """
        self.value_net.eval()

        with torch.no_grad():
            obs = torch.FloatTensor(observations).to(self.device)
            values = self.value_net(obs).squeeze(-1)
            return values.cpu().numpy()

    def save(self, path: Path):
        """Save estimator to disk.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
        }, path)

    def load(self, path: Path):
        """Load estimator from disk.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

    def get_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        return {
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'training_step': self.training_step,
        }
