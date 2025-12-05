"""DQN-style value estimator with target network."""

import torch
from typing import Dict, Any
import copy

from src.estimators.base import ValueEstimator


class DQNEstimator(ValueEstimator):
    """DQN-style value estimator with target network"""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float = 0.99,
        target_update_rate: float = 1e-5,
        activation: str = "relu",
        learning_rate: float = 0.001,
        device: str = "auto"
    ):
        """Initialize DQN estimator.

        Args:
            obs_dim: Observation dimension
            hidden_sizes: List of hidden layer sizes
            discount_factor: Discount factor (gamma)
            target_update_rate: Polyak averaging coefficient for target network updates
            activation: Activation function
            learning_rate: Learning rate
            device: Device to use
        """
        super().__init__(obs_dim, hidden_sizes, discount_factor, activation, learning_rate, device)
        self.target_update_rate = target_update_rate

        # Create target network (copy of value network)
        self.target_net = copy.deepcopy(self.value_net).to(self.device)
        self.target_net.eval()

        # Track when to update target network
        self.steps_since_target_update = 0

    @classmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        """Get method-specific parameters from config.

        Args:
            method_config: DQNConfig instance

        Returns:
            Dictionary with DQN-specific parameters
        """
        return {
            'target_update_rate': method_config.target_update_rate,
        }

    def update_target_network(self):
        """Update target network with current value network weights."""
        weight_dicts = {}
        for name in self.target_net.state_dict().keys():
            weight_dicts[name] = (1 - self.target_update_rate) * self.target_net.state_dict()[name] + self.target_update_rate * self.value_net.state_dict()[name]
        self.target_net.load_state_dict(weight_dicts)
        self.steps_since_target_update = 0

    def compute_targets(self, mini_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute DQN targets using target network.

        Args:
            mini_batch: Dictionary containing mini-batch data (torch tensors):
                - next_observations: (batch_size, obs_dim)
                - rewards: (batch_size,)
                - dones: (batch_size,)

        Returns:
            Target values as torch tensor
        """
        # Move tensors to device
        next_obs = mini_batch['next_observations'].to(self.device)
        rewards = mini_batch['rewards'].to(self.device)
        dones = mini_batch['dones'].to(self.device)

        with torch.no_grad():
            # Normalization happens inside target network
            next_values = self.target_net(next_obs).squeeze(-1)
            targets = rewards + self.discount_factor * next_values * (1 - dones)

        return targets

    def train_step(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step on a mini-batch.

        Args:
            mini_batch: Dictionary containing mini-batch data (torch tensors)

        Returns:
            Dictionary of training metrics
        """
        # Train and update target network
        metrics = super().train_step(mini_batch)

        # Update target network
        self.update_target_network()

        return metrics


    def save(self, path):
        """Save estimator to disk (including target network)."""
        checkpoint = {
            'value_net_state_dict': self.value_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'steps_since_target_update': self.steps_since_target_update,
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'target_update_rate': self.target_update_rate,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        """Load estimator from disk (including target network)."""
        checkpoint = torch.load(path, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.steps_since_target_update = checkpoint['steps_since_target_update']

    def get_config(self) -> Dict:
        """Get estimator configuration."""
        config = super().get_config()
        config.update({
            'discount_factor': self.discount_factor,
            'target_update_rate': self.target_update_rate,
            'estimator_type': 'dqn',
        })
        return config
