"""DQN-style value estimator with target network."""

import torch
import numpy as np
from typing import Dict, Any
import copy

from src.estimators.base import ValueEstimator


class DQNEstimator(ValueEstimator):
    """DQN-style value estimator with target network and optional Double DQN."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float = 0.99,
        target_update_rate: float = 1e-5,
        double_dqn: bool = True,
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
            double_dqn: Whether to use Double DQN
            activation: Activation function
            learning_rate: Learning rate
            device: Device to use
        """
        super().__init__(obs_dim, hidden_sizes, discount_factor, activation, learning_rate, device)
        self.target_update_rate = target_update_rate
        self.double_dqn = double_dqn

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
            'double_dqn': method_config.double_dqn,
        }

    def _format_batch(self, batch: Dict[str, np.ndarray]):
        if isinstance(batch['observations'], list):
            batch['next_observations'] = np.concatenate(batch['next_observations'])
            batch['observations'] = np.concatenate(batch['observations'])
            batch['rewards'] = np.concatenate(batch['rewards'])
            batch['dones'] = np.concatenate(batch['dones'])
        batch['next_observations'] = batch['next_observations'].reshape(-1, self.obs_dim)
        batch['observations'] = batch['observations'].reshape(-1, self.obs_dim)
        batch['rewards'] = batch['rewards'].flatten()
        batch['dones'] = batch['dones'].flatten()
        return batch

    def update_target_network(self):
        """Update target network with current value network weights."""
        weight_dicts = {}
        for name in self.target_net.state_dict().keys():
            weight_dicts[name] = (1 - self.target_update_rate) * self.target_net.state_dict()[name] + self.target_update_rate * self.value_net.state_dict()[name]
        self.target_net.load_state_dict(weight_dicts)
        self.steps_since_target_update = 0

    def compute_targets(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute DQN targets using target network.

        Args:
            batch: Dictionary containing:
                - observations: Array or list of arrays
                - next_observations: Array or list of arrays
                - rewards: Array or list of arrays
                - dones: Array or list of arrays

        Returns:
            Target values as torch tensor
        """


        # Convert to tensors
        next_obs_tensor = torch.FloatTensor(batch['next_observations']).to(self.device)
        rewards_tensor = torch.FloatTensor(batch['rewards']).to(self.device)
        dones_tensor = torch.FloatTensor(batch['dones']).to(self.device)

        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Use online network to select action, target network to evaluate
                # In this value estimation setting, we just use next state values
                # For Double DQN, we use online network predictions with target network evaluation
                next_values_online = self.value_net(next_obs_tensor).squeeze(-1)
                next_values_target = self.target_net(next_obs_tensor).squeeze(-1)
                # Use average of both for more stable estimates
                next_values = (next_values_online + next_values_target) / 2.0
            else:
                # Standard DQN: Use target network
                next_values = self.target_net(next_obs_tensor).squeeze(-1)

            # Compute TD targets: r + γ * V(s') * (1 - done)
            targets = rewards_tensor + self.discount_factor * next_values * (1 - dones_tensor)

        return targets

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing episode data

        Returns:
            Dictionary of training metrics
        """
        # Flatten observations if needed
        metrics = super().train_step(batch)

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
            'double_dqn': self.double_dqn,
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
            'double_dqn': self.double_dqn,
            'estimator_type': 'dqn',
        })
        return config
