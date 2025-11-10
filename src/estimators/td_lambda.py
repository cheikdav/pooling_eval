"""TD(λ) value estimator."""

import torch
import numpy as np
from typing import Dict

from src.estimators.base import ValueEstimator


class TDLambdaEstimator(ValueEstimator):
    """TD(λ) value estimator with eligibility traces."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float = 0.99,
        lambda_: float = 0.95,
        n_step: int = 1,
        activation: str = "relu",
        learning_rate: float = 0.001,
        device: str = "auto"
    ):
        """Initialize TD(λ) estimator.

        Args:
            obs_dim: Observation dimension
            hidden_sizes: List of hidden layer sizes
            discount_factor: Discount factor (gamma)
            lambda_: Eligibility trace decay parameter
            n_step: Number of steps for n-step returns (1 for TD(0))
            activation: Activation function
            learning_rate: Learning rate
            device: Device to use
        """
        super().__init__(obs_dim, hidden_sizes, activation, learning_rate, device)
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.n_step = n_step

    def compute_td_lambda_targets(self, rewards: np.ndarray, values: np.ndarray,
                                   next_values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Compute TD(λ) targets using eligibility traces.

        Args:
            rewards: Array of rewards of shape (T,)
            values: Current value estimates of shape (T,)
            next_values: Next state values of shape (T,)
            dones: Done flags of shape (T,)

        Returns:
            TD(λ) targets of shape (T,)
        """
        T = len(rewards)
        targets = np.zeros(T, dtype=np.float32)

        # Compute TD errors
        td_errors = rewards + self.discount_factor * next_values * (1 - np.array(dones)) - values

        # Compute λ-returns backward
        running_lambda_return = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                running_lambda_return = 0.0

            # λ-return: δ_t + γλ * (running_return from t+1)
            running_lambda_return = td_errors[t] + \
                self.discount_factor * self.lambda_ * running_lambda_return * (1 - dones[t])

            targets[t] = values[t] + running_lambda_return

        return targets

    def compute_n_step_targets(self, rewards: np.ndarray, next_values: np.ndarray,
                               dones: np.ndarray) -> np.ndarray:
        """Compute n-step TD targets.

        Args:
            rewards: Array of rewards of shape (T,)
            next_values: Next state values of shape (T,)
            dones: Done flags of shape (T,)

        Returns:
            n-step targets of shape (T,)
        """
        T = len(rewards)
        targets = np.zeros(T, dtype=np.float32)

        for t in range(T):
            # Compute n-step return
            n_step_return = 0.0
            discount = 1.0

            for k in range(self.n_step):
                if t + k >= T:
                    break

                n_step_return += discount * rewards[t + k]
                discount *= self.discount_factor

                if dones[t + k]:
                    break
            else:
                # Add bootstrap value if we didn't terminate
                if t + self.n_step < T:
                    n_step_return += discount * next_values[t + self.n_step - 1]

            targets[t] = n_step_return

        return targets

    def compute_targets(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute TD(λ) targets.

        Args:
            batch: Dictionary containing:
                - observations: Array or list of arrays
                - next_observations: Array or list of arrays
                - rewards: Array or list of arrays
                - dones: Array or list of arrays

        Returns:
            Target values as torch tensor
        """
        # Flatten data if needed
        if isinstance(batch['observations'], list):
            observations = np.concatenate(batch['observations'])
            next_observations = np.concatenate(batch['next_observations'])
            rewards = np.concatenate(batch['rewards'])
            dones = np.concatenate(batch['dones'])
        else:
            observations = batch['observations']
            next_observations = batch['next_observations']
            rewards = batch['rewards']
            dones = batch['dones']

        # Get current value estimates (no gradients for target computation)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            next_obs_tensor = torch.FloatTensor(next_observations).to(self.device)

            values = self.value_net(obs_tensor).squeeze(-1).cpu().numpy()
            next_values = self.value_net(next_obs_tensor).squeeze(-1).cpu().numpy()

        # Compute targets based on method
        if self.lambda_ < 1.0 and self.lambda_ > 0.0:
            # Use TD(λ) with eligibility traces
            targets = self.compute_td_lambda_targets(rewards, values, next_values, dones)
        else:
            # Use n-step TD
            targets = self.compute_n_step_targets(rewards, next_values, dones)

        return torch.FloatTensor(targets).to(self.device)

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

        # Keep full batch for compute_targets
        flat_batch = {
            'observations': obs_array,
            'next_observations': batch['next_observations'],
            'rewards': batch['rewards'],
            'dones': batch['dones'],
        }

        metrics = super().train_step(flat_batch)
        return metrics


    def get_config(self) -> Dict:
        """Get estimator configuration."""
        config = super().get_config()
        config.update({
            'discount_factor': self.discount_factor,
            'lambda': self.lambda_,
            'n_step': self.n_step,
            'estimator_type': 'td_lambda',
        })
        return config
