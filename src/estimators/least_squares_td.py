"""Least Squares Temporal Difference (LSTD) estimator using policy network representations."""

import torch
from typing import Dict, Any

from src.estimators.base import LeastSquaresEstimator


class LeastSquaresTDEstimator(LeastSquaresEstimator):
    """Least Squares Temporal Difference estimator using frozen policy representations.

    Solves the Bellman equation directly using closed-form solution.

    Incremental formulation:
    - A = Φ^T (Φ - γΦ') + λI  (d, d)
    - b = Φ^T r  (d, 1)
    - w = solve(A, b)  (d, 1) via torch.linalg.solve for numerical stability

    Where:
    - Φ: (n, d) representations of current states
    - Φ': (n, d) representations of next states
    - γ: discount factor
    - r: (n, 1) rewards
    """

    @classmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        """Get method-specific parameters from config."""
        use_pca = method_config.preprocess_fraction > 0.0
        return {
            'policy_path': method_config.policy_path,
            'algorithm': method_config.algorithm,
            'ridge_lambda': method_config.ridge_lambda,
            'use_pca_projection': use_pca,
            'n_components': method_config.n_components if use_pca else None,
        }

    def _update_A_and_b(self, mini_batch: Dict[str, torch.Tensor], phi: torch.Tensor) -> None:
        """Update A and b for LSTD estimation.

        Accumulates:
        - A += Φ^T (Φ - γΦ')
        - b += Φ^T r

        Args:
            mini_batch: Dictionary containing mini-batch data
            phi: (batch_size, repr_dim+1) representations with bias
        """
        # Extract next state representations
        next_obs = mini_batch['next_observations'].to(self.device)
        next_representations = self.repr_extractor(next_obs)
        ones = torch.ones(next_representations.shape[0], 1, device=self.device)
        phi_next = torch.cat([next_representations, ones], dim=1)

        rewards = mini_batch['rewards'].to(self.device).unsqueeze(1)  # (n, 1)
        dones = mini_batch['dones'].to(self.device).unsqueeze(1)  # (n, 1)

        # Compute TD difference features: (Φ - γΦ')
        # For terminal states, Φ' = 0 (no next state value)
        gamma_mask = self.discount_factor * (1.0 - dones)  # (n, 1)
        phi_diff = phi - gamma_mask * phi_next  # (n, d)

        # Accumulate: A += Φ^T (Φ - γΦ')
        self.A = self.A + phi.T @ phi_diff  # (d, n) @ (n, d) = (d, d)

        # Accumulate: b += Φ^T r
        self.b = self.b + phi.T @ rewards  # (d, n) @ (n, 1) = (d, 1)

    def _compute_targets_for_metrics(self, mini_batch: Dict[str, torch.Tensor], phi: torch.Tensor) -> torch.Tensor:
        """Compute TD targets for metric evaluation.

        Args:
            mini_batch: Dictionary containing mini-batch data
            phi: (batch_size, repr_dim+1) representations with bias

        Returns:
            TD targets: r + γ * V(s') * (1 - done)
        """
        # Extract next state representations
        next_obs = mini_batch['next_observations'].to(self.device)
        next_representations = self.repr_extractor(next_obs)
        ones = torch.ones(next_representations.shape[0], 1, device=self.device)
        phi_next = torch.cat([next_representations, ones], dim=1)

        rewards = mini_batch['rewards'].to(self.device)
        dones = mini_batch['dones'].to(self.device)

        # Compute TD target: r + γ * V(s') * (1 - done)
        next_values = (phi_next @ self.w).squeeze(-1)
        targets = rewards + self.discount_factor * next_values * (1.0 - dones)

        return targets

    def compute_targets(self, mini_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute TD targets for evaluation."""
        rewards = mini_batch['rewards'].to(self.device)
        next_obs = mini_batch['next_observations'].to(self.device)
        dones = mini_batch['dones'].to(self.device)

        with torch.no_grad():
            # Get next state values
            next_representations = self.repr_extractor(next_obs)
            ones = torch.ones(next_representations.shape[0], 1, device=self.device)
            phi_next = torch.cat([next_representations, ones], dim=1)
            next_values = (phi_next @ self.w).squeeze(-1)

            # TD target: r + γ * V(s') * (1 - done)
            targets = rewards + self.discount_factor * next_values * (1.0 - dones)

        return targets


    def save(self, path):
        """Save estimator to disk."""
        torch.save({
            'value_net_state_dict': self.value_net.state_dict(),
            'repr_extractor_state_dict': self.repr_extractor.state_dict(),
            'A': self.A,
            'b': self.b,
            'w': self.w,
            'training_step': self.training_step,
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'normalize_observations': self.normalize_observations,
            'ridge_lambda': self.ridge_lambda,
            'algorithm': self.algorithm,
            'repr_dim': self.repr_dim,
            'discount_factor': self.discount_factor,
            'policy_path': self.policy_path,
        }, path)

    def get_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        config = super().get_config()
        config.update({
            'discount_factor': self.discount_factor,
            'estimator_type': 'least_squares_td',
        })
        return config
