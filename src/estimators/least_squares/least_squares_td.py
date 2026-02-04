"""Least Squares Temporal Difference (LSTD) estimator using policy network representations."""

import torch
from typing import Dict, Any

from .base import LeastSquaresEstimator


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
            'ridge_lambda': method_config.ridge_lambda,
            'use_pca_projection': use_pca,
            'n_components': method_config.n_components if use_pca else None,
        }

    def _update_A_and_b(self, feature_batch: Dict[str, torch.Tensor], phi: torch.Tensor) -> None:
        """Update A and b for LSTD estimation.

        Accumulates:
        - A += Φ^T (Φ - γΦ')
        - b += Φ^T r

        Args:
            feature_batch: Dictionary containing feature batch data
            phi: (batch_size, working_dim+1) features with bias
        """
        next_features = feature_batch['next_features']
        phi_next = self._get_features(next_features)

        rewards = feature_batch['rewards'].unsqueeze(1)
        dones = feature_batch['dones'].unsqueeze(1)

        gamma_mask = self.discount_factor * (1.0 - dones)
        phi_diff = phi - gamma_mask * phi_next

        self.A = self.A + phi.T @ phi_diff
        self.b = self.b + phi.T @ rewards

    def _compute_targets_for_metrics(self, feature_batch: Dict[str, torch.Tensor], phi: torch.Tensor) -> torch.Tensor:
        """Compute TD targets for metric evaluation.

        Args:
            feature_batch: Dictionary containing feature batch data
            phi: (batch_size, working_dim+1) features with bias

        Returns:
            TD targets: r + γ * V(s') * (1 - done)
        """
        next_features = feature_batch['next_features']
        phi_next = self._get_features(next_features)

        rewards = feature_batch['rewards']
        dones = feature_batch['dones']

        next_values = (phi_next @ self.w).squeeze(-1)
        targets = rewards + self.discount_factor * next_values * (1.0 - dones)

        return targets

    def compute_targets(self, feature_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute TD targets from feature batch.

        Args:
            feature_batch: Dictionary with feature data (includes next_features, rewards, dones)

        Returns:
            TD targets: r + γ * V(s') * (1 - done)
        """
        next_features = feature_batch['next_features']
        phi_next = self._get_features(next_features)

        rewards = feature_batch['rewards']
        dones = feature_batch['dones']

        with torch.no_grad():
            next_values = (phi_next @ self.w).squeeze(-1)
            targets = rewards + self.discount_factor * next_values * (1.0 - dones)

        return targets

    def get_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        config = super().get_config()
        config.update({
            'discount_factor': self.discount_factor,
            'estimator_type': 'least_squares_td',
        })
        return config
