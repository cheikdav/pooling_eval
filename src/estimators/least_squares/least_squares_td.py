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
        return {
            'ridge_lambda': method_config.ridge_lambda,
            'n_components': method_config.n_components,
        }

    def _update_A_and_b(self, feature_batch: Dict[str, torch.Tensor]) -> None:
        """Update A and b for LSTD estimation.

        Accumulates:
        - A += Φ^T (Φ - γΦ')
        - b += Φ^T r

        Args:
            feature_batch: Dictionary containing feature batch data (features include bias)
        """
        features = feature_batch['features']  # Already includes bias
        next_features = feature_batch['next_features']  # Already includes bias

        rewards = feature_batch['rewards'].unsqueeze(1)
        dones = feature_batch['dones'].unsqueeze(1)

        gamma_mask = self.discount_factor * (1.0 - dones)
        features_diff = features - gamma_mask * next_features

        # Debug: Check feature similarity and magnitude
        features_norm = torch.norm(features, dim=1).mean().item()
        features_diff_norm = torch.norm(features_diff, dim=1).mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(features, next_features, dim=1).mean().item()
        print(f"[LSTD DEBUG] Features: {features_norm:.4f}, Diff: {features_diff_norm:.4f}, Ratio: {features_diff_norm/features_norm:.4f}, Cosine: {cosine_sim:.4f}")

        self.A = self.A + features.T @ features_diff
        self.b = self.b + features.T @ rewards

    def compute_targets(self, feature_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute TD targets from feature batch.

        Args:
            feature_batch: Dictionary with feature data (includes next_features, rewards, dones)

        Returns:
            TD targets: r + γ * V(s') * (1 - done)
        """
        next_features = feature_batch['next_features']
        rewards = feature_batch['rewards']
        dones = feature_batch['dones']

        with torch.no_grad():
            # Use _predict which will call _update_w() if needed
            next_values = torch.FloatTensor(self._predict(next_features)).to(self.device)
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
