"""Least Squares Monte Carlo estimator using policy network representations."""

import torch
from typing import Dict, Any

from .base import LeastSquaresEstimator


class LeastSquaresMCEstimator(LeastSquaresEstimator):
    """Least Squares Monte Carlo estimator using frozen policy representations.

    Uses incremental updates by accumulating A = Φ^T Φ + λI and b = Φ^T y.
    Solution: w = solve(A, b) computed via torch.linalg.solve for numerical stability.
    """

    @classmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        """Get method-specific parameters from config."""
        return {
            'ridge_lambda': method_config.ridge_lambda,
            'n_components': method_config.n_components,
        }

    def _update_A_and_b(self, feature_batch: Dict[str, torch.Tensor]) -> None:
        """Update A and b for Monte Carlo estimation.

        Accumulates: A += Φ^T Φ and b += Φ^T y

        Args:
            feature_batch: Dictionary containing feature batch data (features include bias)
        """
        features = feature_batch['features']  # Already includes bias
        targets = feature_batch['mc_returns'].unsqueeze(1)

        self.A = self.A + features.T @ features
        self.b = self.b + features.T @ targets

    def compute_targets(self, feature_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Monte Carlo targets from feature batch.

        Args:
            feature_batch: Dictionary with feature data (includes mc_returns)

        Returns:
            Monte Carlo returns
        """
        # MC targets don't depend on features, just return mc_returns
        return feature_batch['mc_returns']

    def get_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        config = super().get_config()
        config['estimator_type'] = 'least_squares_mc'
        return config
