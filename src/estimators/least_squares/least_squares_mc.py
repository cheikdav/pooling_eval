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
        use_pca = method_config.preprocess_fraction > 0.0
        return {
            'ridge_lambda': method_config.ridge_lambda,
            'use_pca_projection': use_pca,
            'n_components': method_config.n_components if use_pca else None,
        }

    def _update_A_and_b(self, feature_batch: Dict[str, torch.Tensor], phi: torch.Tensor) -> None:
        """Update A and b for Monte Carlo estimation.

        Accumulates: A += Φ^T Φ and b += Φ^T y

        Args:
            feature_batch: Dictionary containing feature batch data
            phi: (batch_size, working_dim+1) features with bias
        """
        targets = feature_batch['mc_returns'].unsqueeze(1)

        self.A = self.A + phi.T @ phi
        self.b = self.b + phi.T @ targets

    def _compute_targets_for_metrics(self, feature_batch: Dict[str, torch.Tensor], phi: torch.Tensor) -> torch.Tensor:
        """Compute Monte Carlo targets for metric evaluation.

        Args:
            feature_batch: Dictionary containing feature batch data
            phi: (batch_size, working_dim+1) features with bias

        Returns:
            Monte Carlo returns
        """
        return feature_batch['mc_returns']

    def compute_targets(self, feature_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Monte Carlo targets from feature batch.

        Args:
            feature_batch: Dictionary with feature data (includes mc_returns)

        Returns:
            Monte Carlo returns
        """
        return feature_batch['mc_returns']

    def get_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        config = super().get_config()
        config['estimator_type'] = 'least_squares_mc'
        return config
