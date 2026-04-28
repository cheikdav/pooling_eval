"""Monte Carlo value estimator."""

from typing import Dict, Any
import torch

from .base import NeuralNetEstimator


class MonteCarloEstimator(NeuralNetEstimator):
    """Monte Carlo value estimator. Target is the uncentered MC return."""

    def compute_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch['mc_returns'] - self.reward_offset

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg['estimator_type'] = 'monte_carlo'
        return cfg
