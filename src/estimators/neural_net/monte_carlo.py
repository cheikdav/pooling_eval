"""Monte Carlo value estimator."""

from typing import Dict, Any
import torch

from .base import NeuralNetEstimator


class MonteCarloEstimator(NeuralNetEstimator):
    """Monte Carlo value estimator. Targets are uncentered MC returns,
    written once into self.cached_targets at fit start."""

    def recompute_targets(self, dataset) -> None:
        self.cached_targets = (dataset.mc_returns - self.reward_offset).to(self.device)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg['estimator_type'] = 'monte_carlo'
        return cfg
