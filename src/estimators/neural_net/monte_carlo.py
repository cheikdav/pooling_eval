"""Monte Carlo value estimator."""

import torch
import numpy as np
from typing import Dict, Any

from .base import NeuralNetEstimator


class MonteCarloEstimator(NeuralNetEstimator):
    """Monte Carlo value estimator."""

    @classmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        return {}

    def compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = np.zeros_like(rewards[0], dtype=np.float32)

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.discount_factor * running_return
            returns[t] = running_return

        return returns

    def compute_targets(self, feature_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return feature_batch['mc_returns']

    def get_config(self) -> Dict:
        config = super().get_config()
        config['discount_factor'] = self.discount_factor
        config['estimator_type'] = 'monte_carlo'
        return config
