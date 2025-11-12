"""Value estimators for reinforcement learning."""

from typing import Type, Dict
from src.estimators.base import ValueEstimator
from src.estimators.monte_carlo import MonteCarloEstimator
from src.estimators.dqn import DQNEstimator
from src.config import MonteCarloConfig, DQNConfig, BaseEstimatorConfig

# Registry mapping config class to estimator class
ESTIMATOR_REGISTRY: Dict[Type[BaseEstimatorConfig], Type[ValueEstimator]] = {
    MonteCarloConfig: MonteCarloEstimator,
    DQNConfig: DQNEstimator,
}

__all__ = [
    'ValueEstimator',
    'MonteCarloEstimator',
    'DQNEstimator',
    'ESTIMATOR_REGISTRY',
]
