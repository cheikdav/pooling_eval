"""Value estimators for reinforcement learning."""

from typing import Type, Dict
from src.estimators.base import ValueEstimator
from src.estimators.neural_net import NeuralNetEstimator, MonteCarloEstimator, TDEstimator
from src.config import MonteCarloConfig, TDConfig, BaseEstimatorConfig

ESTIMATOR_REGISTRY: Dict[Type[BaseEstimatorConfig], Type[ValueEstimator]] = {
    MonteCarloConfig: MonteCarloEstimator,
    TDConfig: TDEstimator,
}

__all__ = [
    'ValueEstimator',
    'NeuralNetEstimator',
    'MonteCarloEstimator',
    'TDEstimator',
    'ESTIMATOR_REGISTRY',
]
