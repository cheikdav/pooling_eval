"""Value estimators for reinforcement learning."""

from typing import Type, Dict
from src.estimators.base import ValueEstimator
from src.estimators.neural_net import NeuralNetEstimator, MonteCarloEstimator, DQNEstimator
from src.estimators.least_squares import LeastSquaresEstimator, LeastSquaresMCEstimator, LeastSquaresTDEstimator
from src.config import MonteCarloConfig, DQNConfig, LeastSquaresMCConfig, LeastSquaresTDConfig, BaseEstimatorConfig

# Registry mapping config class to estimator class
ESTIMATOR_REGISTRY: Dict[Type[BaseEstimatorConfig], Type[ValueEstimator]] = {
    MonteCarloConfig: MonteCarloEstimator,
    DQNConfig: DQNEstimator,
    LeastSquaresMCConfig: LeastSquaresMCEstimator,
    LeastSquaresTDConfig: LeastSquaresTDEstimator,
}

__all__ = [
    'ValueEstimator',
    'NeuralNetEstimator',
    'LeastSquaresEstimator',
    'MonteCarloEstimator',
    'DQNEstimator',
    'LeastSquaresMCEstimator',
    'LeastSquaresTDEstimator',
    'ESTIMATOR_REGISTRY',
]
