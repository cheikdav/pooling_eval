"""Value estimators for reinforcement learning."""

from typing import Type, Dict
from src.estimators.base import ValueEstimator
from src.estimators.neural_net import (
    NeuralNetEstimator,
    MonteCarloEstimator,
    TDEstimator,
    TDLambdaEstimator,
)
from src.config import MonteCarloConfig, TDConfig, TDLambdaConfig, BaseEstimatorConfig

ESTIMATOR_REGISTRY: Dict[Type[BaseEstimatorConfig], Type[ValueEstimator]] = {
    MonteCarloConfig: MonteCarloEstimator,
    TDConfig: TDEstimator,
    TDLambdaConfig: TDLambdaEstimator,
}

__all__ = [
    'ValueEstimator',
    'NeuralNetEstimator',
    'MonteCarloEstimator',
    'TDEstimator',
    'TDLambdaEstimator',
    'ESTIMATOR_REGISTRY',
]
