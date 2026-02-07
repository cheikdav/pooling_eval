"""Utilities for environment creation and configuration."""

import gymnasium as gym
from pathlib import Path
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from typing import Tuple, Callable

from src.config import ExperimentConfig
from src.estimators.neural_net import MonteCarloEstimator, DQNEstimator
from src.estimators.least_squares import LeastSquaresMCEstimator, LeastSquaresTDEstimator


# Centralized mapping from method names to estimator classes
# Handles all method name variations including RBF and NNLS variants
ESTIMATOR_CLASSES = {
    # Base methods
    'monte_carlo': MonteCarloEstimator,
    'dqn': DQNEstimator,
    'least_squares_mc': LeastSquaresMCEstimator,
    'least_squares_td': LeastSquaresTDEstimator,
    # RBF feature extractor variants (same underlying classes)
    'least_squares_mc_rbf': LeastSquaresMCEstimator,
    'least_squares_td_rbf': LeastSquaresTDEstimator,
    # NNLS (Non-negative least squares) variants with policy representation features
    'nnls_mc': MonteCarloEstimator,
    'nnls_td': DQNEstimator,
}


def make_env_fn(env_name: str, use_monitor: bool = True, seed: int = None) -> Callable:
    """Create a factory function for environment creation.

    Args:
        env_name: Gymnasium environment name
        use_monitor: Whether to wrap environment with Monitor for tracking episode statistics
        seed: Random seed for this environment instance

    Returns:
        Function that creates and returns the environment
    """
    def _make():
        env = gym.make(env_name)
        if use_monitor:
            env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return env
    return _make


def create_vec_env(
    config: ExperimentConfig,
    n_envs: int = 1,
    use_monitor: bool = True,
    vec_normalize_path: Path = None,
    seed: int = None,
) -> Tuple[SubprocVecEnv, bool]:
    """Create a vectorized environment with optional normalization.

    Args:
        config: Experiment configuration
        n_envs: Number of parallel environments
        use_monitor: Whether to wrap with Monitor
        vec_normalize_path: Path to load existing VecNormalize stats
        seed: Random seed (defaults to config.seed)

    Returns:
        Tuple of (environment, use_vec_normalize flag)
    """
    if seed is None:
        seed = config.seed

    env_fns = [
        make_env_fn(config.environment.name, use_monitor, seed + i)
        for i in range(n_envs)
    ]
    env = SubprocVecEnv(env_fns, start_method='fork')

    use_vec_normalize = False

    if config.policy.use_vec_normalize:
        if vec_normalize_path is not None and vec_normalize_path.exists():
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
            use_vec_normalize = True
        else:
            vec_normalize_kwargs = {
                "norm_obs": config.policy.normalize_obs,
                "norm_reward": config.policy.normalize_reward,
                "gamma": config.policy.gamma,
            }
            vec_normalize_kwargs.update(config.policy.vec_normalize_kwargs)
            env = VecNormalize(env, **vec_normalize_kwargs)
            use_vec_normalize = True

    return env, use_vec_normalize
