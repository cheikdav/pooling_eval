"""Utilities for environment creation and configuration."""

import gymnasium as gym
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from typing import Tuple, Callable

from src.config import ExperimentConfig


def make_env_fn(env_name: str, use_monitor: bool = True) -> Callable:
    """Create a factory function for environment creation.

    Args:
        env_name: Gymnasium environment name
        use_monitor: Whether to wrap environment with Monitor for tracking episode statistics

    Returns:
        Function that creates and returns the environment
    """
    def _make():
        env = gym.make(env_name)
        if use_monitor:
            env = Monitor(env)
        return env
    return _make


def create_vec_env(
    config: ExperimentConfig,
    use_monitor: bool = True,
    vec_normalize_path: Path = None,
    seed: int = None,
) -> Tuple[DummyVecEnv, bool]:
    """Create a vectorized environment with optional normalization.

    Args:
        config: Experiment configuration
        use_monitor: Whether to wrap with Monitor (typically True for training, False for evaluation)
        vec_normalize_path: Path to load existing VecNormalize stats (for evaluation)
        seed: Random seed (defaults to config.seed)

    Returns:
        Tuple of (environment, use_vec_normalize flag)
    """
    if seed is None:
        seed = config.seed

    env = DummyVecEnv([make_env_fn(config.environment.name, use_monitor)])

    use_vec_normalize = False

    if config.policy.use_vec_normalize:
        if vec_normalize_path is not None and vec_normalize_path.exists():
            # Load existing VecNormalize for evaluation
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
            use_vec_normalize = True
        else:
            # Create new VecNormalize for training
            vec_normalize_kwargs = {
                "norm_obs": config.policy.normalize_obs,
                "norm_reward": config.policy.normalize_reward,
                "gamma": config.policy.gamma,
            }
            vec_normalize_kwargs.update(config.policy.vec_normalize_kwargs)
            env = VecNormalize(env, **vec_normalize_kwargs)
            use_vec_normalize = True

    env.seed(seed)

    return env, use_vec_normalize
