"""Utilities for environment creation and configuration."""

import gymnasium as gym
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from typing import Optional, Tuple, Callable

from src.config import ExperimentConfig
from src.estimators.neural_net import MonteCarloEstimator, DQNEstimator
from src.estimators.least_squares import LeastSquaresMCEstimator, LeastSquaresTDEstimator

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='gymnasium')

class ActionNoiseWrapper(gym.Wrapper):
    """Adds fixed Gaussian noise to actions, clipped to the action space bounds."""

    def __init__(self, env: gym.Env, noise_std: float):
        super().__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        noisy_action = action + self.np_random.normal(0, self.noise_std, size=action.shape)
        noisy_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
        return self.env.step(noisy_action)


ALGORITHM_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}

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


def make_env_fn(env_name: str, use_monitor: bool = True, seed: int = None, max_episode_steps: int = None, reset_noise_scale: float = None, action_noise_std: float = None) -> Callable:
    """Create a factory function for environment creation.

    Args:
        env_name: Gymnasium environment name
        use_monitor: Whether to wrap environment with Monitor for tracking episode statistics
        seed: Random seed for this environment instance
        max_episode_steps: Maximum number of steps per episode (None = use default)
        reset_noise_scale: Noise scale for MuJoCo environment resets (None = use MuJoCo default of 0.01)

    Returns:
        Function that creates and returns the environment
    """
    def _make():
        env_kwargs = {}
        if max_episode_steps is not None:
            env_kwargs['max_episode_steps'] = max_episode_steps
        if reset_noise_scale is not None:
            env_kwargs['reset_noise_scale'] = reset_noise_scale

        env = gym.make(env_name, **env_kwargs)
        if action_noise_std is not None:
            env = ActionNoiseWrapper(env, action_noise_std)
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
    max_episode_steps: Optional[int] = None,
    reset_noise_scale: Optional[float] = None,
    action_noise_std: Optional[float] = None,
) -> Tuple[SubprocVecEnv, bool]:
    """Create a vectorized environment with optional normalization.

    Caller is responsible for resolving env params (via config.get_policy_env_params()
    or config.get_data_env_params()) and passing them explicitly.

    Returns:
        Tuple of (environment, use_vec_normalize flag)
    """
    if seed is None:
        raise ValueError("seed must be provided to create_vec_env")

    env_fns = [
        make_env_fn(
            config.environment.name,
            use_monitor,
            seed + i,
            max_episode_steps,
            reset_noise_scale,
            action_noise_std,
        )
        for i in range(n_envs)
    ]
    env = SubprocVecEnv(env_fns, start_method='fork')

    use_vec_normalize = False

    if config.policy.use_vec_normalize:
        if vec_normalize_path is not None:
            if not vec_normalize_path.exists():
                raise FileNotFoundError(f"VecNormalize path {vec_normalize_path} does not exist.")
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
