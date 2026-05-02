"""Shared helpers for the per-mode evaluators."""

from pathlib import Path
from typing import Optional

import numpy as np

from src.config import BaseEstimatorConfig, ExperimentConfig
from src.estimators import ESTIMATOR_REGISTRY


def find_method_config(config: ExperimentConfig, method_name: str) -> BaseEstimatorConfig:
    for mc in config.value_estimators.method_configs:
        if mc.name == method_name:
            return mc
    raise ValueError(f"Method '{method_name}' not found in config")


def load_v_estimator(config: ExperimentConfig, method_name: str,
                     n_episodes: int, batch_idx: int, device: str = "cpu"):
    method_config = find_method_config(config, method_name)
    estimator_dir = config.get_estimator_dir(method_config)
    ckpt_path = estimator_dir / str(n_episodes) / f"batch_{batch_idx}" / "estimator.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Estimator checkpoint not found: {ckpt_path}")
    cls = ESTIMATOR_REGISTRY[type(method_config)]
    return cls.load_estimator(ckpt_path, device=device)


def results_root(config: ExperimentConfig, method_name: str) -> Path:
    """`{method}_estimator_NNN/eval_NNN/results/`."""
    method_config = find_method_config(config, method_name)
    return config.get_eval_dir(method_config) / "results"


def lam_dir(lam: float) -> str:
    """Format a λ as a stable directory token, e.g. 0.95 → 'lambda_0.95'."""
    return f"lambda_{lam:g}"


def gae_advantage(rewards: np.ndarray, values: np.ndarray, gamma: float,
                  lam: float, last_value: float = 0.0) -> np.ndarray:
    """A_GAE-λ at every step of one trajectory.

    A_t = δ_t + γλ · A_{t+1}, where δ_t = r_t + γ·V(s_{t+1}) − V(s_t).
    Truncates with last_value (0 = treat as terminal).
    """
    T = len(rewards)
    advs = np.zeros(T, dtype=np.float64)
    last = 0.0
    for t in range(T - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < T else last_value
        delta = float(rewards[t]) + gamma * next_v - values[t]
        advs[t] = delta + gamma * lam * last
        last = advs[t]
    return advs


def mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """G_t = Σ γ^k r_{t+k} for one trajectory."""
    T = len(rewards)
    g = np.zeros(T, dtype=np.float64)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        g[t] = running
    return g
