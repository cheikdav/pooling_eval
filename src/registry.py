"""Registry for parameter-based experiment directory management.

Provides sequential numbered directories (e.g., policy_001, data_001) where each
directory's params.json records the parameters that produced it. When resolving a
directory, existing dirs are scanned for a match; if none is found, the next
sequential number is created.
"""

import json
import re
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict

from src.config import ExperimentConfig, BaseEstimatorConfig


def _normalize_params(params: dict) -> dict:
    """Recursively sort and normalize a params dict for stable comparison."""
    result = {}
    for k, v in sorted(params.items()):
        if isinstance(v, dict):
            result[k] = _normalize_params(v)
        elif isinstance(v, list):
            result[k] = [_normalize_params(x) if isinstance(x, dict) else x for x in v]
        else:
            result[k] = v
    return result


def _params_match(a: dict, b: dict) -> bool:
    """Check if two param dicts are equal after normalization."""
    return json.dumps(_normalize_params(a), sort_keys=True) == json.dumps(_normalize_params(b), sort_keys=True)


def resolve_dir(parent_dir: Path, prefix: str, params: dict) -> Path:
    """Find or create a numbered directory matching the given params.

    Scans parent_dir for dirs like {prefix}_001, {prefix}_002, etc.
    If a match is found (params.json matches), returns that directory.
    Otherwise creates the next sequential number and writes params.json.

    Args:
        parent_dir: Directory to scan/create in
        prefix: Directory name prefix (e.g., "policy", "data")
        params: Parameter dict to match against

    Returns:
        Path to the matched or newly created directory
    """
    parent_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    existing = []
    for d in parent_dir.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                existing.append((int(m.group(1)), d))

    existing.sort(key=lambda x: x[0])

    # Check for match
    for num, d in existing:
        params_file = d / "params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                stored = json.load(f)
            if _params_match(params, stored):
                return d

    # No match — create next sequential
    next_num = (existing[-1][0] + 1) if existing else 1
    new_dir = parent_dir / f"{prefix}_{next_num:03d}"
    new_dir.mkdir(parents=True, exist_ok=True)

    with open(new_dir / "params.json", 'w') as f:
        json.dump(_normalize_params(params), f, indent=2, sort_keys=True)

    return new_dir


# --- Parameter extraction for each hierarchy level ---

def get_policy_params(config: ExperimentConfig) -> dict:
    """Extract parameters that define the policy level identity."""
    p = config.policy
    return {
        'algorithm': p.algorithm,
        'total_timesteps': p.total_timesteps,
        'learning_rate': p.learning_rate,
        'gamma': p.gamma,
        'n_steps': p.n_steps,
        'batch_size': p.batch_size,
        'n_epochs': p.n_epochs,
        'gae_lambda': p.gae_lambda,
        'n_envs': p.n_envs,
        'ent_coef': p.ent_coef,
        'clip_range': p.clip_range,
        'max_grad_norm': p.max_grad_norm,
        'vf_coef': p.vf_coef,
        'learning_starts': p.learning_starts,
        'buffer_size': p.buffer_size,
        'tau': p.tau,
        'train_freq': p.train_freq,
        'gradient_steps': p.gradient_steps,
        'policy_kwargs': p.policy_kwargs,
        'use_vec_normalize': p.use_vec_normalize,
        'normalize_obs': p.normalize_obs,
        'normalize_reward': p.normalize_reward,
        'seed': config.seed,
        'network': {
            'hidden_sizes': config.network.hidden_sizes,
            'activation': config.network.activation,
        },
        'environment': {
            'max_episode_steps': config.environment.max_episode_steps,
            'reset_noise_scale': config.environment.reset_noise_scale,
            'action_noise_std': config.environment.action_noise_std,
        },
        'code_version': config.code_versions.policy,
    }


def get_data_params(config: ExperimentConfig) -> dict:
    """Extract parameters that define the data level identity."""
    dg = config.data_generation
    return {
        'n_batches': dg.n_batches,
        'episodes_per_batch': dg.episodes_per_batch,
        'deterministic_policy': dg.deterministic_policy,
        'n_envs': dg.n_envs,
        'tuning_episodes': dg.tuning_episodes,
        'validation_episodes_per_batch': dg.validation_episodes_per_batch,
        'code_version': config.code_versions.data,
    }


def get_estimator_params(config: ExperimentConfig, method_config: BaseEstimatorConfig) -> dict:
    """Extract parameters that define the estimator level identity.

    Includes method-specific config and shared training params,
    but excludes episode_subsets (those are just subfolders).
    """
    t = config.value_estimators.training
    shared = {
        'max_epochs': t.max_epochs,
        'batch_size': t.batch_size,
        'convergence_patience': t.convergence_patience,
        'convergence_threshold': t.convergence_threshold,
        'eval_frequency': t.eval_frequency,
        'gamma': t.gamma,
        'shuffle_frequency': t.shuffle_frequency,
        'truncation_coefficient': t.truncation_coefficient,
        'reward_centering': t.reward_centering,
    }

    method = method_config.to_dict()

    return {
        'method': method,
        'shared_training': shared,
        'network': {
            'hidden_sizes': config.network.hidden_sizes,
            'activation': config.network.activation,
        },
        'code_version': config.code_versions.estimator,
    }


def get_eval_params(config: ExperimentConfig) -> dict:
    """Extract parameters that define the evaluation level identity."""
    ev = config.evaluation
    return {
        'eval_episodes': ev.eval_episodes,
        'paired_states_n_pairs': ev.paired_states_n_pairs,
        'paired_states_n_trajectories': ev.paired_states_n_trajectories,
        'paired_states_seed': ev.paired_states_seed,
        'gamma': config.value_estimators.training.gamma,
        'code_version': config.code_versions.evaluation,
    }
