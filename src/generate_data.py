"""Generate episode data using a trained policy."""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

from src.config import ExperimentConfig
from src.env_utils import ALGORITHM_MAP, create_vec_env


def collect_episodes_parallel(env, model, n_episodes: int, deterministic: bool = False,
                              use_vec_normalize: bool = False, seed: int = None) -> List[Dict[str, np.ndarray]]:
    """Collect multiple episodes in parallel using vectorized environments.

    Runs n_envs episodes in parallel, waits for all to complete, then repeats.
    n_envs is capped at n_episodes to avoid bias toward short episodes.
    """
    n_envs = env.num_envs
    completed_episodes = []
    first_reset = True

    while len(completed_episodes) < n_episodes:
        n_active = min(n_envs, n_episodes - len(completed_episodes))

        episode_data = [
            {'observations': [], 'actions': [], 'rewards': [], 'dones': [], 'next_observations': []}
            for _ in range(n_active)
        ]
        active = [True] * n_active

        if first_reset and seed is not None:
            env.seed(seed)
            first_reset = False
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        while any(active):
            if use_vec_normalize:
                original_obs = env.get_original_obs()
            else:
                original_obs = obs

            for i in range(n_active):
                if active[i]:
                    episode_data[i]['observations'].append(original_obs[i].copy())

            action, _ = model.predict(obs, deterministic=deterministic)

            step_result = env.step(action)
            next_obs = step_result[0]
            rewards = step_result[1]
            dones = step_result[2]
            infos = step_result[3] if len(step_result) > 3 else [{} for _ in range(n_envs)]

            if use_vec_normalize:
                original_next_obs = env.get_original_obs()
            else:
                original_next_obs = next_obs

            for i in range(n_active):
                if active[i]:
                    episode_data[i]['actions'].append(action[i].copy() if isinstance(action[i], np.ndarray) else action[i])
                    episode_data[i]['rewards'].append(float(rewards[i]))
                    episode_data[i]['dones'].append(bool(dones[i]))
                    episode_data[i]['next_observations'].append(original_next_obs[i].copy())

                    if dones[i]:
                        info = infos[i] if isinstance(infos, (list, tuple)) else {}
                        truncated = info.get('TimeLimit.truncated', False)
                        completed_episodes.append({
                            'observations': np.array(episode_data[i]['observations']),
                            'actions': np.array(episode_data[i]['actions']),
                            'rewards': np.array(episode_data[i]['rewards']),
                            'dones': np.array(episode_data[i]['dones']),
                            'next_observations': np.array(episode_data[i]['next_observations']),
                            'truncated': truncated,
                        })
                        active[i] = False

            obs = next_obs

    np.random.shuffle(completed_episodes)
    return completed_episodes[:n_episodes]


def episodes_to_batch(episodes: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Convert a list of episode dicts into batch dict format."""
    batch_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'next_observations': [],
        'truncated': [],
        'episode_lengths': [],
        'episode_returns': [],
    }

    for episode in episodes:
        for key in episode:
            batch_data[key].append(episode[key])
        batch_data['episode_lengths'].append(len(episode['rewards']))
        batch_data['episode_returns'].append(episode['rewards'].sum())

    for key in batch_data.keys():
        if key not in ['episode_lengths', 'episode_returns', 'truncated']:
            batch_data[key] = np.array(batch_data[key], dtype=object)
    batch_data['truncated'] = np.array(batch_data['truncated'], dtype=bool)
    batch_data['episode_lengths'] = np.array(batch_data['episode_lengths'])
    batch_data['episode_returns'] = np.array(batch_data['episode_returns'])
    return batch_data


def save_and_log_batch(batch_data: dict, output_path: Path, batch_name: str) -> dict:
    """Save batch data and return statistics."""
    np.savez_compressed(output_path, **batch_data)

    mean_return = batch_data['episode_returns'].mean()
    std_return = batch_data['episode_returns'].std()
    mean_length = batch_data['episode_lengths'].mean()

    print(f"  Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"  Mean episode length: {mean_length:.1f}\n")

    return {
        'name': batch_name,
        'mean_return': mean_return,
        'std_return': std_return,
        'mean_length': mean_length,
    }


def _compute_batch_seeds(config: ExperimentConfig):
    """Compute deterministic seed assignments for tuning + training batches."""
    base = config.data_generation.seed
    seeds = {}
    idx = 0

    if config.data_generation.tuning_episodes > 0:
        seeds['tuning'] = base + idx
        idx += 1

    seeds['training'] = []
    for i in range(config.data_generation.n_batches):
        seeds['training'].append(base + idx)
        idx += 1

    return seeds


def generate_data(config: ExperimentConfig, policy_path: Path, output_dir: Path,
                  phase: str = "all",
                  start_batch_idx: int = 0, end_batch_idx: int = None):
    """Generate episode batches using trained policy.

    Phases:
        "all" — tuning + training
        "tuning" — only batch_tuning (+ validation)
        "training" — only regular batches (respects start/end_batch_idx)
    """
    valid_phases = ("all", "tuning", "training")
    if phase not in valid_phases:
        raise ValueError(f"Invalid phase '{phase}'. Must be one of {valid_phases}")

    output_dir.mkdir(parents=True, exist_ok=True)

    policy_metadata = {}
    policy_metadata_path = policy_path.parent / "policy_metadata.json"
    if policy_metadata_path.exists():
        with open(policy_metadata_path, 'r') as f:
            policy_metadata = json.load(f)

    if config.policy.algorithm not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {config.policy.algorithm}")

    AlgorithmClass = ALGORITHM_MAP[config.policy.algorithm]
    model = AlgorithmClass.load(policy_path)

    vec_normalize_path = policy_path.parent / "vec_normalize.pkl" if config.policy.use_vec_normalize else None
    env_params = config.get_data_env_params()
    env, use_vec_normalize = create_vec_env(
        config,
        n_envs=config.data_generation.n_envs,
        use_monitor=False,
        vec_normalize_path=vec_normalize_path,
        seed=config.data_generation.seed,
        **env_params,
    )

    if use_vec_normalize:
        print(f"Loaded VecNormalize stats from {vec_normalize_path}")

    print(f"\nGenerating data using policy from {policy_path}")
    print(f"Algorithm: {config.policy.algorithm}")
    print(f"Environment: {config.environment.name}")
    print(f"Phase: {phase}")
    print(f"VecNormalize: {use_vec_normalize}")
    print(f"Deterministic policy: {config.data_generation.deterministic_policy}")
    print(f"Output directory: {output_dir}\n")

    batch_seeds = _compute_batch_seeds(config)
    all_batch_stats = []
    val_eps = config.data_generation.validation_episodes_per_batch

    def collect_and_save(batch_name, n_train, n_val, seed):
        total = n_train + n_val
        print(f"Collecting {batch_name} ({n_train}+{n_val} val = {total} episodes)")
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        episodes = collect_episodes_parallel(env, model, total,
                                             config.data_generation.deterministic_policy,
                                             use_vec_normalize, seed=seed)

        train_data = episodes_to_batch(episodes[:n_train])
        stats = save_and_log_batch(train_data, output_dir / f"{batch_name}.npz", batch_name)
        stats['batch_seed'] = seed
        all_batch_stats.append(stats)

        if n_val > 0:
            val_name = f"{batch_name}_validation"
            val_data = episodes_to_batch(episodes[n_train:])
            val_stats = save_and_log_batch(val_data, output_dir / f"{val_name}.npz", val_name)
            val_stats['batch_seed'] = seed
            all_batch_stats.append(val_stats)

        return total

    total_episodes = 0

    if phase in ("all", "tuning") and 'tuning' in batch_seeds:
        total_episodes += collect_and_save(
            "batch_tuning", config.data_generation.tuning_episodes,
            val_eps, batch_seeds['tuning'])

    if phase in ("all", "training"):
        for i in range(config.data_generation.n_batches):
            if i < start_batch_idx or (end_batch_idx is not None and i >= end_batch_idx):
                continue
            total_episodes += collect_and_save(
                f"batch_{i}", config.data_generation.episodes_per_batch,
                val_eps, batch_seeds['training'][i])

    stats_file = output_dir / "data_statistics.npz"
    np.savez(
        stats_file,
        batch_stats=all_batch_stats,
        config_seed=config.data_generation.seed,
        total_episodes=total_episodes,
    )

    data_metadata = {
        'policy_path': str(policy_path),
        'deterministic_policy': config.data_generation.deterministic_policy,
        'total_episodes': int(total_episodes),
        'n_batches': config.data_generation.n_batches,
        'episodes_per_batch': config.data_generation.episodes_per_batch,
        'tuning_episodes': config.data_generation.tuning_episodes,
        'validation_episodes_per_batch': config.data_generation.validation_episodes_per_batch,
        'seed': config.data_generation.seed,
        'policy_metadata': policy_metadata,
    }

    metadata_path = output_dir / "data_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(data_metadata, f, indent=2)
    print(f"Data generation metadata saved to {metadata_path}")

    print(f"Data generation complete! Files saved to {output_dir}")
    print(f"Total episodes: {total_episodes}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Generate episode data using trained policy")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument("--policy-path", type=Path, default=None,
                        help="Path to trained policy (default: experiments/<experiment_id>/policy/policy_final.zip)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: experiments/<experiment_id>/data)")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "tuning", "training"],
                        help="Which data to generate (default: all)")
    parser.add_argument("--start-batch-idx", type=int, default=0,
                        help="Skip training batches before this index (for resuming)")
    parser.add_argument("--end-batch-idx", type=int, default=None,
                        help="Stop after this training batch index (exclusive)")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Reserved for compatibility; unused now that paired-state generation is gone")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    policy_path = args.policy_path or config.get_policy_dir() / "policy_final.zip"
    output_dir = args.output_dir or config.get_data_dir()

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    generate_data(config, policy_path, output_dir,
                  phase=args.phase,
                  start_batch_idx=args.start_batch_idx,
                  end_batch_idx=args.end_batch_idx)


if __name__ == "__main__":
    main()
