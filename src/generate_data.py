"""Generate episode data using a trained policy."""

import argparse
import json
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, A2C, SAC, TD3
from tqdm import tqdm
from typing import Dict, List

from src.config import ExperimentConfig
from src.env_utils import create_vec_env


ALGORITHM_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


def collect_episodes_parallel(env, model, n_episodes: int, deterministic: bool = False,
                              use_vec_normalize: bool = False) -> List[Dict[str, np.ndarray]]:
    """Collect multiple episodes in parallel using vectorized environments.

    Args:
        env: VecEnv with n_envs parallel environments
        model: Trained SB3 model
        n_episodes: Total number of episodes to collect
        deterministic: Whether to use deterministic actions
        use_vec_normalize: Whether VecNormalize is used

    Returns:
        List of episode dictionaries, each containing:
            - observations: (T, obs_dim) array
            - actions: (T,) or (T, act_dim) array
            - rewards: (T,) array
            - dones: (T,) array
            - next_observations: (T, obs_dim) array
    """
    n_envs = env.num_envs
    completed_episodes = []

    episode_data = {
        i: {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'next_observations': []
        }
        for i in range(n_envs)
    }

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    while len(completed_episodes) < n_episodes:
        if use_vec_normalize:
            original_obs = env.get_original_obs()
        else:
            original_obs = obs

        for i in range(n_envs):
            episode_data[i]['observations'].append(original_obs[i].copy())

        action, _ = model.predict(obs, deterministic=deterministic)

        step_result = env.step(action)
        next_obs = step_result[0]
        rewards = step_result[1]
        dones = step_result[2]

        if use_vec_normalize:
            original_next_obs = env.get_original_obs()
        else:
            original_next_obs = next_obs

        for i in range(n_envs):
            if episode_data[i]['observations']:
                episode_data[i]['actions'].append(action[i].copy() if isinstance(action[i], np.ndarray) else action[i])
                episode_data[i]['rewards'].append(float(rewards[i]))
                episode_data[i]['dones'].append(bool(dones[i]))
                episode_data[i]['next_observations'].append(original_next_obs[i].copy())

                if dones[i] and len(completed_episodes) < n_episodes:
                    completed_episodes.append({
                        'observations': np.array(episode_data[i]['observations']),
                        'actions': np.array(episode_data[i]['actions']),
                        'rewards': np.array(episode_data[i]['rewards']),
                        'dones': np.array(episode_data[i]['dones']),
                        'next_observations': np.array(episode_data[i]['next_observations']),
                    })

                    episode_data[i] = {
                        'observations': [],
                        'actions': [],
                        'rewards': [],
                        'dones': [],
                        'next_observations': []
                    }

        obs = next_obs

    return completed_episodes[:n_episodes]


def collect_batch(env, model, n_episodes: int, deterministic: bool = False,
                  batch_seed: int = None, use_vec_normalize: bool = False) -> Dict[str, List[np.ndarray]]:
    """Collect a batch of episodes using parallel environments.

    Args:
        env: VecEnv (SubprocVecEnv, optionally wrapped with VecNormalize)
        model: Trained SB3 model
        n_episodes: Number of episodes to collect
        deterministic: Whether to use deterministic actions
        batch_seed: Random seed for this batch
        use_vec_normalize: Whether VecNormalize is used

    Returns:
        Dictionary containing lists of arrays (one per episode):
            - observations: List of (T_i, obs_dim) arrays
            - actions: List of (T_i,) or (T_i, act_dim) arrays
            - rewards: List of (T_i,) arrays
            - dones: List of (T_i,) arrays
            - next_observations: List of (T_i, obs_dim) arrays
            - episode_lengths: (n_episodes,) array of episode lengths
            - episode_returns: (n_episodes,) array of total returns
    """
    if batch_seed is not None:
        np.random.seed(batch_seed)

    episodes = collect_episodes_parallel(env, model, n_episodes, deterministic, use_vec_normalize)

    batch_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'next_observations': [],
        'episode_lengths': [],
        'episode_returns': [],
    }

    for episode in episodes:
        for key in episode:
            batch_data[key].append(episode[key])

        batch_data['episode_lengths'].append(len(episode['rewards']))
        batch_data['episode_returns'].append(episode['rewards'].sum())

    for key in batch_data.keys():
        if key not in ['episode_lengths', 'episode_returns']:
            batch_data[key] = np.array(batch_data[key], dtype=object)
    batch_data['episode_lengths'] = np.array(batch_data['episode_lengths'])
    batch_data['episode_returns'] = np.array(batch_data['episode_returns'])
    return batch_data


def save_and_log_batch(batch_data: dict, output_path: Path, batch_name: str) -> dict:
    """Save batch data and return statistics.

    Args:
        batch_data: Dictionary containing batch data
        output_path: Path to save the NPZ file
        batch_name: Name for logging

    Returns:
        Dictionary of statistics
    """
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


def generate_data(config: ExperimentConfig, policy_path: Path, output_dir: Path):
    """Generate n batches of k episodes using trained policy.

    Args:
        config: Experiment configuration
        policy_path: Path to trained policy (.zip file)
        output_dir: Directory to save generated data
    """
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

    vec_normalize_path = policy_path.parent / "vec_normalize.pkl"
    env, use_vec_normalize = create_vec_env(
        config,
        n_envs=config.data_generation.n_envs,
        use_monitor=False,
        vec_normalize_path=vec_normalize_path
    )

    if use_vec_normalize:
        print(f"Loaded VecNormalize stats from {vec_normalize_path}")

    print(f"\nGenerating data using policy from {policy_path}")
    print(f"Environment: {config.environment.name}")
    print(f"VecNormalize: {use_vec_normalize}")
    print(f"Deterministic policy: {config.data_generation.deterministic_policy}")
    print(f"Output directory: {output_dir}\n")

    all_batch_stats = []
    current_seed = config.seed


    batches_to_generate = []

    # Regular batches
    for i in range(config.data_generation.n_batches):
        batches_to_generate.append((f"batch_{i}", config.data_generation.episodes_per_batch))

    # Special batches
    if config.data_generation.tuning_episodes > 0:
        batches_to_generate.append(("batch_tuning", config.data_generation.tuning_episodes))
    if config.data_generation.ground_truth_episodes > 0:
        batches_to_generate.append(("batch_ground_truth", config.data_generation.ground_truth_episodes))
    if config.data_generation.eval_episodes > 0:
        batches_to_generate.append(("batch_eval", config.data_generation.eval_episodes))

    # Generate all batches
    for batch_name, n_episodes in batches_to_generate:
        print(f"Collecting {batch_name} ({n_episodes} episodes)")

        batch_data = collect_batch(
            env=env,
            model=model,
            n_episodes=n_episodes,
            deterministic=config.data_generation.deterministic_policy,
            batch_seed=current_seed,
            use_vec_normalize=use_vec_normalize
        )

        stats = save_and_log_batch(batch_data, output_dir / f"{batch_name}.npz", batch_name)
        stats['batch_seed'] = current_seed
        all_batch_stats.append(stats)

        current_seed += 1

    # Save overall statistics
    total_episodes = sum(n_eps for _, n_eps in batches_to_generate)

    stats_file = output_dir / "data_statistics.npz"
    np.savez(
        stats_file,
        batch_stats=all_batch_stats,
        config_seed=config.seed,
        total_episodes=total_episodes,
    )

    # Save data generation metadata
    data_metadata = {
        'policy_path': str(policy_path),
        'deterministic_policy': config.data_generation.deterministic_policy,
        'total_episodes': int(total_episodes),
        'n_batches': config.data_generation.n_batches,
        'episodes_per_batch': config.data_generation.episodes_per_batch,
        'tuning_episodes': config.data_generation.tuning_episodes,
        'ground_truth_episodes': config.data_generation.ground_truth_episodes,
        'eval_episodes': config.data_generation.eval_episodes,
        'seed': config.seed,
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
    args = parser.parse_args()

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)

    # Set default paths
    if args.policy_path is None:
        policy_path = Path("experiments") / config.experiment_id / "policy" / "policy_final.zip"
    else:
        policy_path = args.policy_path

    if args.output_dir is None:
        output_dir = Path("experiments") / config.experiment_id / "data"
    else:
        output_dir = args.output_dir

    # Save config to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Generate data
    generate_data(config, policy_path, output_dir)


if __name__ == "__main__":
    main()
