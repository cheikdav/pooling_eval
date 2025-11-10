"""Generate episode data using a trained policy."""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, A2C, SAC, TD3
import gymnasium as gym
from tqdm import tqdm
from typing import List, Dict, Any

from src.config import ExperimentConfig


ALGORITHM_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


def collect_episode(env, model, deterministic: bool = False) -> Dict[str, np.ndarray]:
    """Collect a single episode of data.

    Args:
        env: Gymnasium environment
        model: Trained SB3 model
        deterministic: Whether to use deterministic actions

    Returns:
        Dictionary containing episode data:
            - observations: (T, obs_dim) array
            - actions: (T,) or (T, act_dim) array
            - rewards: (T,) array
            - dones: (T,) array (terminal flags)
            - next_observations: (T, obs_dim) array
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []

    obs, _ = env.reset()
    done = False

    while not done:
        observations.append(obs)

        # Get action from policy
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        next_observations.append(next_obs)

        obs = next_obs

    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'next_observations': np.array(next_observations),
    }


def collect_batch(env, model, n_episodes: int, deterministic: bool = False,
                  batch_seed: int = None) -> Dict[str, List[np.ndarray]]:
    """Collect a batch of episodes.

    Args:
        env: Gymnasium environment
        model: Trained SB3 model
        n_episodes: Number of episodes to collect
        deterministic: Whether to use deterministic actions
        batch_seed: Random seed for this batch

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
        env.reset(seed=batch_seed)
        np.random.seed(batch_seed)

    batch_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'next_observations': [],
        'episode_lengths': [],
        'episode_returns': [],
    }

    for _ in tqdm(range(n_episodes), desc="Collecting episodes", leave=False):
        episode = collect_episode(env, model, deterministic)

        for key in episode:
            batch_data[key].append(np.array(episode[key]))
        

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

    # Load trained policy
    if config.policy.algorithm not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {config.policy.algorithm}")

    AlgorithmClass = ALGORITHM_MAP[config.policy.algorithm]
    model = AlgorithmClass.load(policy_path)

    # Create environment
    env = gym.make(config.environment.name)

    print(f"\nGenerating data using policy from {policy_path}")
    print(f"Environment: {config.environment.name}")
    print(f"Deterministic policy: {config.data_generation.deterministic_policy}")
    print(f"Output directory: {output_dir}\n")

    all_batch_stats = []
    current_seed = config.seed

    # Define all batches to generate
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
            batch_seed=current_seed
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
