"""Utility to estimate storage requirements for batch generation."""

import argparse
import os
import tempfile
import numpy as np
from pathlib import Path

from src.config import ExperimentConfig
from src.generate_data import collect_episodes_parallel
from src.env_utils import ALGORITHM_MAP, create_vec_env


def estimate_storage(config_path: str, target_gb: float, n_batches: int = 80, sample_episodes: int = 30):
    """
    Estimate episodes per batch needed to reach target storage size.

    Args:
        config_path: Path to config file
        target_gb: Target total storage in GB
        n_batches: Number of batches (default: 80)
        sample_episodes: Number of episodes to generate for estimation (default: 30)
    """
    # Load config
    config = ExperimentConfig.from_yaml(config_path)
    experiment_dir = Path("experiments") / config.experiment_id
    policy_path = experiment_dir / "policy/policy_final.zip"

    print(f"Loading policy from {policy_path}")

    # Load policy using SB3
    if config.policy.algorithm not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {config.policy.algorithm}")

    AlgorithmClass = ALGORITHM_MAP[config.policy.algorithm]
    model = AlgorithmClass.load(policy_path)

    # Create environment
    vec_normalize_path = policy_path.parent / "vec_normalize.pkl"
    env, use_vec_normalize = create_vec_env(
        config,
        n_envs=1,
        use_monitor=False,
        vec_normalize_path=vec_normalize_path
    )

    # Generate sample episodes
    print(f"Generating {sample_episodes} sample episodes...")
    episodes_list = collect_episodes_parallel(
        env, model, sample_episodes,
        deterministic=config.data_generation.deterministic_policy,
        use_vec_normalize=use_vec_normalize
    )

    # Convert to batch format
    episodes_data = {
        'observations': [ep['observations'] for ep in episodes_list],
        'actions': [ep['actions'] for ep in episodes_list],
        'rewards': [ep['rewards'] for ep in episodes_list],
        'dones': [ep['dones'] for ep in episodes_list],
        'next_observations': [ep['next_observations'] for ep in episodes_list],
        'episode_lengths': np.array([len(ep['rewards']) for ep in episodes_list]),
        'episode_returns': np.array([ep['rewards'].sum() for ep in episodes_list])
    }

    env.close()

    # Save to temporary file to measure size
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        np.savez_compressed(
            tmp_path,
            observations=np.array(episodes_data['observations'], dtype=object),
            actions=np.array(episodes_data['actions'], dtype=object),
            rewards=np.array(episodes_data['rewards'], dtype=object),
            dones=np.array(episodes_data['dones'], dtype=object),
            next_observations=np.array(episodes_data['next_observations'], dtype=object),
            episode_lengths=episodes_data['episode_lengths'],
            episode_returns=episodes_data['episode_returns']
        )

    # Measure file size
    file_size_bytes = os.path.getsize(tmp_path)
    file_size_mb = file_size_bytes / (1024 ** 2)
    os.unlink(tmp_path)  # Clean up temp file

    # Calculate estimates
    size_per_episode_mb = file_size_mb / sample_episodes
    target_bytes = target_gb * (1024 ** 3)
    size_per_batch_bytes = target_bytes / n_batches
    episodes_per_batch = size_per_batch_bytes / (size_per_episode_mb * 1024 ** 2)

    # Calculate actual storage with recommended episodes per batch
    recommended_episodes = int(episodes_per_batch)
    actual_total_gb = (recommended_episodes * size_per_episode_mb * n_batches) / 1024

    # Print results
    print("\n" + "=" * 70)
    print("STORAGE ESTIMATION RESULTS")
    print("=" * 70)
    print(f"\nSample size: {sample_episodes} episodes")
    print(f"Sample file size: {file_size_mb:.2f} MB")
    print(f"Size per episode: {size_per_episode_mb:.3f} MB")
    print(f"\nTarget configuration:")
    print(f"  Total storage: {target_gb:.1f} GB")
    print(f"  Number of batches: {n_batches}")
    print(f"\nRecommendation:")
    print(f"  Episodes per batch: {recommended_episodes}")
    print(f"  Actual total storage: {actual_total_gb:.2f} GB")
    print(f"  Storage per batch: {actual_total_gb / n_batches * 1024:.2f} MB")
    print("=" * 70)

    return recommended_episodes


def main():
    parser = argparse.ArgumentParser(
        description="Estimate episodes per batch for target storage size"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--target-gb",
        type=float,
        required=True,
        help="Target total storage in GB"
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=80,
        help="Number of batches (default: 80)"
    )
    parser.add_argument(
        "--sample-episodes",
        type=int,
        default=30,
        help="Number of episodes to generate for estimation (default: 30)"
    )

    args = parser.parse_args()

    estimate_storage(
        config_path=args.config,
        target_gb=args.target_gb,
        n_batches=args.n_batches,
        sample_episodes=args.sample_episodes
    )


if __name__ == "__main__":
    main()
