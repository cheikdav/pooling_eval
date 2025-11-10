"""Train a policy using Stable Baselines 3."""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import torch

from src.config import ExperimentConfig


ALGORITHM_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


def train_policy(config: ExperimentConfig, output_dir: Path):
    """Train a policy using Stable Baselines 3.

    Args:
        config: Experiment configuration
        output_dir: Directory to save the trained policy
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create environment
    env = gym.make(config.environment.name)
    env.reset(seed=config.seed)

    # Get algorithm class
    if config.policy.algorithm not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {config.policy.algorithm}. "
                        f"Available: {list(ALGORITHM_MAP.keys())}")

    AlgorithmClass = ALGORITHM_MAP[config.policy.algorithm]

    # Prepare algorithm kwargs
    algo_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": config.policy.learning_rate,
        "gamma": config.policy.gamma,
        "verbose": 1,
        "seed": config.seed,
    }

    # Add algorithm-specific parameters
    if config.policy.algorithm in ["PPO", "A2C"]:
        algo_kwargs.update({
            "n_steps": config.policy.n_steps,
            "batch_size": config.policy.batch_size,
            "n_epochs": config.policy.n_epochs,
            "gae_lambda": config.policy.gae_lambda,
        })

    # Create model
    model = AlgorithmClass(**algo_kwargs)

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.policy.total_timesteps // 10, 1000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="policy"
    )

    print(f"\nTraining {config.policy.algorithm} on {config.environment.name}...")
    print(f"Total timesteps: {config.policy.total_timesteps}")
    print(f"Output directory: {output_dir}\n")

    # Train the model
    model.learn(
        total_timesteps=config.policy.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # Save final model
    model_path = output_dir / "policy_final.zip"
    model.save(model_path)
    print(f"\nPolicy saved to {model_path}")

    # Save the critic network architecture info for later use
    try:
        if hasattr(model.policy, 'value_net'):
            critic_info = {
                'hidden_sizes': [layer.out_features for layer in model.policy.value_net
                            if hasattr(layer, 'out_features')],
            }
        elif hasattr(model.policy, 'q_net'):
            critic_info = {
                'hidden_sizes': [layer.out_features for layer in model.policy.q_net
                            if hasattr(layer, 'out_features')],
            }
        else:
            critic_info = {'hidden_sizes': config.network.hidden_sizes}

        np.savez(
            output_dir / "critic_architecture.npz",
            **critic_info
        )
    except Exception as e:
        print(f"Warning: Could not save critic architecture info: {e}")

    env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train a policy using Stable Baselines 3")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (default: experiments/<experiment_id>/policy)")
    args = parser.parse_args()

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)

    # Set output directory
    if args.output_dir is None:
        output_dir = Path("experiments") / config.experiment_id / "policy"
    else:
        output_dir = args.output_dir

    # Save config to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Train policy
    train_policy(config, output_dir)


if __name__ == "__main__":
    main()
