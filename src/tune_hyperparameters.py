"""Hyperparameter tuning wrapper for W&B sweeps.

Simple wrapper that modifies learning rate and calls core training logic.
Uses batch_tuning.npz for all hyperparameter search.
"""

import argparse
from pathlib import Path
import sys
import wandb

from src.config import ExperimentConfig
from src.train_estimator import train_estimator


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with W&B sweeps")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", type=str, required=True,
                       help="Method name (corresponds to 'name' field in method config)")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-update-rate", type=float, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    # Initialize wandb (will pick up sweep config automatically)
    wandb.init(tags=["hyperparameter-tuning", "sweep"])

    # Load config early to get paths
    config_temp = ExperimentConfig.from_yaml(args.config)

    # Redirect stdout/stderr to log file
    log_dir = config_temp.get_estimators_dir() / "sweeps" / args.method / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{wandb.run.id}.log"

    sys.stdout = open(log_file, 'w', buffering=1)
    sys.stderr = sys.stdout

    print(f"Sweep agent {wandb.run.id} starting - logging to {log_file}")

    # Load base config
    config = ExperimentConfig.from_yaml(args.config)

    # Find method config by name
    method_config = None
    for mc in config.value_estimators.method_configs:
        if mc.name == args.method:
            method_config = mc
            break

    if method_config is None:
        available_methods = [mc.name for mc in config.value_estimators.method_configs]
        raise ValueError(f"Method '{args.method}' not found in config. Available methods: {available_methods}")

    # Override hyperparameters from wandb sweep or CLI
    learning_rate = wandb.config.get('learning_rate', args.learning_rate)
    if learning_rate is not None:
        method_config.learning_rate = learning_rate

    # Override target_update_rate for DQN from wandb sweep or CLI
    target_update_rate = wandb.config.get('target_update_rate', args.target_update_rate)
    if target_update_rate is not None and hasattr(method_config, 'target_update_rate'):
        method_config.target_update_rate = target_update_rate

    # Override num_episodes from wandb sweep or CLI
    num_episodes = wandb.config.get('num_episodes', args.num_episodes)
    if num_episodes is not None:
        config.value_estimators.training.episode_subsets = [num_episodes]

    # Override batch_size from wandb sweep or CLI
    batch_size = wandb.config.get('batch_size', args.batch_size)
    if batch_size is not None:
        config.value_estimators.training.batch_size = batch_size

    # Force single initialization for tuning
    method_config.n_initializations = 1

    # Setup paths
    batch_path = config.get_data_dir() / "batch_tuning.npz"
    output_dir = config.get_estimators_dir() / "sweeps" / args.method / wandb.run.id

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Call core training
    train_estimator(
        config=config,
        method_config=method_config,
        batch_path=batch_path,
        output_dir=output_dir,
        batch_name="tuning",
        overwrite=True,
        use_wandb=True,
        sweep_mode=True
    )


if __name__ == "__main__":
    main()
