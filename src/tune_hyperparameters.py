"""Hyperparameter tuning wrapper for W&B sweeps.

Simple wrapper that modifies learning rate and calls core training logic.
Uses batch_tuning.npz for all hyperparameter search.
"""

import argparse
from pathlib import Path
import wandb

from src.config import ExperimentConfig
from src.train_estimator import train_estimator


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with W&B sweeps")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", type=str, required=True,
                       choices=['monte_carlo', 'td_lambda', 'dqn'])
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    # Initialize wandb (will pick up sweep config automatically)
    wandb.init()

    # Load base config
    config = ExperimentConfig.from_yaml(args.config)

    # Override learning rate from wandb sweep or CLI
    learning_rate = wandb.config.get('learning_rate', args.learning_rate)

    if learning_rate is not None:
        if args.method == 'monte_carlo':
            config.value_estimators.monte_carlo.learning_rate = learning_rate
        elif args.method == 'td_lambda':
            config.value_estimators.td_lambda.learning_rate = learning_rate
        elif args.method == 'dqn':
            config.value_estimators.dqn.learning_rate = learning_rate

    # Setup paths
    batch_path = Path("experiments") / config.experiment_id / "data" / "batch_tuning.npz"
    output_dir = Path("experiments") / config.experiment_id / "sweeps" / args.method / wandb.run.id

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Call core training
    train_estimator(
        config=config,
        method=args.method,
        batch_path=batch_path,
        output_dir=output_dir,
        use_wandb=True
    )


if __name__ == "__main__":
    main()
