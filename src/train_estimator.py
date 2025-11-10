"""Train a single value estimator on a batch of data."""

import argparse
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import wandb
import json

from src.config import ExperimentConfig
from src.estimators import MonteCarloEstimator, TDLambdaEstimator, DQNEstimator


ESTIMATOR_MAP = {
    'monte_carlo': MonteCarloEstimator,
    'td_lambda': TDLambdaEstimator,
    'dqn': DQNEstimator,
}


def create_estimator(method: str, config: ExperimentConfig, obs_dim: int):
    """Create an estimator instance based on method name.

    Args:
        method: Estimator method name
        config: Experiment configuration
        obs_dim: Observation dimension

    Returns:
        Estimator instance
    """
    if method not in ESTIMATOR_MAP:
        raise ValueError(f"Unknown method: {method}. Available: {list(ESTIMATOR_MAP.keys())}")

    EstimatorClass = ESTIMATOR_MAP[method]
    hidden_sizes = config.network.hidden_sizes
    activation = config.network.activation

    # Get method-specific config
    if method == 'monte_carlo':
        method_config = config.value_estimators.monte_carlo
        if method_config is None:
            raise ValueError(
                f"Configuration for '{method}' is missing in the config file. "
                f"Please add a 'monte_carlo' section under 'value_estimators' "
                f"or remove '{method}' from the 'methods' list."
            )
        return EstimatorClass(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            discount_factor=method_config.discount_factor,
            activation=activation,
            learning_rate=method_config.learning_rate,
        )
    elif method == 'td_lambda':
        method_config = config.value_estimators.td_lambda
        if method_config is None:
            raise ValueError(
                f"Configuration for '{method}' is missing in the config file. "
                f"Please add a 'td_lambda' section under 'value_estimators' "
                f"or remove '{method}' from the 'methods' list."
            )
        return EstimatorClass(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            discount_factor=method_config.discount_factor,
            lambda_=method_config.lambda_,
            n_step=method_config.n_step,
            activation=activation,
            learning_rate=method_config.learning_rate,
        )
    elif method == 'dqn':
        method_config = config.value_estimators.dqn
        if method_config is None:
            raise ValueError(
                f"Configuration for '{method}' is missing in the config file. "
                f"Please add a 'dqn' section under 'value_estimators' "
                f"or remove '{method}' from the 'methods' list."
            )
        return EstimatorClass(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            discount_factor=method_config.discount_factor,
            target_update_rate=method_config.target_update_rate,
            double_dqn=method_config.double_dqn,
            activation=activation,
            learning_rate=method_config.learning_rate,
        )


def load_batch_data(batch_path: Path) -> dict:
    """Load batch data from NPZ file.

    Args:
        batch_path: Path to batch NPZ file

    Returns:
        Dictionary containing batch data
    """
    data = np.load(batch_path, allow_pickle=True)

    # Convert to dictionary and handle numpy arrays
    batch = {}
    for key in data.keys():
        if key in ['observations', 'actions', 'rewards', 'dones', 'next_observations']:
            # These are lists of arrays (one per episode)
            batch[key] = [arr for arr in data[key]]
        else:
            batch[key] = data[key]

    return batch


def check_convergence(loss_history: list, patience: int, threshold: float) -> bool:
    """Check if training has converged.

    Args:
        loss_history: List of recent losses
        patience: Number of epochs to check
        threshold: Relative improvement threshold

    Returns:
        True if converged, False otherwise
    """
    if len(loss_history) < patience:
        return False

    recent_losses = loss_history[-patience:]
    if len(recent_losses) < 2:
        return False

    # Check if improvement is below threshold
    initial_loss = recent_losses[0]
    final_loss = recent_losses[-1]

    if initial_loss == 0:
        return False

    relative_improvement = abs((final_loss - initial_loss) / initial_loss)
    return relative_improvement < threshold


def train_estimator(
    config: ExperimentConfig,
    method: str,
    batch_path: Path,
    output_dir: Path,
    use_wandb: bool = True
):
    """Train a value estimator on a batch of data.

    Args:
        config: Experiment configuration
        method: Estimator method name
        batch_path: Path to batch data file
        output_dir: Directory to save outputs
        use_wandb: Whether to use wandb logging
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    checkpoint_path = output_dir / "final_checkpoint.pt"
    if checkpoint_path.exists():
        print(f"Training already completed for {method} at {output_dir}. Skipping.")
        return

    # Load batch data
    print(f"Loading batch data from {batch_path}")
    batch = load_batch_data(batch_path)

    # Get observation dimension
    obs_dim = batch['observations'][0].shape[-1]

    # Create estimator
    print(f"Creating {method} estimator")
    estimator = create_estimator(method, config, obs_dim)

    # Initialize wandb
    batch_name = batch_path.stem  # e.g., "batch_0" or "batch_tuning"
    if use_wandb and config.logging.use_wandb:
        run_name = f"{method}_{batch_name}"
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=run_name,
            group=config.experiment_id,
            config={
                'method': method,
                'batch_name': batch_name,
                'experiment_id': config.experiment_id,
                **estimator.get_config(),
            },
            tags=[method, batch_name],
        )

    # Training loop
    print(f"\nTraining {method} estimator on {batch_name}")
    print(f"Max epochs: {config.value_estimators.training.max_epochs}")
    print(f"Batch size: {config.value_estimators.training.batch_size}\n")

    training_config = config.value_estimators.training
    loss_history = []
    best_loss = float('inf')

    for epoch in tqdm(range(training_config.max_epochs), desc="Training"):
        # Perform training step (full batch for now, can add mini-batching later)
        metrics = estimator.train_step(batch)

        loss_history.append(metrics['loss'])
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']

        # Logging
        if epoch % config.logging.log_frequency == 0:
            log_dict = {
                'epoch': epoch,
                'train/loss': metrics['loss'],
                'train/mean_value': metrics['mean_value'],
                'train/mean_target': metrics['mean_target'],
                'train/best_loss': best_loss,
            }

            if use_wandb and config.logging.use_wandb:
                wandb.log(log_dict)

        # Evaluation
        if epoch % training_config.eval_frequency == 0:
            eval_log_dict = {
                'epoch': epoch,
                'eval/mse': metrics['loss'],
                'eval/mae': metrics['mae'],
            }

            if use_wandb and config.logging.use_wandb:
                wandb.log(eval_log_dict)

        # Check convergence
        if check_convergence(loss_history, training_config.convergence_patience,
                           training_config.convergence_threshold):
            print(f"\nConverged at epoch {epoch}")
            break

        # Save checkpoint periodically
        if epoch % 100 == 0 and epoch > 0:
            checkpoint = output_dir / f"checkpoint_epoch{epoch}.pt"
            estimator.save(checkpoint)

    # Save final model
    print(f"\nSaving final checkpoint to {checkpoint_path}")
    estimator.save(checkpoint_path)

    # Save training statistics
    stats = {
        'method': method,
        'batch_name': batch_name,
        'final_epoch': epoch,
        'best_loss': best_loss,
        'final_loss': loss_history[-1] if loss_history else None,
        'converged': epoch < training_config.max_epochs - 1,
    }

    with open(output_dir / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    if use_wandb and config.logging.use_wandb:
        wandb.log({'final/best_loss': best_loss, 'final/converged': stats['converged']})
        wandb.finish()

    print(f"Training complete for {method} {batch_name}")


def main():
    parser = argparse.ArgumentParser(description="Train a value estimator")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument("--method", type=str, required=True,
                       choices=list(ESTIMATOR_MAP.keys()),
                       help="Estimator method")
    parser.add_argument("--batch-idx", type=int, required=True,
                       help="Batch index to train on")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    args = parser.parse_args()

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)

    # Set paths based on batch index
    batch_path = Path("experiments") / config.experiment_id / "data" / f"batch_{args.batch_idx}.npz"
    output_dir = Path("experiments") / config.experiment_id / "estimators" / args.method / f"batch_{args.batch_idx}"

    # Save config to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Train estimator
    train_estimator(
        config=config,
        method=args.method,
        batch_path=batch_path,
        output_dir=output_dir,
        use_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()
