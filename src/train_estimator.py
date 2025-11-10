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
    device = config.network.device

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
            device=device,
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
            device=device,
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
            device=device,
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


def get_n_initializations(config: ExperimentConfig, method: str) -> int:
    """Get number of initializations for a method.

    Args:
        config: Experiment configuration
        method: Estimator method name

    Returns:
        Number of initializations to train
    """
    if method == 'monte_carlo':
        return config.value_estimators.monte_carlo.n_initializations
    elif method == 'td_lambda':
        return config.value_estimators.td_lambda.n_initializations
    elif method == 'dqn':
        return config.value_estimators.dqn.n_initializations
    return 1


def train_single_initialization(
    estimator,
    batch: dict,
    config: ExperimentConfig,
    method: str,
    batch_name: str,
    init_idx: int,
    use_wandb: bool
):
    """Train a single initialization and return its final loss.

    Args:
        estimator: Estimator instance to train
        batch: Batch data
        config: Experiment configuration
        method: Method name
        batch_name: Batch name for logging
        init_idx: Initialization index
        use_wandb: Whether to use wandb

    Returns:
        Final loss value
    """
    # Initialize wandb for this initialization
    if use_wandb and config.logging.use_wandb:
        run_name = f"{method}_{batch_name}_init{init_idx}"
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=run_name,
            group=config.experiment_id,
            config={
                'method': method,
                'batch_name': batch_name,
                'init_idx': init_idx,
                'experiment_id': config.experiment_id,
                **estimator.get_config(),
            },
            tags=[method, batch_name, f"init{init_idx}"],
        )

    training_config = config.value_estimators.training
    loss_history = []
    best_loss = float('inf')

    for epoch in tqdm(range(training_config.max_epochs), desc=f"Init {init_idx}", leave=False):
        metrics = estimator.train_step(batch)
        loss_history.append(metrics['loss'])

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']

        # Logging
        if epoch % config.logging.log_frequency == 0 and use_wandb and config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': metrics['loss'],
                'train/mean_value': metrics['mean_value'],
                'train/mean_target': metrics['mean_target'],
                'train/best_loss': best_loss,
            })

        # Evaluation logging
        if epoch % training_config.eval_frequency == 0 and use_wandb and config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'eval/mse': metrics['loss'],
                'eval/mae': metrics['mae'],
            })

        # Check convergence
        if check_convergence(loss_history, training_config.convergence_patience,
                           training_config.convergence_threshold):
            break

    final_loss = loss_history[-1] if loss_history else float('inf')

    if use_wandb and config.logging.use_wandb:
        wandb.log({'final/best_loss': best_loss})
        wandb.finish()

    return final_loss


def train_estimator(
    config: ExperimentConfig,
    method: str,
    batch_path: Path,
    output_dir: Path,
    use_wandb: bool = True
):
    """Train multiple estimator initializations and keep the best one.

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

    # Get observation dimension and number of initializations
    obs_dim = batch['observations'][0].shape[-1]
    n_inits = get_n_initializations(config, method)
    batch_name = batch_path.stem

    print(f"\nTraining {n_inits} initialization(s) of {method} on {batch_name}")
    print(f"Max epochs: {config.value_estimators.training.max_epochs}\n")

    # Train multiple initializations
    best_loss = float('inf')
    best_estimator = None

    for init_idx in range(n_inits):
        # Create fresh estimator with different random seed
        torch.manual_seed(config.seed + init_idx)
        np.random.seed(config.seed + init_idx)
        estimator = create_estimator(method, config, obs_dim)

        # Train this initialization
        final_loss = train_single_initialization(
            estimator, batch, config, method, batch_name, init_idx, use_wandb
        )

        print(f"Init {init_idx}: final loss = {final_loss:.6f}")

        # Keep track of best estimator
        if final_loss < best_loss:
            best_loss = final_loss
            best_estimator = estimator
            print(f"  -> New best!")

    # Save best model
    print(f"\nSaving best estimator (loss={best_loss:.6f}) to {checkpoint_path}")
    best_estimator.save(checkpoint_path)

    # Save training statistics
    stats = {
        'method': method,
        'batch_name': batch_name,
        'n_initializations': n_inits,
        'best_loss': best_loss,
    }

    with open(output_dir / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

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
