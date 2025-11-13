"""Train a single value estimator on a batch of data."""

import argparse
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import wandb
import json

from src.config import ExperimentConfig, BaseEstimatorConfig
from src.estimators import ESTIMATOR_REGISTRY
from src.data_preprocessing import preprocess_episodes


def create_estimator(method_config: BaseEstimatorConfig, network_config, obs_dim: int, gamma: float):
    """Create an estimator instance from configuration using registry.

    Args:
        method_config: Method-specific configuration
        network_config: Network configuration
        obs_dim: Observation dimension
        gamma: Discount factor (from training config)

    Returns:
        Estimator instance
    """
    # Look up estimator class from registry based on config type
    EstimatorClass = ESTIMATOR_REGISTRY[type(method_config)]

    # Use the from_config classmethod to create the estimator
    return EstimatorClass.from_config(method_config, network_config, obs_dim, gamma)


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




def train_single_initialization(
    estimator,
    batch: dict,
    config: ExperimentConfig,
    method_name: str,
    batch_name: str,
    init_idx: int,
    use_wandb: bool,
    sweep_mode: bool = False
):
    """Train a single initialization and return its final loss.

    Args:
        estimator: Estimator instance to train
        batch: Batch data
        config: Experiment configuration
        method_name: Method name (from method_config.name)
        batch_name: Batch name for logging
        init_idx: Initialization index
        use_wandb: Whether to use wandb
        sweep_mode: If True, skip wandb.init() (run already initialized by sweep)

    Returns:
        Final loss value
    """
    # Initialize wandb for this initialization (unless in sweep mode)
    if use_wandb and config.logging.use_wandb and not sweep_mode:
        run_name = f"{method_name}_{batch_name}_init{init_idx}"
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=run_name,
            group=config.experiment_id,
            config={
                'method': method_name,
                'batch_name': batch_name,
                'init_idx': init_idx,
                'experiment_id': config.experiment_id,
                **estimator.get_config(),
            },
            tags=[method_name, batch_name, f"init{init_idx}"],
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
                'train/mse': metrics['loss'],
                'train/mae': metrics['mae'],
                'train/mean_value': metrics['mean_value'],
                'train/mean_target': metrics['mean_target'],
                'train/best_loss': best_loss,
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
    method_config: BaseEstimatorConfig,
    batch_path: Path,
    output_dir: Path,
    use_wandb: bool = True,
    save_model: bool = True,
    sweep_mode: bool = False
):
    """Train multiple estimator initializations and keep the best one.

    Args:
        config: Experiment configuration
        method_config: Method-specific configuration
        batch_path: Path to batch data file
        output_dir: Directory to save outputs
        use_wandb: Whether to use wandb logging
        save_model: Whether to save the trained model (default: True)
        sweep_mode: If True, skip wandb.init() (for W&B sweeps where init already called)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed (only if we're saving models)
    if save_model:
        checkpoint_path = output_dir / "estimator_final.pt"
        if checkpoint_path.exists():
            print(f"Training already completed for {method_config.name} at {output_dir}. Skipping.")
            return

    # Load batch data
    print(f"Loading batch data from {batch_path}")
    batch = load_batch_data(batch_path)

    # Preprocess batch: flatten episodes and compute MC returns
    gamma = config.value_estimators.training.gamma
    print(f"Preprocessing batch (flattening episodes, computing MC returns with gamma={gamma})")
    batch = preprocess_episodes(batch, gamma)

    # Get observation dimension and number of initializations
    obs_dim = batch['observations'].shape[-1]
    n_inits = method_config.n_initializations
    batch_name = batch_path.stem
    method_name = method_config.name

    print(f"\nTraining {n_inits} initialization(s) of {method_name} on {batch_name}")
    print(f"Max epochs: {config.value_estimators.training.max_epochs}")
    if not save_model:
        print("Model saving disabled (tuning mode)")
    print()

    # Train multiple initializations
    best_loss = float('inf')
    best_estimator = None

    for init_idx in range(n_inits):
        # Create fresh estimator with different random seed
        torch.manual_seed(config.seed + init_idx)
        np.random.seed(config.seed + init_idx)
        estimator = create_estimator(method_config, config.network, obs_dim, gamma)

        # Train this initialization
        final_loss = train_single_initialization(
            estimator, batch, config, method_name, batch_name, init_idx, use_wandb, sweep_mode
        )

        print(f"Init {init_idx}: final loss = {final_loss:.6f}")

        # Keep track of best estimator
        if final_loss < best_loss:
            best_loss = final_loss
            best_estimator = estimator
            print(f"  -> New best!")

    # Save best model only if requested
    if save_model:
        checkpoint_path = output_dir / "estimator_final.pt"
        print(f"\nSaving best estimator (loss={best_loss:.6f}) to {checkpoint_path}")
        best_estimator.save(checkpoint_path)

        # Save training statistics
        stats = {
            'method': method_name,
            'batch_name': batch_name,
            'n_initializations': n_inits,
            'best_loss': best_loss,
        }

        with open(output_dir / "training_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    else:
        print(f"\nSkipping model save (best loss: {best_loss:.6f})")

    print(f"Training complete for {method_name} {batch_name}")


def main():
    parser = argparse.ArgumentParser(description="Train a value estimator")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument("--method", type=str, required=True,
                       help="Method name (corresponds to 'name' field in method config)")
    parser.add_argument("--batch-idx", type=int, required=False,
                       help="Batch index to train on (if not set, uses SGE_TASK_ID-1 from environment)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    args = parser.parse_args()

    # Determine batch index: use argument if provided, otherwise check SGE_TASK_ID
    if args.batch_idx is not None:
        batch_idx = args.batch_idx
    else:
        import os
        sge_task_id = os.getenv('SGE_TASK_ID')
        if sge_task_id is None:
            parser.error("--batch-idx is required when not running as SGE array job")
        batch_idx = int(sge_task_id) - 1  # SGE_TASK_ID starts at 1, batch_idx starts at 0
        print(f"Using SGE_TASK_ID={sge_task_id} -> batch_idx={batch_idx}")

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)

    # Find method config by name
    method_config = None
    for mc in config.value_estimators.method_configs:
        if mc.name == args.method:
            method_config = mc
            break

    if method_config is None:
        available_methods = [mc.name for mc in config.value_estimators.method_configs]
        parser.error(f"Method '{args.method}' not found in config. Available methods: {available_methods}")

    # Set paths based on batch index
    batch_path = Path("experiments") / config.experiment_id / "data" / f"batch_{batch_idx}.npz"
    output_dir = Path("experiments") / config.experiment_id / "estimators" / args.method / f"batch_{batch_idx}"

    # Save config to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Train estimator
    train_estimator(
        config=config,
        method_config=method_config,
        batch_path=batch_path,
        output_dir=output_dir,
        use_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()
