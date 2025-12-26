"""Train a single value estimator on a batch of data."""

import argparse
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import wandb
import json

from src.config import ExperimentConfig, BaseEstimatorConfig, LeastSquaresMCConfig, LeastSquaresTDConfig
from src.estimators import ESTIMATOR_REGISTRY
from src.data_preprocessing import preprocess_episodes, sample_episodes, TransitionDataset
from torch.utils.data import DataLoader


def create_estimator(method_config: BaseEstimatorConfig, network_config, obs_dim: int, gamma: float, experiment_id: str = None):
    """Create an estimator instance from configuration using registry.

    Args:
        method_config: Method-specific configuration
        network_config: Network configuration
        obs_dim: Observation dimension
        gamma: Discount factor (from training config)
        experiment_id: Experiment ID (needed for auto-setting policy_path)

    Returns:
        Estimator instance
    """
    # Auto-set policy_path for LeastSquares methods if not provided
    if isinstance(method_config, (LeastSquaresMCConfig, LeastSquaresTDConfig)) and method_config.policy_path is None:
        if experiment_id is None:
            raise ValueError("experiment_id is required when policy_path is not set in LeastSquares config")
        policy_path = Path("experiments") / experiment_id / "policy" / "policy_final.zip"
        method_config.policy_path = str(policy_path)
        print(f"Auto-set policy_path to: {policy_path}")

    EstimatorClass = ESTIMATOR_REGISTRY[type(method_config)]
    return EstimatorClass.from_config(method_config, network_config, obs_dim, gamma)


def load_batch_data(batch_path: Path, max_episodes: int = None) -> dict:
    """Load batch data from NPZ file.

    Args:
        batch_path: Path to batch NPZ file
        max_episodes: If specified, only load the first N episodes

    Returns:
        Dictionary containing batch data
    """
    data = np.load(batch_path, allow_pickle=True)

    batch = {}
    for key in data.keys():
        if key in ['observations', 'actions', 'rewards', 'dones', 'next_observations']:
            batch[key] = ([arr for arr in data[key]] if max_episodes is None
                         else [arr for arr in data[key]][:max_episodes])
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
    train_batch: dict,
    test_batch: dict,
    config: ExperimentConfig,
    method_name: str,
    batch_name: str,
    num_episodes: int,
    init_idx: int,
    use_wandb: bool,
    sweep_mode: bool = False
):
    """Train a single initialization and return its final MC loss.

    Args:
        estimator: Estimator instance to train
        train_batch: Training batch data (preprocessed)
        test_batch: Test batch data (preprocessed, None if no test set)
        config: Experiment configuration
        method_name: Method name (from method_config.name)
        batch_name: Batch name (e.g., '0', '1', 'tuning', 'eval')
        num_episodes: Number of episodes in batch
        init_idx: Initialization index
        use_wandb: Whether to use wandb
        sweep_mode: If True, skip wandb.init() (run already initialized by sweep)

    Returns:
        Final MC loss value (test set if available, otherwise train set)
    """
    if use_wandb and config.logging.use_wandb and not sweep_mode:
        run_name = f"{method_name}_batch{batch_name}_{num_episodes}ep_init{init_idx}"
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=run_name,
            group=config.experiment_id,
            config={
                'method': method_name,
                'batch_name': batch_name,
                'num_episodes': num_episodes,
                'init_idx': init_idx,
                'experiment_id': config.experiment_id,
                **estimator.get_config(),
            },
            tags=[method_name, f"batch_{batch_name}", f"{num_episodes}_episodes", f"init{init_idx}"],
        )

    training_config = config.value_estimators.training
    loss_history = []
    final_mc_loss_train = float('inf')
    final_mc_loss_test = float('inf')
    best_mc_loss = float('inf')
    use_test_set = test_batch is not None
    converged = False
    final_epoch = 0

    # Create dataset once
    dataset = TransitionDataset(train_batch)
    dataloader = None

    # Get method config from estimator
    method_config = None
    for mc in config.value_estimators.method_configs:
        if mc.name == method_name:
            method_config = mc
            break

    # Use method-specific max_epochs if set, otherwise use global
    max_epochs = method_config.max_epochs if (method_config and method_config.max_epochs is not None) else training_config.max_epochs

    for epoch in tqdm(range(max_epochs), desc=f"Init {init_idx}", leave=False):
        final_epoch = epoch
        # Recreate dataloader when needed for shuffling
        if (dataloader is None or
            (training_config.shuffle_frequency > 0 and epoch % training_config.shuffle_frequency == 0)):
            dataloader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True)

        # Accumulate metrics across mini-batches
        epoch_losses = []
        epoch_maes = []
        epoch_mc_losses = []
        epoch_values = []
        epoch_targets = []

        for mini_batch in dataloader:
            metrics = estimator.train_step(mini_batch)
            epoch_losses.append(metrics['loss'])
            epoch_maes.append(metrics['mae'])
            epoch_mc_losses.append(metrics['mc_loss'])
            epoch_values.append(metrics['mean_value'])
            epoch_targets.append(metrics['mean_target'])

        # Average metrics across mini-batches
        avg_metrics = {
            'loss': np.mean(epoch_losses),
            'mae': np.mean(epoch_maes),
            'mc_loss': np.mean(epoch_mc_losses),
            'mean_value': np.mean(epoch_values),
            'mean_target': np.mean(epoch_targets),
        }

        loss_history.append(avg_metrics['loss'])
        final_mc_loss_train = avg_metrics['mc_loss']

        # Compute test MC loss every epoch for model selection
        if use_test_set:
            estimator.eval()
            with torch.no_grad():
                test_mc_returns = torch.FloatTensor(test_batch['mc_returns']).to(estimator.device)
                test_values = estimator.predict(test_batch['observations'])
                test_values = torch.FloatTensor(test_values).to(estimator.device)
                final_mc_loss_test = torch.nn.functional.mse_loss(test_values, test_mc_returns).item()


        current_mc_loss = final_mc_loss_train

        if current_mc_loss < best_mc_loss:
            best_mc_loss = current_mc_loss

        if epoch % config.logging.log_frequency == 0 and use_wandb and config.logging.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': avg_metrics['loss'],
                'train/mse': avg_metrics['loss'],
                'train/mae': avg_metrics['mae'],
                'train/mean_value': avg_metrics['mean_value'],
                'train/mean_target': avg_metrics['mean_target'],
                'train/mc_loss_train': final_mc_loss_train,
                'train/best_mc_loss': best_mc_loss,
            }
            if use_test_set:
                log_dict['test/mc_loss'] = final_mc_loss_test
            wandb.log(log_dict)

        if check_convergence(loss_history, training_config.convergence_patience,
                           training_config.convergence_threshold):
            converged = True
            break

    # Print training summary
    print(f"\n  Init {init_idx} Summary:")
    print(f"    Epochs: {final_epoch + 1}/{max_epochs}")
    if converged:
        print(f"    Stopped: Converged (loss improvement < {training_config.convergence_threshold} for {training_config.convergence_patience} epochs)")
    else:
        print(f"    Stopped: Reached max epochs")
    print(f"    Final train MC loss: {final_mc_loss_train:.6f}")
    if use_test_set:
        print(f"    Final test MC loss: {final_mc_loss_test:.6f}")
    print(f"    Best MC loss: {best_mc_loss:.6f}")

    if use_wandb and config.logging.use_wandb:
        final_log = {'final/best_mc_loss': best_mc_loss}
        if use_test_set:
            final_log['final/mc_loss_train'] = final_mc_loss_train
            final_log['final/mc_loss_test'] = final_mc_loss_test
        wandb.log(final_log)
        wandb.finish()

    return best_mc_loss


def train_estimator(
    config: ExperimentConfig,
    method_config: BaseEstimatorConfig,
    batch_path: Path,
    output_dir: Path,
    batch_name: str,
    overwrite: bool,
    use_wandb: bool = True,
    sweep_mode: bool = False
):
    """Train multiple estimator initializations and keep the best one.

    Args:
        config: Experiment configuration
        method_config: Method-specific configuration
        batch_path: Path to batch data file
        output_dir: Base method directory (estimators/<method>)
        batch_name: Batch name (e.g., '0', '1', 'tuning', 'eval')
        overwrite: If True, overwrite existing models; if False, skip training if model exists
        use_wandb: Whether to use wandb logging
        sweep_mode: If True, skip wandb.init() (for W&B sweeps where init already called)
    """
    gamma = config.value_estimators.training.gamma
    episode_subsets = config.value_estimators.training.episode_subsets
    test_episodes = config.value_estimators.training.test_episodes
    method_name = method_config.name

    data_metadata = {}
    data_metadata_path = batch_path.parent / "data_metadata.json"
    if data_metadata_path.exists():
        with open(data_metadata_path, 'r') as f:
            data_metadata = json.load(f)

    # Load and preprocess test batch if test_episodes > 0
    test_batch = None
    if test_episodes > 0:
        eval_batch_path = batch_path.parent / "batch_ground_truth.npz"
        if eval_batch_path.exists():
            print(f"Loading test batch from {eval_batch_path}")
            eval_batch_raw = load_batch_data(eval_batch_path)
            print(f"Sampling {test_episodes} episodes for test set")
            test_batch_raw = sample_episodes(eval_batch_raw, test_episodes, seed=config.seed)
            print(f"Preprocessing test batch (flattening episodes, computing MC returns with gamma={gamma})")
            test_batch = preprocess_episodes(test_batch_raw, gamma)
            print(f"Test batch: {len(test_batch['observations'])} transitions")
        else:
            print(f"Warning: test_episodes={test_episodes} but {eval_batch_path} not found. No test set will be used.")

    for n_episodes in episode_subsets:
        print(f"\nLoading training batch from {batch_path}")
        train_batch_raw = load_batch_data(batch_path, max_episodes=n_episodes)

        # Directory structure: estimators/<method>/<n_episodes>/batch_<name>
        episodes_dir = output_dir / str(n_episodes) / f"batch_{batch_name}"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = episodes_dir / "estimator.pt"

        if checkpoint_path.exists() and not overwrite:
            print(f"Model exists at {checkpoint_path} and overwrite=False. Skipping.")
            continue

        print(f"\n{'='*80}")
        print(f"Training {method_name} on batch_{batch_name} with {n_episodes} episodes")
        print(f"{'='*80}")

        print(f"Preprocessing training batch (flattening episodes, computing MC returns with gamma={gamma})")
        train_batch = preprocess_episodes(train_batch_raw, gamma)

        obs_dim = train_batch['observations'].shape[-1]
        n_inits = method_config.n_initializations

        # Use method-specific max_epochs if set
        max_epochs_to_use = method_config.max_epochs if method_config.max_epochs is not None else config.value_estimators.training.max_epochs

        print(f"\nTraining {n_inits} initialization(s) of {method_name}")
        print(f"Episodes used: {n_episodes}")
        print(f"Max epochs: {max_epochs_to_use}")
        print()

        best_mc_loss = float('inf')
        best_estimator = None

        for init_idx in range(n_inits):
            torch.manual_seed(config.seed + init_idx)
            np.random.seed(config.seed + init_idx)
            estimator = create_estimator(method_config, config.network, obs_dim, gamma, config.experiment_id)

            final_mc_loss = train_single_initialization(
                estimator, train_batch, test_batch, config, method_name, batch_name, n_episodes, init_idx, use_wandb, sweep_mode
            )

            if final_mc_loss < best_mc_loss:
                best_mc_loss = final_mc_loss
                best_estimator = estimator
                print(f"  -> New best!")

        print(f"\nSaving best estimator (MC loss={best_mc_loss:.6f}) to {checkpoint_path}")
        best_estimator.save(checkpoint_path)

        stats = {
            'method': method_name,
            'batch_name': batch_name,
            'n_episodes': n_episodes,
            'n_initializations': n_inits,
            'best_mc_loss': best_mc_loss,
        }

        stats_path = episodes_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Save estimator metadata
        estimator_metadata = {
            'method': method_name,
            'batch_name': batch_name,
            'n_episodes': n_episodes,
            'batch_path': str(batch_path),
            'gamma': gamma,
            'n_initializations': n_inits,
            'best_mc_loss': best_mc_loss,
            'seed': config.seed,
            'max_epochs': config.value_estimators.training.max_epochs,
            'convergence_threshold': config.value_estimators.training.convergence_threshold,
            'convergence_patience': config.value_estimators.training.convergence_patience,
            'estimator_config': method_config.__dict__,
            'network_config': {
                'hidden_sizes': config.network.hidden_sizes,
                'activation': config.network.activation,
            },
            'data_metadata': data_metadata,
        }

        metadata_path = episodes_dir / "estimator_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(estimator_metadata, f, indent=2)

        print(f"Training complete for {method_name} batch_{batch_name} with {n_episodes} episodes.\n")


def main():
    parser = argparse.ArgumentParser(description="Train a value estimator")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument("--method", type=str, required=True,
                       help="Method name (corresponds to 'name' field in method config)")
    parser.add_argument("--batch-idx", type=int, required=False,
                       help="Batch index to train on (if not set, uses SGE_TASK_ID-1 from environment)")

    overwrite_group = parser.add_mutually_exclusive_group(required=True)
    overwrite_group.add_argument("--overwrite", dest="overwrite", action="store_true",
                       help="Overwrite existing models")
    overwrite_group.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                       help="Skip training if model already exists")
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

    # Set paths
    batch_name = str(batch_idx)
    batch_path = Path("experiments") / config.experiment_id / "data" / f"batch_{batch_name}.npz"
    output_dir = Path("experiments") / config.experiment_id / "estimators" / args.method

    # Save config to method directory
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Train estimator
    train_estimator(
        config=config,
        method_config=method_config,
        batch_path=batch_path,
        output_dir=output_dir,
        batch_name=batch_name,
        overwrite=args.overwrite,
        use_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()
