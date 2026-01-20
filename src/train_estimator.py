"""Train a single value estimator on a batch of data."""

import argparse
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import wandb
import json
import copy

from src.config import ExperimentConfig, BaseEstimatorConfig, LeastSquaresMCConfig, LeastSquaresTDConfig
from src.estimators import ESTIMATOR_REGISTRY
from src.data_preprocessing import preprocess_episodes, sample_episodes, TransitionDataset, split_episodes_for_preprocessing
from torch.utils.data import DataLoader


def get_method_abbreviation(method_name: str) -> str:
    """Convert method name to abbreviation for wandb display.

    Args:
        method_name: Full method name (e.g., 'monte_carlo', 'dqn')

    Returns:
        Abbreviated method name (e.g., 'MC', 'TD')
    """
    abbreviations = {
        'monte_carlo': 'MC',
        'dqn': 'TD',
        'least_squares_mc': 'LSMC',
        'least_squares_td': 'LSTD',
    }
    return abbreviations.get(method_name, method_name)


def create_estimator(method_config: BaseEstimatorConfig, network_config, obs_dim: int, gamma: float, num_episodes: int, experiment_config=None):
    """Create an estimator instance from configuration using registry.

    Args:
        method_config: Method-specific configuration
        network_config: Network configuration
        obs_dim: Observation dimension
        gamma: Discount factor (from training config)
        num_episodes: Number of episodes (used to resolve per-episode hyperparameters)
        experiment_config: ExperimentConfig (needed for auto-setting policy_path)

    Returns:
        Estimator instance
    """
    # Resolve hyperparameters for this episode count
    resolved_config = method_config.resolve_for_episodes(num_episodes)

    # Auto-set policy_path for LeastSquares methods if not provided
    if isinstance(resolved_config, (LeastSquaresMCConfig, LeastSquaresTDConfig)) and resolved_config.policy_path is None:
        if experiment_config is None:
            raise ValueError("experiment_config is required when policy_path is not set in LeastSquares config")
        policy_path = experiment_config.get_policy_dir() / "policy_final.zip"
        resolved_config.policy_path = str(policy_path)
        print(f"Auto-set policy_path to: {policy_path}")

    EstimatorClass = ESTIMATOR_REGISTRY[type(resolved_config)]
    return EstimatorClass.from_config(resolved_config, network_config, obs_dim, gamma)


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


def check_convergence(current_epoch: int, current_loss: float, min_loss: float,
                     last_improvement_epoch: int, patience: int, threshold: float) -> tuple[bool, float, int]:
    """Check if training has converged based on validation loss.

    Converged if no significant improvement in last `patience` epochs.

    Args:
        current_epoch: Current epoch number
        current_loss: Current validation loss
        min_loss: Best validation loss seen so far
        last_improvement_epoch: Last epoch where loss improved significantly
        patience: Number of epochs without improvement before stopping
        threshold: Absolute improvement threshold

    Returns:
        (converged, new_min_loss, new_last_improvement_epoch)
    """
    # Check if current loss is significantly better than min loss
    if current_loss < min_loss - threshold:
        # Significant improvement found
        return False, current_loss, current_epoch

    # No significant improvement - check if we've exceeded patience
    epochs_since_improvement = current_epoch - last_improvement_epoch
    converged = epochs_since_improvement >= patience

    return converged, min_loss, last_improvement_epoch


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
    sweep_mode: bool = False,
    n_inits: int = 1
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
        n_inits: Total number of initializations (for logging prefix)

    Returns:
        Final MC loss value (test set if available, otherwise train set)
    """
    if use_wandb and config.logging.use_wandb and not sweep_mode:
        method_abbr = get_method_abbreviation(method_name)
        run_name = f"{method_abbr} ({config.environment.name}, #{batch_name}, #ep {num_episodes}, init {init_idx})"
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=run_name,
            group=config.experiment_id,
            mode=config.logging.wandb_mode,
            dir=str(config.get_estimators_dir() / "wandb_offline") if config.logging.wandb_mode == "offline" else None,
            config={
                'method': method_name,
                'batch_name': batch_name,
                'num_episodes': num_episodes,
                'init_idx': init_idx,
                'experiment_id': config.experiment_id,
                'environment': config.environment.name,
                **estimator.get_config(),
            },
            tags=[method_name, f"batch_{batch_name}", f"{num_episodes}_episodes", f"init{init_idx}", config.environment.name],
        )

    training_config = config.value_estimators.training
    loss_history = []
    mc_loss_history = []
    val_mc_loss_history = []
    final_mc_loss_train = float('inf')
    final_mc_loss_val = float('inf')
    best_mc_loss = float('inf')
    best_estimator = None
    use_validation = test_batch is not None
    converged = False
    final_epoch = 0

    # Track convergence state
    min_loss = float('inf')
    last_improvement_epoch = 0

    # Determine logging prefix for multiple initializations
    init_suffix = f"_{init_idx}" if n_inits > 1 else ""

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
    # For least squares methods, default to 1 epoch (closed-form solution)
    if isinstance(method_config, (LeastSquaresMCConfig, LeastSquaresTDConfig)):
        max_epochs = method_config.max_epochs if method_config.max_epochs is not None else 1
    else:
        max_epochs = method_config.max_epochs if (method_config and method_config.max_epochs is not None) else training_config.max_epochs

    # Offset step for wandb logging to ensure monotonicity across initializations
    step_offset = init_idx * max_epochs

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
        mc_loss_history.append(avg_metrics['mc_loss'])
        final_mc_loss_train = avg_metrics['mc_loss']

        # Compute validation MC loss every epoch for convergence check and model selection
        if use_validation:
            estimator.eval()
            with torch.no_grad():
                val_mc_returns = torch.FloatTensor(test_batch['mc_returns']).to(estimator.device)
                val_values = estimator.predict(test_batch['observations'])
                val_values = torch.FloatTensor(val_values).to(estimator.device)
                final_mc_loss_val = torch.nn.functional.mse_loss(val_values, val_mc_returns).item()
            val_mc_loss_history.append(final_mc_loss_val)

        if epoch % config.logging.log_frequency == 0 and use_wandb and config.logging.use_wandb:
            log_dict = {
                f'train{init_suffix}/loss': avg_metrics['loss'],
                f'train{init_suffix}/mse': avg_metrics['loss'],
                f'train{init_suffix}/mae': avg_metrics['mae'],
                f'train{init_suffix}/mean_value': avg_metrics['mean_value'],
                f'train{init_suffix}/mean_target': avg_metrics['mean_target'],
                f'train{init_suffix}/mc_loss_train': final_mc_loss_train,
                f'train{init_suffix}/best_mc_loss': best_mc_loss,
            }
            if use_validation:
                log_dict[f'val{init_suffix}/mc_loss'] = final_mc_loss_val
                log_dict[f'val{init_suffix}/min_mc_loss'] = min_loss
            wandb.log(log_dict, step=epoch + step_offset)

        # Check convergence based on validation MC loss (or training MC loss if no validation)
        final_mc_loss = final_mc_loss_val if use_validation else final_mc_loss_train

        converged, min_loss, last_improvement_epoch = check_convergence(
            epoch, final_mc_loss, min_loss, last_improvement_epoch,
            training_config.convergence_patience, training_config.convergence_threshold
        )
        if last_improvement_epoch == epoch:
            best_mc_loss = final_mc_loss
            best_estimator = copy.deepcopy(estimator)
            print(f"  [DEBUG] Epoch {epoch}: Updated best_estimator (MC loss: {best_mc_loss:.6f})")
            print(f"  [DEBUG] best_estimator is None: {best_estimator is None}")
        if converged:
            break


    # Log final epoch metrics to wandb (ensures last epoch is always logged)
    if use_wandb and config.logging.use_wandb:
        final_log_dict = {
            f'train{init_suffix}/loss': avg_metrics['loss'],
            f'train{init_suffix}/mse': avg_metrics['loss'],
            f'train{init_suffix}/mae': avg_metrics['mae'],
            f'train{init_suffix}/mean_value': avg_metrics['mean_value'],
            f'train{init_suffix}/mean_target': avg_metrics['mean_target'],
            f'train{init_suffix}/mc_loss_train': final_mc_loss_train,
            f'train{init_suffix}/best_mc_loss': best_mc_loss,
            f'train{init_suffix}/stop_reason': 'convergence' if converged else 'max_epochs',
        }
        if use_validation:
            final_log_dict[f'val{init_suffix}/mc_loss'] = final_mc_loss_val
            final_log_dict[f'val{init_suffix}/min_mc_loss'] = min_loss
        wandb.log(final_log_dict, step=final_epoch + step_offset)

    # Print training summary
    print(f"\n  Init {init_idx} Summary:")
    print(f"    Epochs: {final_epoch + 1}/{max_epochs}")
    if converged:
        if use_validation:
            print(f"    Stopped: Converged (validation MC loss improvement < {training_config.convergence_threshold} for {training_config.convergence_patience} epochs)")
        else:
            print(f"    Stopped: Converged (training MC loss improvement < {training_config.convergence_threshold} for {training_config.convergence_patience} epochs)")
    else:
        print(f"    Stopped: Reached max epochs")
    print(f"    Final train loss: {loss_history[-1]:.6f}")
    print(f"    Final train MC loss: {final_mc_loss_train:.6f}")
    if use_validation:
        print(f"    Final val MC loss: {final_mc_loss_val:.6f}")
    print(f"    Best MC loss: {best_mc_loss:.6f}")

    if use_wandb and config.logging.use_wandb:
        # Log per-initialization final metrics
        final_log = {f'final{init_suffix}/best_mc_loss': best_mc_loss}
        if use_validation:
            final_log[f'final{init_suffix}/mc_loss_train'] = final_mc_loss_train
            final_log[f'final{init_suffix}/mc_loss_val'] = final_mc_loss_val
        else:
            final_log[f'final{init_suffix}/mc_loss_train'] = final_mc_loss_train
        wandb.log(final_log)

        # Only finish wandb run if not in sweep mode (sweep manages the run lifecycle)
        if not sweep_mode:
            # wandb.run.dir points to 'files' subdirectory; sync needs parent directory
            if config.logging.wandb_mode == "offline":
                run_dir = str(Path(wandb.run.dir).parent)
                print(f"\n  [DEBUG] Wandb run directory: {run_dir}")
            else:
                run_dir = None

            wandb.finish()

            if config.logging.wandb_mode == "offline" and run_dir:
                print(f"\n  Syncing offline run to W&B...")
                print(f"  [DEBUG] Syncing directory: {run_dir}")
                import subprocess
                try:
                    subprocess.run(["wandb", "sync", run_dir], check=True, capture_output=True, text=True)
                    print(f"  ✓ Successfully synced to W&B")
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Warning: Failed to sync offline run")
                    print(f"  Error: {e}")
                    if e.stdout:
                        print(f"  stdout: {e.stdout}")
                    if e.stderr:
                        print(f"  stderr: {e.stderr}")
                    print(f"  You can manually sync later with: wandb sync {run_dir}")

    print(f"\n  [DEBUG] End of training:")
    print(f"  [DEBUG] best_estimator is None: {best_estimator is None}")
    print(f"  [DEBUG] best_mc_loss: {best_mc_loss}")
    print(f"  [DEBUG] Returning estimator is None: {(best_estimator if best_estimator is not None else estimator) is None}")

    return best_mc_loss, best_estimator if best_estimator is not None else estimator


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

    # Load and preprocess validation batch if it exists
    test_batch = None
    if test_episodes > 0:
        # Try to load per-batch validation set first
        validation_batch_path = batch_path.parent / f"{batch_path.stem}_validation.npz"
        if validation_batch_path.exists():
            print(f"Loading validation batch from {validation_batch_path}")
            validation_batch_raw = load_batch_data(validation_batch_path)
            print(f"Sampling {test_episodes} episodes for test set")
            test_batch_raw = sample_episodes(validation_batch_raw, test_episodes, seed=config.seed)
            print(f"Preprocessing test batch (flattening episodes, computing MC returns with gamma={gamma})")
            test_batch = preprocess_episodes(test_batch_raw, gamma)
            print(f"Test batch: {len(test_batch['observations'])} transitions")
        else:
            print(f"Warning: test_episodes={test_episodes} but {validation_batch_path} not found. No test set will be used.")

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

        # Split for preprocessing if needed (for least squares with PCA)
        preprocess_batch_raw = None
        preprocess_batch = None
        if isinstance(method_config, (LeastSquaresMCConfig, LeastSquaresTDConfig)) and method_config.preprocess_fraction > 0.0:
            print(f"Splitting {method_config.preprocess_fraction*100:.1f}% of episodes for PCA preprocessing")
            preprocess_batch_raw, train_batch_raw = split_episodes_for_preprocessing(
                train_batch_raw, method_config.preprocess_fraction, seed=config.seed
            )
            print(f"  Preprocessing: {len(preprocess_batch_raw['observations'])} episodes")
            print(f"  Training: {len(train_batch_raw['observations'])} episodes")
            print(f"Preprocessing preprocessing batch (flattening episodes, computing MC returns with gamma={gamma})")
            preprocess_batch = preprocess_episodes(preprocess_batch_raw, gamma)

        print(f"Preprocessing training batch (flattening episodes, computing MC returns with gamma={gamma})")
        train_batch = preprocess_episodes(train_batch_raw, gamma)

        obs_dim = train_batch['observations'].shape[-1]
        n_inits = method_config.n_initializations

        # Use method-specific max_epochs if set, otherwise use global
        # For least squares methods, default to 1 epoch (closed-form solution)
        if isinstance(method_config, (LeastSquaresMCConfig, LeastSquaresTDConfig)):
            max_epochs_to_use = method_config.max_epochs if method_config.max_epochs is not None else 1
        else:
            max_epochs_to_use = method_config.max_epochs if method_config.max_epochs is not None else config.value_estimators.training.max_epochs

        print(f"\nTraining {n_inits} initialization(s) of {method_name}")
        print(f"Episodes used: {n_episodes}")
        print(f"Max epochs: {max_epochs_to_use}")
        print()

        best_mc_loss = float('inf')
        best_estimator = None
        all_mc_losses = []  # Collect MC losses from all initializations

        for init_idx in range(n_inits):
            torch.manual_seed(config.seed + init_idx)
            np.random.seed(config.seed + init_idx)
            estimator = create_estimator(method_config, config.network, obs_dim, gamma, n_episodes, config)

            # Fit PCA projection if preprocessing data available
            if preprocess_batch is not None and hasattr(estimator, 'fit_pca_projection'):
                print(f"Fitting PCA projection on preprocessing batch")
                estimator.fit_pca_projection(preprocess_batch)

            final_mc_loss, estimator = train_single_initialization(
                estimator, train_batch, test_batch, config, method_name, batch_name, n_episodes, init_idx, use_wandb, sweep_mode, n_inits
            )

            print(f"  [DEBUG] Received estimator from train_single_initialization, is None: {estimator is None}")

            all_mc_losses.append(final_mc_loss)

            if final_mc_loss < best_mc_loss:
                best_mc_loss = final_mc_loss
                best_estimator = estimator
                print(f"  -> New best!")

        # Compute statistics across all initializations
        min_mc_loss = float(np.min(all_mc_losses))
        mean_mc_loss = float(np.mean(all_mc_losses))
        std_mc_loss = float(np.std(all_mc_losses))

        print(f"\n[DEBUG] Before saving:")
        print(f"[DEBUG] best_estimator is None: {best_estimator is None}")
        print(f"[DEBUG] best_mc_loss: {best_mc_loss}")
        print(f"\nStatistics across {n_inits} initialization(s):")
        print(f"  Min MC loss:  {min_mc_loss:.6f}")
        print(f"  Mean MC loss: {mean_mc_loss:.6f}")
        print(f"  Std MC loss:  {std_mc_loss:.6f}")
        print(f"\nSaving best estimator (MC loss={best_mc_loss:.6f}) to {checkpoint_path}")
        best_estimator.save(checkpoint_path)

        stats = {
            'method': method_name,
            'batch_name': batch_name,
            'n_episodes': n_episodes,
            'n_initializations': n_inits,
            'best_mc_loss': best_mc_loss,
            'min_mc_loss': min_mc_loss,
            'mean_mc_loss': mean_mc_loss,
            'std_mc_loss': std_mc_loss,
            'all_mc_losses': all_mc_losses,
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

        # Log aggregate statistics to wandb in sweep mode
        if sweep_mode and use_wandb and config.logging.use_wandb:
            import wandb
            aggregate_log = {
                'final/min_mc_loss': min_mc_loss,
                'final/mean_mc_loss': mean_mc_loss,
                'final/std_mc_loss': std_mc_loss,
                'final/best_mc_loss': best_mc_loss,
                'final/best_val_mc_loss': best_mc_loss,  # For sweep optimization
            }

            # Create scatter plot: mean vs std
            # This will show the relationship between mean loss and variability across initializations
            data = [[mean_mc_loss, std_mc_loss]]
            table = wandb.Table(data=data, columns=["mean_mc_loss", "std_mc_loss"])
            aggregate_log['final/mean_vs_std_scatter'] = wandb.plot.scatter(
                table, "mean_mc_loss", "std_mc_loss",
                title="Mean vs Std MC Loss (across initializations)"
            )

            wandb.log(aggregate_log)
            print(f"\nLogged aggregate statistics to wandb:")
            print(f"  Min MC loss:  {min_mc_loss:.6f}")
            print(f"  Mean MC loss: {mean_mc_loss:.6f}")
            print(f"  Std MC loss:  {std_mc_loss:.6f}")

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
    batch_path = config.get_data_dir() / f"batch_{batch_name}.npz"
    output_dir = config.get_estimators_dir() / args.method

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
