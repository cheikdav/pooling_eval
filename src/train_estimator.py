"""Train a single value estimator on a batch of data."""

import argparse
import copy
from collections import defaultdict
import numpy as np
from pathlib import Path
import torch
import wandb
import json
from dataclasses import asdict
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from src.config import ExperimentConfig, BaseEstimatorConfig, LeastSquaresMCConfig, LeastSquaresTDConfig
from src.estimators import ESTIMATOR_REGISTRY
from src.data_preprocessing import preprocess_episodes, TransitionDataset
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

    # Auto-set policy_path for PolicyRepresentation feature extractors if not provided
    from src.config import FeatureExtractorType
    if resolved_config.feature_extractor and resolved_config.feature_extractor.type == FeatureExtractorType.POLICY_REPRESENTATION:
        if resolved_config.feature_extractor.policy_path is None:
            if experiment_config is None:
                raise ValueError("experiment_config is required when policy_path is not set in feature_extractor config")
            policy_path = experiment_config.get_policy_dir() / "policy_final.zip"
            resolved_config.feature_extractor.policy_path = str(policy_path)
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


def train_single_estimator(
    method_config: BaseEstimatorConfig,
    train_batch: dict,
    validation_dataset,
    config: ExperimentConfig,
    method_name: str,
    batch_name: str,
    num_episodes: int,
    use_wandb: bool,
    sweep_mode: bool = False,
    log_frequency: int = 10
):
    """Train a value estimator and return its final MC loss.

    Args:
        method_config: Method-specific configuration
        train_batch: Training batch data (preprocessed)
        validation_dataset: Validation TransitionDataset (None if no validation set)
        config: Experiment configuration
        method_name: Method name (from method_config.name)
        batch_name: Batch name (e.g., '0', '1', 'tuning', 'eval')
        num_episodes: Number of episodes in batch
        use_wandb: Whether to use wandb
        sweep_mode: If True, skip wandb.init() (run already initialized by sweep)
        log_frequency: Log progress every N epochs (0 = no periodic logging)

    Returns:
        Tuple of (final MC loss, trained estimator)
    """
    print("[train_single_estimator] v2 - with best checkpoint restoration")
    # Create estimator
    gamma = config.value_estimators.training.gamma
    obs_dim = train_batch['observations'].shape[-1]
    estimator = create_estimator(method_config, config.network, obs_dim, gamma, num_episodes, config)
    if use_wandb and config.logging.use_wandb and not sweep_mode:
        method_abbr = get_method_abbreviation(method_name)
        run_name = f"{method_abbr} ({config.environment.name}, #{batch_name}, #ep {num_episodes})"
        wandb.init(
            project=config.logging.get_project_name(config.environment.name),
            entity=config.logging.wandb_entity,
            name=run_name,
            group=config.experiment_id,
            mode=config.logging.wandb_mode,
            dir=str(config.get_estimators_dir() / "wandb_offline") if config.logging.wandb_mode == "offline" else None,
            config={
                'method': method_name,
                'batch_name': batch_name,
                'num_episodes': num_episodes,
                'experiment_id': config.experiment_id,
                'environment': config.environment.name,
                **estimator.get_config(),
            },
            tags=[method_name, f"batch_{batch_name}", f"{num_episodes}_episodes", config.environment.name],
        )

    training_config = config.value_estimators.training
    val_mc_loss_history = []
    final_mc_loss_train = float('inf')
    final_mc_loss_val = float('inf')
    best_mc_loss = float('inf')
    best_checkpoint = None
    use_validation = validation_dataset is not None
    converged = False
    final_epoch = 0

    # Track convergence state
    min_loss = float('inf')
    last_improvement_epoch = 0

    # Determine logging suffix structure
    if sweep_mode:
        # Sweep mode: metrics grouped by episode count (train/mc_loss/50ep)
        suffix = f"/{num_episodes}ep"

        # Define custom step metrics
        # In sweep mode, hide train/* but keep val/mc_loss/* visible
        if use_wandb and config.logging.use_wandb:
            # Define exact metric names (hidden only works with exact matches, not wildcards)
            wandb.define_metric(f"train/mc_loss{suffix}", step_metric=f"step{suffix}", hidden=True)
            wandb.define_metric(f"train/loss{suffix}", step_metric=f"step{suffix}", hidden=True)
            wandb.define_metric(f"train/mse{suffix}", step_metric=f"step{suffix}", hidden=True)
            wandb.define_metric(f"train/mae{suffix}", step_metric=f"step{suffix}", hidden=True)
            wandb.define_metric(f"train/mean_value{suffix}", step_metric=f"step{suffix}", hidden=True)
            wandb.define_metric(f"train/mean_target{suffix}", step_metric=f"step{suffix}", hidden=True)
            wandb.define_metric(f"train/best_mc_loss{suffix}", step_metric=f"step{suffix}", hidden=True)
            # Keep validation mc_loss visible
            wandb.define_metric(f"val/mc_loss{suffix}", step_metric=f"step{suffix}", hidden=False)
            wandb.define_metric(f"val/min_loss{suffix}", step_metric=f"step{suffix}", hidden=True)
            wandb.define_metric(f"step{suffix}", hidden=True)
    else:
        # Normal mode: no suffix needed
        suffix = ""

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
    from src.config import resolve_param_for_episodes
    if isinstance(method_config, (LeastSquaresMCConfig, LeastSquaresTDConfig)):
        max_epochs_raw = method_config.max_epochs if method_config.max_epochs is not None else 1
    else:
        max_epochs_raw = method_config.max_epochs if (method_config and method_config.max_epochs is not None) else training_config.max_epochs
    max_epochs = resolve_param_for_episodes(max_epochs_raw, num_episodes)

    # Use method-specific batch_size if set, otherwise use global
    # Resolve dict-based batch_size for the specific episode count
    batch_size_raw = method_config.batch_size if (method_config and method_config.batch_size is not None) else training_config.batch_size
    batch_size = resolve_param_for_episodes(batch_size_raw, num_episodes)

    # Pre-training pass to initialize normalizer and accumulate reward statistics
    print("Running pre-training pass to initialize normalizer...")
    pre_train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for mini_batch in pre_train_dataloader:
        estimator.pre_training_pass(mini_batch)

    # Per-method reward_centering overrides global setting
    global_reward_centering = config.value_estimators.training.reward_centering
    reward_centering = method_config.reward_centering if method_config.reward_centering is not None else global_reward_centering
    estimator.finalize_pre_training(reward_centering)
    if reward_centering:
        print(f"Reward centering enabled: mean_reward={estimator.mean_reward:.6f}, offset={estimator.reward_offset:.6f}")
    print("Normalizer initialized and frozen.")

    # Cache features to avoid recomputation during training
    print("Caching features for training dataset...")
    estimator.cache_features_in_dataset(dataset, batch_size=batch_size)
    print("Training features cached successfully.")

    # Cache validation features if validation dataset exists
    if use_validation:
        print("Caching features for validation dataset...")
        estimator.cache_features_in_dataset(validation_dataset, batch_size=batch_size)
        print("Validation features cached successfully.")

    for epoch in range(max_epochs):
        final_epoch = epoch
        # Recreate dataloader when needed for shuffling
        if (dataloader is None or
            (training_config.shuffle_frequency > 0 and epoch % training_config.shuffle_frequency == 0)):
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        epoch_metrics = defaultdict(list)
        for mini_batch in dataloader:
            metrics = estimator.train_step(mini_batch)
            for loss_name, loss_value in metrics.items():
                epoch_metrics[loss_name].append(loss_value)
            
        # Average metrics across mini-batches
        avg_metrics = {loss_name: np.mean(values) for loss_name, values in epoch_metrics.items()}

        # Update final training metrics if available
        if 'mc_loss' in avg_metrics:
            final_mc_loss_train = avg_metrics['mc_loss']

        # Compute validation MC loss every epoch for convergence check and model selection
        if use_validation:
            estimator.eval()
            with torch.no_grad():
                # Use cached features from validation dataset
                val_mc_returns = validation_dataset.mc_returns.to(estimator.device)
                val_values = estimator.predict(
                    validation_dataset.observations.cpu().numpy(),
                    features=validation_dataset.features
                )
                val_values = torch.FloatTensor(val_values).to(estimator.device)
                final_mc_loss_val = torch.nn.functional.mse_loss(val_values, val_mc_returns).item()
            val_mc_loss_history.append(final_mc_loss_val)
            
        # Check convergence based on validation MC loss (or training MC loss if no validation)
        final_mc_loss = final_mc_loss_val if use_validation else final_mc_loss_train

        converged, min_loss, last_improvement_epoch = check_convergence(
            epoch, final_mc_loss, min_loss, last_improvement_epoch,
            training_config.convergence_patience, training_config.convergence_threshold
        )
        if last_improvement_epoch == epoch:
            best_mc_loss = final_mc_loss
            best_checkpoint = copy.deepcopy(estimator._build_checkpoint())
        
        is_last_epoch = (epoch == max_epochs - 1) or converged
        if (log_frequency > 0 and epoch % log_frequency == 0) or is_last_epoch:
            if use_wandb and config.logging.use_wandb:
                log_dict = {
                    f'train/{metric_name}{suffix}': metric_value
                    for metric_name, metric_value in avg_metrics.items()
                }
                log_dict[f'step{suffix}'] = epoch
                log_dict[f'train/best_mc_loss{suffix}'] = best_mc_loss

                if use_validation:
                    log_dict[f'val/mc_loss{suffix}'] = final_mc_loss_val
                    log_dict[f'val/min_loss{suffix}'] = min_loss
                wandb.log(log_dict)


            print(f"Epoch {epoch+1}/{max_epochs}: train_loss={final_mc_loss_train:.6f}, best_loss={best_mc_loss:.6f}", end="")
            if use_validation:
                print(f", val_loss={final_mc_loss_val:.6f}", end="")
            print()
       
        if converged:
            break

    # Print training summary
    print(f"\nTraining Summary:")
    print(f"  Epochs: {final_epoch + 1}/{max_epochs}")
    if converged:
        if use_validation:
            print(f"  Stopped: Converged (validation MC loss improvement < {training_config.convergence_threshold} for {training_config.convergence_patience} epochs)")
        else:
            print(f"  Stopped: Converged (training MC loss improvement < {training_config.convergence_threshold} for {training_config.convergence_patience} epochs)")
    else:
        print(f"  Stopped: Reached max epochs")
    if 'loss' in avg_metrics:
        print(f"  Final train loss: {avg_metrics['loss']:.6f}")
    print(f"  Final train MC loss: {final_mc_loss_train:.6f}")
    if use_validation:
        print(f"  Final val MC loss: {final_mc_loss_val:.6f}")
    print(f"  Best MC loss: {best_mc_loss:.6f}")

    if use_wandb and config.logging.use_wandb:
        # Log final metrics
        final_log = {f'final/best_mc_loss{suffix}': best_mc_loss}
        if use_validation:
            final_log[f'final/mc_loss_train{suffix}'] = final_mc_loss_train
            final_log[f'final/mc_loss_val{suffix}'] = final_mc_loss_val
        else:
            final_log[f'final/mc_loss_train{suffix}'] = final_mc_loss_train
        wandb.log(final_log)

        # Only finish wandb run if not in sweep mode (sweep manages the run lifecycle)
        if not sweep_mode:
            # wandb.run.dir points to 'files' subdirectory; sync needs parent directory
            if config.logging.wandb_mode == "offline":
                run_dir = str(Path(wandb.run.dir).parent)
            else:
                run_dir = None

            wandb.finish()

            if config.logging.wandb_mode == "offline" and run_dir:
                print(f"\nSyncing offline run to W&B...")
                import subprocess
                try:
                    subprocess.run(["wandb", "sync", run_dir], check=True, capture_output=True, text=True)
                    print(f"✓ Successfully synced to W&B")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Warning: Failed to sync offline run")
                    print(f"Error: {e}")
                    if e.stdout:
                        print(f"stdout: {e.stdout}")
                    if e.stderr:
                        print(f"stderr: {e.stderr}")
                    print(f"You can manually sync later with: wandb sync {run_dir}")

    # Restore best checkpoint weights (deepcopy fails with torch.compile)
    if best_checkpoint is not None:
        estimator._load_from_checkpoint_dict(best_checkpoint)
    return best_mc_loss, estimator


def train_episode_count_worker(
    n_episodes: int,
    config: ExperimentConfig,
    method_config: BaseEstimatorConfig,
    batch_path: Path,
    output_dir: Path,
    batch_name: str,
    use_wandb: bool,
    sweep_mode: bool,
    log_dir: Path,
    log_frequency: int,
    validation_batch_path: Path,
    gamma: float
) -> dict:
    """Worker function to train on a single episode count (used for parallel execution).

    Returns:
        Dictionary with 'num_episodes' and 'best_mc_loss'
    """
    method_name = method_config.name

    # Load and preprocess validation batch if it exists
    validation_dataset = None
    truncation_coefficient = config.value_estimators.training.truncation_coefficient
    if validation_batch_path.exists():
        validation_batch_raw = load_batch_data(validation_batch_path)
        validation_batch = preprocess_episodes(validation_batch_raw, gamma, truncation_coefficient)
        validation_dataset = TransitionDataset(validation_batch)

    # Load training batch
    train_batch_raw = load_batch_data(batch_path, max_episodes=n_episodes)

    # Directory structure: estimators/<method>/<n_episodes>/batch_<name>
    episodes_dir = output_dir / str(n_episodes) / f"batch_{batch_name}"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = episodes_dir / "estimator.pt"

    # Preprocess training batch
    train_batch = preprocess_episodes(train_batch_raw, gamma, truncation_coefficient)

    # Use method-specific max_epochs if set, otherwise use global
    from src.config import resolve_param_for_episodes
    if isinstance(method_config, (LeastSquaresMCConfig, LeastSquaresTDConfig)):
        max_epochs_raw = method_config.max_epochs if method_config.max_epochs is not None else 1
    else:
        max_epochs_raw = method_config.max_epochs if method_config.max_epochs is not None else config.value_estimators.training.max_epochs
    max_epochs_to_use = resolve_param_for_episodes(max_epochs_raw, n_episodes)

    # Setup log file redirection if log_dir is provided
    log_file_handle = None
    original_stdout = None
    original_stderr = None
    log_file_path = None

    if log_dir is not None:
        episode_log_dir = log_dir / f"{n_episodes}ep"
        episode_log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = episode_log_dir / "training.log"
        log_exists = log_file_path.exists() and log_file_path.stat().st_size > 0

        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Redirect to log file
        log_file_handle = open(log_file_path, 'a', buffering=1)
        sys.stdout = log_file_handle
        sys.stderr = log_file_handle

        if log_exists:
            print("\n" + "="*80)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Log file exists - this appears to be a re-run!")
            print("="*80 + "\n")

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training")

    print(f"\nTraining {method_name}")
    print(f"Episodes used: {n_episodes}")
    print(f"Max epochs: {max_epochs_to_use}")
    print()

    # Train estimator
    final_mc_loss, trained_estimator = train_single_estimator(
        method_config, train_batch, validation_dataset, config, method_name, batch_name,
        n_episodes, use_wandb=use_wandb, sweep_mode=sweep_mode, log_frequency=log_frequency
    )

 

    if not sweep_mode:
        print(f"\nSaving estimator (MC loss={final_mc_loss:.6f}) to {checkpoint_path}")
        trained_estimator.save(checkpoint_path)

        # Save stats and metadata
        stats = {
            'method': method_name,
            'batch_name': batch_name,
            'n_episodes': n_episodes,
            'final_mc_loss': final_mc_loss,
        }

        stats_path = episodes_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Get data metadata
        data_metadata = {}
        data_metadata_path = batch_path.parent / "data_metadata.json"
        if data_metadata_path.exists():
            with open(data_metadata_path, 'r') as f:
                data_metadata = json.load(f)

        # Save estimator metadata
        estimator_metadata = {
            'method': method_name,
            'batch_name': batch_name,
            'n_episodes': n_episodes,
            'batch_path': str(batch_path),
            'gamma': gamma,
            'final_mc_loss': final_mc_loss,
            'seed': config.seed,
            'max_epochs': config.value_estimators.training.max_epochs,
            'convergence_threshold': config.value_estimators.training.convergence_threshold,
            'convergence_patience': config.value_estimators.training.convergence_patience,
            'estimator_config': asdict(method_config),
            'network_config': {
                'hidden_sizes': config.network.hidden_sizes,
                'activation': config.network.activation,
            },
            'data_metadata': data_metadata,
        }

        metadata_path = episodes_dir / "estimator_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(estimator_metadata, f, indent=2)

   # Restore stdout/stderr if we redirected
    if log_file_handle is not None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_handle.close()

    if log_file_handle is None:
        print(f"Training complete for {method_name} batch_{batch_name} with {n_episodes} episodes.\n")
    else:
        print(f"Training complete for {method_name} batch_{batch_name} with {n_episodes} episodes. Logs: {log_file_path}")

    return {
        'num_episodes': n_episodes,
        'best_mc_loss': final_mc_loss,
    }


def train_estimator(
    config: ExperimentConfig,
    method_config: BaseEstimatorConfig,
    batch_path: Path,
    output_dir: Path,
    batch_name: str,
    overwrite: bool,
    use_wandb: bool = True,
    sweep_mode: bool = False,
    log_dir: Path = None,
    log_frequency: int = None,
    n_jobs: int = None
):
    """Train a value estimator.

    Args:
        config: Experiment configuration
        method_config: Method-specific configuration
        batch_path: Path to batch data file
        output_dir: Base method directory (estimators/<method>)
        batch_name: Batch name (e.g., '0', '1', 'tuning', 'eval')
        overwrite: If True, overwrite existing models; if False, skip training if model exists
        use_wandb: Whether to use wandb logging
        sweep_mode: If True, skip wandb.init() (for W&B sweeps where init already called)
        log_dir: Directory for logs (if None, no file logging - output goes to stdout)
        log_frequency: Override logging frequency (if None, uses config.logging.log_frequency)
        n_jobs: Number of parallel jobs for training different episode counts (None or 1 = sequential)
    """
    gamma = config.value_estimators.training.gamma
    episode_subsets = config.value_estimators.training.episode_subsets
    method_name = method_config.name

    # Prepare common arguments for worker
    validation_batch_path = batch_path.parent / f"{batch_path.stem}_validation.npz"
    if validation_batch_path.exists():
        print(f"Found validation batch at {validation_batch_path}")
    else:
        print(f"No validation batch found at {validation_batch_path}. Training without validation.")

    log_freq = log_frequency if log_frequency is not None else config.logging.log_frequency

    # Sort episode_subsets in descending order to start with longest
    episode_subsets = sorted(episode_subsets, reverse=True)

    # Check for existing models if overwrite=False
    episode_subsets_to_train = []
    for n_episodes in episode_subsets:
        episodes_dir = output_dir / str(n_episodes) / f"batch_{batch_name}"
        checkpoint_path = episodes_dir / "estimator.pt"

        if checkpoint_path.exists() and not overwrite:
            print(f"Model exists at {checkpoint_path} and overwrite=False. Skipping {n_episodes} episodes.")
        else:
            episode_subsets_to_train.append(n_episodes)

    if not episode_subsets_to_train:
        print("All models already exist. Nothing to train.")
        return [] if sweep_mode else None

    # Train on all episode counts (parallel or sequential)
    episode_results = []

    if n_jobs and n_jobs > 1:
        # Parallel mode: train different episode counts concurrently
        print(f"\nTraining {len(episode_subsets_to_train)} episode counts in parallel (n_jobs={n_jobs})...")
        available_cpus = mp.cpu_count()
        max_workers = min(n_jobs, len(episode_subsets_to_train), available_cpus)
        print(f"Available CPUs: {available_cpus}, Using {max_workers} workers\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for n_episodes in episode_subsets_to_train:
                print(f"\n{'='*80}")
                print(f"Submitting job: {method_name} on batch_{batch_name} with {n_episodes} episodes")
                print(f"{'='*80}\n")

                future = executor.submit(
                    train_episode_count_worker,
                    n_episodes, config, method_config, batch_path, output_dir,
                    batch_name, use_wandb, sweep_mode, log_dir, log_freq,
                    validation_batch_path, gamma
                )
                futures.append((n_episodes, future))

            for n_episodes, future in futures:
                try:
                    result = future.result()
                    episode_results.append(result)
                    print(f"Episode count {n_episodes} completed: MC loss = {result['best_mc_loss']:.6f}")
                except Exception as e:
                    print(f"Episode count {n_episodes} failed with error: {e}")
                    raise

    else:
        # Sequential mode: train episode counts one by one
        print(f"\nTraining {len(episode_subsets_to_train)} episode counts sequentially...\n")

        for n_episodes in episode_subsets_to_train:
            print(f"\n{'='*80}")
            print(f"Training {method_name} on batch_{batch_name} with {n_episodes} episodes")
            print(f"{'='*80}\n")

            result = train_episode_count_worker(
                n_episodes, config, method_config, batch_path, output_dir,
                batch_name, use_wandb, sweep_mode, log_dir, log_freq,
                validation_batch_path, gamma
            )
            episode_results.append(result)

    if sweep_mode:
        return episode_results



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
    parser.add_argument("--timestamp", type=str, default=None,
                       help="Timestamp for grouping logs (auto-generated if not provided)")
    parser.add_argument("--n-jobs", type=int, default=None,
                       help="Number of parallel jobs for training different episode counts (None or 1 = sequential)")
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

    # Get or generate timestamp for this training session
    if args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create log directory: logs/estimator/<exp_id>/<method>/batch_<name>/<timestamp>/
    log_dir = config.get_logs_dir() / "estimator" / config.experiment_id / args.method / timestamp / f"batch_{batch_name}" 
    log_dir.mkdir(parents=True, exist_ok=True)

    # Train estimator
    train_estimator(
        config=config,
        method_config=method_config,
        batch_path=batch_path,
        output_dir=output_dir,
        batch_name=batch_name,
        overwrite=args.overwrite,
        use_wandb=not args.no_wandb,
        log_dir=log_dir,
        n_jobs=args.n_jobs
    )


if __name__ == "__main__":
    main()
