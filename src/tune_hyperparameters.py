"""Hyperparameter tuning wrapper for W&B sweeps.

Simple wrapper that modifies learning rate and calls core training logic.
Uses batch_tuning.npz for all hyperparameter search.
"""

import argparse
import csv
from pathlib import Path
import sys
import wandb
from io import StringIO
import os
from datetime import datetime
import filelock

from src.config import ExperimentConfig
from src.train_estimator import train_estimator


def none_or_int(value):
    """Convert string to int, or None if value is 'None' or 'null'."""
    if value is None or str(value).lower() in ('none', 'null'):
        return None
    return int(value)


# Hyperparameters to record in the summary CSV (order matters for columns)
HYPERPARAM_KEYS = [
    'learning_rate', 'batch_size', 'target_update_rate',
    'ridge_lambda', 'n_components', 'rbf_n_components', 'rbf_gamma',
]


def write_sweep_summary_row(summary_path: Path, run_id: str, method_config,
                            training_config, episode_results: list):
    """Append one row to the sweep summary CSV.

    Each row contains: run_id, hyperparams, mean_val_mc_loss, and
    best_mc_loss for each episode count.
    """
    episode_results_sorted = sorted(episode_results, key=lambda r: r['num_episodes'])

    # Build the row dict
    row = {'run_id': run_id}

    # Hyperparams from method config (fall back to training config for batch_size)
    for key in HYPERPARAM_KEYS:
        if key == 'batch_size':
            val = getattr(method_config, 'batch_size', None)
            if val is None and training_config is not None:
                val = training_config.batch_size
            if val is not None:
                row[key] = val
        elif key == 'rbf_n_components':
            if method_config.feature_extractor is not None:
                row[key] = method_config.feature_extractor.n_components
        elif key == 'rbf_gamma':
            if method_config.feature_extractor is not None:
                row[key] = method_config.feature_extractor.gamma
        elif hasattr(method_config, key):
            row[key] = getattr(method_config, key)

    # Per-episode-count losses
    import numpy as np
    losses = []
    for r in episode_results_sorted:
        col = f"best_mc_loss_{r['num_episodes']}ep"
        row[col] = r['best_mc_loss']
        losses.append(r['best_mc_loss'])
    row['mean_val_mc_loss'] = np.mean(losses)

    # Column order: run_id, hyperparams, per-episode losses, mean
    ep_cols = [f"best_mc_loss_{r['num_episodes']}ep" for r in episode_results_sorted]
    fieldnames = ['run_id'] + HYPERPARAM_KEYS + ep_cols + ['mean_val_mc_loss']

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary_path.with_suffix('.csv.lock')

    with filelock.FileLock(lock_path):
        file_exists = summary_path.exists() and summary_path.stat().st_size > 0
        with open(summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with W&B sweeps")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", type=str, required=True,
                       help="Method name (corresponds to 'name' field in method config)")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-update-rate", type=float, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=None)
    parser.add_argument("--n-components", type=none_or_int, default=None,
                       help="Number of SVD/PCA components for dimensionality reduction (LeastSquares methods)")
    parser.add_argument("--rbf-n-components", type=none_or_int, default=None,
                       help="Number of RBF basis functions (for RBF feature extractor)")
    parser.add_argument("--rbf-gamma", type=float, default=None,
                       help="Gamma parameter for RBF kernel")
    parser.add_argument("--preprocess-fraction", type=float, default=None)
    parser.add_argument("--log-frequency", type=int, default=None,
                       help="Override logging frequency for W&B (log every N epochs)")
    parser.add_argument("--n-jobs", type=int, default=None,
                       help="Number of parallel jobs for training different episode counts (None or 1 = sequential)")
    args = parser.parse_args()

    # Capture all output in buffer until we have the run ID
    buffer = StringIO()
    sys.stdout = buffer
    sys.stderr = buffer

    # Load config early to get paths
    config_temp = ExperimentConfig.from_yaml(args.config)

    # Initialize wandb (will pick up sweep config automatically)
    # Sweeps always use online mode to avoid rate limit issues with proper log_frequency
    print(f"[SWEEP] Initializing wandb in ONLINE mode")

    # Set environment-specific project as fallback (sweep config takes precedence)
    project_name = config_temp.logging.get_project_name(config_temp.environment.name)
    run = wandb.init(project=project_name, tags=["hyperparameter-tuning", "sweep"], mode="online")
    print(f"[SWEEP] Wandb initialized: run_id={run.id}, mode={run.settings.mode}, url={run.url}")

    # Setup log directory: logs/sweep/<exp_id>/<method>/<sweep_id>/<run_id>/
    sweep_id = wandb.run.sweep_id or "no_sweep"
    log_dir = config_temp.get_logs_dir() / "sweep" / config_temp.experiment_id / args.method / sweep_id / wandb.run.id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "main.log"

    # Check if log file exists and is not empty (means we're re-running)
    log_exists = log_file.exists() and log_file.stat().st_size > 0

    # Redirect stdout/stderr to log file (append mode to detect re-runs)
    sys.stdout = open(log_file, 'a', buffering=1)
    sys.stderr = sys.stdout

    # Add separator if this is a re-run
    if log_exists:
        print("\n" + "="*80)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Log file exists - this appears to be a re-run!")
        print("="*80 + "\n")

    # Write buffered content to log file
    buffered_content = buffer.getvalue()
    if buffered_content:
        print(buffered_content, end='')

    # Add agent identification and timestamp
    agent_idx = os.environ.get('WANDB_AGENT_IDX', None)
    agent_label = f"Agent {agent_idx}" if agent_idx is not None else f"Agent (unknown index)"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {agent_label} - Sweep run {wandb.run.id} starting")
    print(f"Agent PID: {os.getpid()}")
    print(f"Parent PID: {os.getppid()}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Logs: {log_dir}")
    print(f"Re-run: {log_exists}")

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

    # Get episode_subsets - either from config or override from sweep/CLI
    episode_subsets = wandb.config.get('episode_subsets', None)
    if episode_subsets is None and args.num_episodes is not None:
        episode_subsets = [args.num_episodes]
    if episode_subsets is None:
        episode_subsets = config.value_estimators.training.episode_subsets

    # Sort in descending order to start with longest episode count
    episode_subsets = sorted(episode_subsets, reverse=True)

    print(f"[SWEEP] Training on episode counts: {episode_subsets}")

    # Override batch_size from wandb sweep or CLI
    batch_size = wandb.config.get('batch_size', args.batch_size)
    if batch_size is not None:
        config.value_estimators.training.batch_size = batch_size

    # Override ridge_lambda for LeastSquares methods from wandb sweep or CLI
    ridge_lambda = wandb.config.get('ridge_lambda', args.ridge_lambda)
    if ridge_lambda is not None and hasattr(method_config, 'ridge_lambda'):
        method_config.ridge_lambda = ridge_lambda

    # Override n_components for LeastSquares methods from wandb sweep or CLI
    n_components = wandb.config.get('n_components', args.n_components)
    if n_components is not None and hasattr(method_config, 'n_components'):
        method_config.n_components = n_components

    # Override rbf_n_components for RBF feature extractor from wandb sweep or CLI
    rbf_n_components = wandb.config.get('rbf_n_components', args.rbf_n_components)
    if rbf_n_components is not None and method_config.feature_extractor is not None:
        method_config.feature_extractor.n_components = rbf_n_components

    # Override rbf_gamma for RBF feature extractor from wandb sweep or CLI
    rbf_gamma = wandb.config.get('rbf_gamma', args.rbf_gamma)
    if rbf_gamma is not None and method_config.feature_extractor is not None:
        method_config.feature_extractor.gamma = rbf_gamma

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training configuration:")
    print(f"  Episode counts: {episode_subsets}")

    # Setup paths
    batch_path = config.get_data_dir() / "batch_tuning.npz"
    output_dir = config.get_estimator_dir(method_config) / "sweeps" / wandb.run.id

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Train on all episode counts and collect statistics
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training on {len(episode_subsets)} episode counts...")
    sys.stdout.flush()
    sys.stderr.flush()

    log_freq = wandb.config.get('log_frequency', args.log_frequency)
    n_jobs = wandb.config.get('n_jobs', args.n_jobs)

    episode_results = train_estimator(
            config=config,
            method_config=method_config,
            batch_path=batch_path,
            output_dir=output_dir,
            batch_name="tuning",
            overwrite=True,
            use_wandb=True,
            sweep_mode=True,
            log_dir=log_dir,
            log_frequency=log_freq,
            n_jobs=n_jobs
        )


    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training completed for all episode counts")

    # Aggregate statistics across episode counts
    if episode_results:
        import numpy as np

        all_best_losses = [r['best_mc_loss'] for r in episode_results]

        aggregate_log = {
            'final/mean_val_mc_loss': np.mean(all_best_losses),
            'final/std_val_mc_loss': np.std(all_best_losses),
            'final/min_val_mc_loss': np.min(all_best_losses),
            'final/max_val_mc_loss': np.max(all_best_losses),
        }

        for r in episode_results:
            ep = r['num_episodes']
            aggregate_log[f'final/{ep}ep/best_mc_loss'] = r['best_mc_loss']

        wandb.log(aggregate_log)

        print(f"\nAggregate statistics across {len(episode_results)} episode counts:")
        print(f"  Mean validation MC loss: {aggregate_log['final/mean_val_mc_loss']:.6f}")
        print(f"  Std validation MC loss:  {aggregate_log['final/std_val_mc_loss']:.6f}")
        print(f"  Min validation MC loss:  {aggregate_log['final/min_val_mc_loss']:.6f}")
        print(f"  Max validation MC loss:  {aggregate_log['final/max_val_mc_loss']:.6f}")

        # Write summary row to local CSV for automated hyperparam selection
        summary_path = config.get_estimator_dir(method_config) / "sweeps" / "sweep_results.csv"
        write_sweep_summary_row(summary_path, wandb.run.id, method_config,
                                config.value_estimators.training, episode_results)
        print(f"  Sweep summary appended to {summary_path}")

    sys.stdout.flush()
    sys.stderr.flush()

    # Properly finish wandb run to ensure all data is synced and run is marked complete
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finishing wandb run...")
    sys.stdout.flush()
    sys.stderr.flush()

    wandb_finish_start = datetime.now()
    print(f"[{wandb_finish_start.strftime('%Y-%m-%d %H:%M:%S')}] Calling wandb.finish()...")
    sys.stdout.flush()
    sys.stderr.flush()

    wandb.finish()

    wandb_finish_end = datetime.now()
    elapsed = (wandb_finish_end - wandb_finish_start).total_seconds()
    print(f"[{wandb_finish_end.strftime('%Y-%m-%d %H:%M:%S')}] wandb.finish() completed (took {elapsed:.2f}s)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Wandb run finished and synced (online mode)")
    sys.stdout.flush()
    sys.stderr.flush()

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script exiting normally")
    sys.stdout.flush()
    sys.stderr.flush()



if __name__ == "__main__":
    main()
