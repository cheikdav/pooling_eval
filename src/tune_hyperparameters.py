"""Hyperparameter tuning wrapper for W&B sweeps.

Simple wrapper that modifies learning rate and calls core training logic.
Uses batch_tuning.npz for all hyperparameter search.
"""

import argparse
from pathlib import Path
import sys
import wandb
from io import StringIO
import os
from datetime import datetime

from src.config import ExperimentConfig
from src.train_estimator import train_estimator


def none_or_int(value):
    """Convert string to int, or None if value is 'None' or 'null'."""
    if value is None or str(value).lower() in ('none', 'null'):
        return None
    return int(value)


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

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training configuration:")
    print(f"  Episode counts: {episode_subsets}")

    # Setup paths
    batch_path = config.get_data_dir() / "batch_tuning.npz"
    output_dir = config.get_estimators_dir() / "sweeps" / args.method / wandb.run.id

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
