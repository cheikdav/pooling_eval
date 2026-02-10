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


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with W&B sweeps")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", type=str, required=True,
                       help="Method name (corresponds to 'name' field in method config)")
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline"], default="online",
                       help="W&B logging mode: online (real-time sync) or offline (sync at end)")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-update-rate", type=float, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=None)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--preprocess-fraction", type=float, default=None)
    parser.add_argument("--n-initializations", type=int, default=None)
    parser.add_argument("--parallel-inits", type=lambda x: str(x).lower() == 'true', default=False,
                       help="Train initializations in parallel")
    parser.add_argument("--n-jobs", type=int, default=None,
                       help="Number of parallel jobs (default: number of CPUs)")
    args = parser.parse_args()

    # Capture all output in buffer until we have the run ID
    buffer = StringIO()
    sys.stdout = buffer
    sys.stderr = buffer

    # Load config early to get paths
    config_temp = ExperimentConfig.from_yaml(args.config)

    # Initialize wandb (will pick up sweep config automatically)
    print(f"[SWEEP] Initializing wandb in {args.wandb_mode.upper()} mode")

    # For offline mode, set wandb dir (will be refined after getting run_id)
    base_wandb_dir = str(config_temp.get_wandb_dir() / "sweep" / config_temp.experiment_id) if args.wandb_mode == "offline" else None
    # Set environment-specific project as fallback (sweep config takes precedence)
    project_name = config_temp.logging.get_project_name(config_temp.environment.name)
    run = wandb.init(project=project_name, tags=["hyperparameter-tuning", "sweep"], mode=args.wandb_mode, dir=base_wandb_dir)
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

    # Override n_initializations from wandb sweep or CLI
    n_initializations = wandb.config.get('n_initializations', args.n_initializations)
    if n_initializations is not None:
        method_config.n_initializations = n_initializations
    else:
        # Default to 1 if not specified
        method_config.n_initializations = 1

    # Pre-declare all metrics with independent step counters to avoid conflicts in parallel mode
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pre-declaring wandb metrics")
    print(f"  Episode counts: {episode_subsets}")
    print(f"  Initializations per episode count: {method_config.n_initializations}")

    for episode_count in episode_subsets:
        for init_idx in range(method_config.n_initializations):
            suffix = f"_{episode_count}ep"
            if method_config.n_initializations > 1:
                suffix += f"_{init_idx}"
            wandb.define_metric(f"train{suffix}/*", step_metric=f"step{suffix}")
            wandb.define_metric(f"val{suffix}/*", step_metric=f"step{suffix}")

    # Setup paths
    batch_path = config.get_data_dir() / "batch_tuning.npz"
    output_dir = config.get_estimators_dir() / "sweeps" / args.method / wandb.run.id

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Train on all episode counts and collect statistics
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training on {len(episode_subsets)} episode counts...")
    sys.stdout.flush()
    sys.stderr.flush()

    episode_results = []
    for episode_count in episode_subsets:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training with {episode_count} episodes...")
        sys.stdout.flush()
        sys.stderr.flush()

        config.value_estimators.training.episode_subsets = [episode_count]

        result = train_estimator(
            config=config,
            method_config=method_config,
            batch_path=batch_path,
            output_dir=output_dir / f"{episode_count}ep",
            batch_name="tuning",
            overwrite=True,
            use_wandb=True,
            sweep_mode=True,
            parallel_inits=args.parallel_inits,
            n_jobs=args.n_jobs,
            log_dir=log_dir
        )

        if result:
            episode_results.append(result)

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
            aggregate_log[f'final/{ep}ep/mean_mc_loss'] = r['mean_mc_loss']
            aggregate_log[f'final/{ep}ep/std_mc_loss'] = r['std_mc_loss']

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

    # Get run directory for offline sync
    if args.wandb_mode == "offline":
        run_dir = str(Path(wandb.run.dir).parent)
        print(f"Wandb run directory: {run_dir}")

    wandb_finish_start = datetime.now()
    print(f"[{wandb_finish_start.strftime('%Y-%m-%d %H:%M:%S')}] Calling wandb.finish()...")
    sys.stdout.flush()
    sys.stderr.flush()

    wandb.finish()

    wandb_finish_end = datetime.now()
    elapsed = (wandb_finish_end - wandb_finish_start).total_seconds()
    print(f"[{wandb_finish_end.strftime('%Y-%m-%d %H:%M:%S')}] wandb.finish() completed (took {elapsed:.2f}s)")
    sys.stdout.flush()
    sys.stderr.flush()

    # Sync offline run to W&B
    if args.wandb_mode == "offline":
        sync_start = datetime.now()
        print(f"[{sync_start.strftime('%Y-%m-%d %H:%M:%S')}] Starting offline sync...")
        print(f"Syncing directory: {run_dir}")
        sys.stdout.flush()
        sys.stderr.flush()

        import subprocess
        try:
            subprocess.run(["wandb", "sync", run_dir], check=True, capture_output=True, text=True)
            sync_end = datetime.now()
            elapsed = (sync_end - sync_start).total_seconds()
            print(f"[{sync_end.strftime('%Y-%m-%d %H:%M:%S')}] ✓ Successfully synced to W&B (took {elapsed:.2f}s)")
        except subprocess.CalledProcessError as e:
            sync_end = datetime.now()
            elapsed = (sync_end - sync_start).total_seconds()
            print(f"[{sync_end.strftime('%Y-%m-%d %H:%M:%S')}] ✗ Warning: Failed to sync offline run (took {elapsed:.2f}s)")
            print(f"Error: {e}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            print(f"You can manually sync later with: wandb sync {run_dir}")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Wandb run finished and synced (online mode)")

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script exiting normally")
    sys.stdout.flush()
    sys.stderr.flush()



if __name__ == "__main__":
    main()
