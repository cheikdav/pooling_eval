"""Hyperparameter tuning wrapper for W&B sweeps.

Simple wrapper that modifies learning rate and calls core training logic.
Uses batch_tuning.npz for all hyperparameter search.
"""

# Enable file tracking FIRST before any other imports
from src.debug_file_tracking import enable_file_tracking
enable_file_tracking()

import argparse
from pathlib import Path
import sys
import wandb
from io import StringIO

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
    run = wandb.init(tags=["hyperparameter-tuning", "sweep"], mode=args.wandb_mode, dir=base_wandb_dir)
    print(f"[SWEEP] Wandb initialized: run_id={run.id}, mode={run.settings.mode}, url={run.url}")

    # Setup log directory: logs/sweep/<exp_id>/<method>/<sweep_id>/<run_id>/
    sweep_id = wandb.run.sweep_id or "no_sweep"
    log_dir = config_temp.get_logs_dir() / "sweep" / config_temp.experiment_id / args.method / sweep_id / wandb.run.id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "main.log"

    # Redirect stdout/stderr to log file
    sys.stdout = open(log_file, 'w', buffering=1)
    sys.stderr = sys.stdout

    # Write buffered content to log file
    buffered_content = buffer.getvalue()
    if buffered_content:
        print(buffered_content, end='')

    print(f"Sweep agent {wandb.run.id} starting")
    print(f"Logs: {log_dir}")

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

    # Override num_episodes from wandb sweep or CLI
    num_episodes = wandb.config.get('num_episodes', args.num_episodes)
    if num_episodes is not None:
        config.value_estimators.training.episode_subsets = [num_episodes]

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

    # Override preprocess_fraction for LeastSquares methods from wandb sweep or CLI
    preprocess_fraction = wandb.config.get('preprocess_fraction', args.preprocess_fraction)
    if preprocess_fraction is not None and hasattr(method_config, 'preprocess_fraction'):
        method_config.preprocess_fraction = preprocess_fraction

    # Override n_initializations from wandb sweep or CLI
    n_initializations = wandb.config.get('n_initializations', args.n_initializations)
    if n_initializations is not None:
        method_config.n_initializations = n_initializations
    else:
        # Default to 1 if not specified
        method_config.n_initializations = 1

    # Pre-declare all metrics for multiple initializations so wandb creates panels for them
    if method_config.n_initializations > 1:
        print(f"Pre-declaring wandb metrics for {method_config.n_initializations} initializations")
        for init_idx in range(method_config.n_initializations):
            suffix = f"_{init_idx}"
            # Define train metrics
            wandb.define_metric(f"train{suffix}/loss")
            wandb.define_metric(f"train{suffix}/mse")
            wandb.define_metric(f"train{suffix}/mae")
            wandb.define_metric(f"train{suffix}/mc_loss_train")
            wandb.define_metric(f"train{suffix}/best_mc_loss")
            wandb.define_metric(f"train{suffix}/mean_value")
            wandb.define_metric(f"train{suffix}/mean_target")
            # Define validation metrics
            wandb.define_metric(f"val{suffix}/mc_loss")
            wandb.define_metric(f"val{suffix}/min_mc_loss")
            # Define final metrics
            wandb.define_metric(f"final{suffix}/best_mc_loss")
            wandb.define_metric(f"final{suffix}/mc_loss_train")
            wandb.define_metric(f"final{suffix}/mc_loss_val")

    # Setup paths
    batch_path = config.get_data_dir() / "batch_tuning.npz"
    output_dir = config.get_estimators_dir() / "sweeps" / args.method / wandb.run.id

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Call core training
    train_estimator(
        config=config,
        method_config=method_config,
        batch_path=batch_path,
        output_dir=output_dir,
        batch_name="tuning",
        overwrite=True,
        use_wandb=True,
        sweep_mode=True,
        parallel_inits=args.parallel_inits,
        n_jobs=args.n_jobs,
        log_dir=log_dir
    )

    # Properly finish wandb run to ensure all data is synced and run is marked complete
    print(f"\nFinishing wandb run...")

    # Get run directory for offline sync
    if args.wandb_mode == "offline":
        run_dir = str(Path(wandb.run.dir).parent)
        print(f"[DEBUG] Wandb run directory: {run_dir}")

    wandb.finish()

    # Sync offline run to W&B
    if args.wandb_mode == "offline":
        print(f"Syncing offline run to W&B...")
        print(f"[DEBUG] Syncing directory: {run_dir}")
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
    else:
        print(f"Wandb run finished and synced")

    # Print any files still open at exit
    from src.debug_file_tracking import print_open_files
    print_open_files()


if __name__ == "__main__":
    main()
