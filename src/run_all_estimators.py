"""Orchestrate training of all value estimators across all batches."""

import argparse
import subprocess
from pathlib import Path

from src.config import ExperimentConfig


def run_sequential(config: ExperimentConfig, config_path: Path):
    """Run all training jobs sequentially."""
    methods = config.value_estimators.methods
    n_batches = config.data_generation.n_batches

    print(f"Running {len(methods) * n_batches} jobs sequentially")
    print(f"Methods: {methods}")
    print(f"Batches: {n_batches}\n")

    for method in methods:
        for batch_idx in range(n_batches):
            print(f"\n{'='*60}")
            print(f"Training: method={method} batch={batch_idx}")
            print(f"{'='*60}\n")

            subprocess.run([
                "python", "-m", "src.train_estimator",
                "--config", str(config_path),
                "--method", method,
                "--batch-idx", str(batch_idx)
            ], check=True)

    print(f"\n{'='*60}")
    print("All jobs completed!")
    print(f"{'='*60}\n")


def run_parallel(config: ExperimentConfig, config_path: Path):
    """Run all training jobs in parallel using bash script."""
    methods = ','.join(config.value_estimators.methods)
    n_batches = str(config.data_generation.n_batches)

    script_path = Path(__file__).parent.parent / "run_parallel_estimators.sh"

    subprocess.run([
        str(script_path),
        str(config_path),
        methods,
        n_batches
    ], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run all value estimator training jobs"
    )
    parser.add_argument("--config", type=Path, required=True,
                       help="Path to config YAML file")
    parser.add_argument("--mode", choices=['sequential', 'parallel'],
                       default='sequential',
                       help="sequential: run jobs one by one, "
                            "parallel: run jobs in parallel across GPUs")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    if args.mode == 'sequential':
        run_sequential(config, args.config)
    elif args.mode == 'parallel':
        run_parallel(config, args.config)


if __name__ == "__main__":
    main()
