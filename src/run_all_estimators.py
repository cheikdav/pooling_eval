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


def run_cluster(config: ExperimentConfig, config_path: Path, memory: str = "8g", max_concurrent: int = None):
    """Run all training jobs on cluster using SGE array jobs.

    Submits one array job per method, where each array task trains one batch.
    SGE_TASK_ID is used directly as the batch index (1-indexed, converted to 0-indexed in train_estimator).

    Args:
        config: Experiment configuration
        config_path: Path to config YAML file
        memory: Memory per job (e.g., "8g", "16g")
        max_concurrent: Maximum number of jobs to run concurrently per method (optional)
    """
    methods = config.value_estimators.methods
    n_batches = config.data_generation.n_batches

    print(f"Submitting {len(methods)} array jobs to cluster")
    print(f"Methods: {methods}")
    print(f"Batches per method: {n_batches}")
    print(f"Memory per job: {memory}")
    if max_concurrent:
        print(f"Max concurrent per method: {max_concurrent}")
    print()

    # Build array specification
    if max_concurrent:
        array_spec = f"1-{n_batches}/{max_concurrent}"
    else:
        array_spec = f"1-{n_batches}"

    # Submit one array job per method
    job_ids = []
    for method in methods:
        print(f"Submitting array job for {method} with array={array_spec}")

        result = subprocess.run([
            "grid_run",
            "--grid_submit=batch",
            f"--grid_array={array_spec}",
            f"--grid_mem={memory}",
            "uv run", "-m", "src.train_estimator",
            "--config", str(config_path.absolute()),
            "--method", method
        ], check=True, capture_output=True, text=True)

        # Try to extract job ID from output
        output = result.stdout + result.stderr
        print(f"  Submitted: {method}")
        if "job" in output.lower():
            # Print relevant lines containing job info
            for line in output.split('\n'):
                if 'job' in line.lower():
                    print(f"  {line.strip()}")
                    job_ids.append(line.strip())

    print(f"\n{len(methods)} array jobs submitted successfully!")
    print(f"Total tasks: {len(methods) * n_batches}")
    print(f"\nMonitor with: qstat")
    print(f"View logs: *.o* and *.e* files for each job")


def main():
    parser = argparse.ArgumentParser(
        description="Run all value estimator training jobs"
    )
    parser.add_argument("--config", type=Path, required=True,
                       help="Path to config YAML file")
    parser.add_argument("--mode", choices=['sequential', 'parallel', 'cluster'],
                       default='sequential',
                       help="sequential: run jobs one by one, "
                            "parallel: run jobs in parallel across GPUs, "
                            "cluster: submit jobs to SGE cluster")
    parser.add_argument("--grid-mem", type=str, default="8g",
                       help="Memory per job for cluster mode (e.g., '8g', '16g')")
    parser.add_argument("--max-concurrent", type=int, default=None,
                       help="Maximum number of concurrent jobs for cluster mode")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    if args.mode == 'sequential':
        run_sequential(config, args.config)
    elif args.mode == 'parallel':
        run_parallel(config, args.config)
    elif args.mode == 'cluster':
        run_cluster(config, args.config, memory=args.grid_mem,
                   max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
