"""Orchestrate training of all value estimators across all batches."""

import argparse
import re
import subprocess
from pathlib import Path

from src.config import ExperimentConfig

QSUB = "/opt/n1ge/bin/lx24-amd64/qsub"


def run_sequential(config: ExperimentConfig, config_path: Path, overwrite: bool):
    """Run all training jobs sequentially."""
    method_configs = config.value_estimators.method_configs
    n_batches = config.data_generation.n_batches

    print(f"Running {len(method_configs) * n_batches} jobs sequentially")
    print(f"Methods: {[mc.name for mc in method_configs]}")
    print(f"Batches: {n_batches}")
    print(f"Overwrite: {overwrite}\n")

    # Generate timestamp for this training session (shared across all jobs)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training session timestamp: {timestamp}\n")

    for method_config in method_configs:
        for batch_idx in range(n_batches):
            batch_name = str(batch_idx)
            print(f"\n{'='*60}")
            print(f"Training: method={method_config.name} batch={batch_name}")
            print(f"{'='*60}\n")

            overwrite_flag = "--overwrite" if overwrite else "--no-overwrite"
            subprocess.run([
                "python", "-m", "src.train_estimator",
                "--config", str(config_path),
                "--method", method_config.name,
                "--batch-idx", str(batch_idx),
                overwrite_flag,
                "--timestamp", timestamp
            ], check=True)

    print(f"\n{'='*60}")
    print("All jobs completed!")
    print(f"{'='*60}\n")


def run_parallel(config: ExperimentConfig, config_path: Path, overwrite: bool):
    """Run all training jobs in parallel using bash script."""
    method_names = ','.join([mc.name for mc in config.value_estimators.method_configs])
    n_batches = str(config.data_generation.n_batches)
    overwrite_flag = "true" if overwrite else "false"

    # Generate timestamp for this training session
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training session timestamp: {timestamp}\n")

    script_path = Path(__file__).parent.parent / "run_parallel_estimators.sh"

    subprocess.run([
        str(script_path),
        str(config_path),
        method_names,
        n_batches,
        overwrite_flag,
        timestamp
    ], check=True)


def run_cluster(config: ExperimentConfig, config_path: Path, overwrite: bool, memory: str = "8g", ncpus: int = 1, n_jobs: int = None, max_concurrent: int = None):
    """Run all training jobs on cluster using SGE array jobs via qsub.

    Submits one array job per method, where each array task trains one batch.
    SGE_TASK_ID is used directly as the batch index (1-indexed, converted to 0-indexed in train_estimator).
    """
    method_configs = config.value_estimators.method_configs
    n_batches = config.data_generation.n_batches

    print(f"Submitting {len(method_configs)} array jobs to cluster")
    print(f"Methods: {[mc.name for mc in method_configs]}")
    print(f"Batches per method: {n_batches}")
    print(f"Memory per job: {memory}")
    print(f"CPUs per job: {ncpus}")
    print(f"Parallel episode counts per job: {n_jobs or 1}")
    print(f"Overwrite: {overwrite}")
    if max_concurrent:
        print(f"Max concurrent per method: {max_concurrent}")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training session timestamp: {timestamp}")
    print()

    array_spec = f"1-{n_batches}"

    # mem per slot (qsub multiplies h_vmem by ncpus)
    if ncpus > 1:
        mem_match = re.match(r'^([0-9.]+)(\D*)$', memory)
        if mem_match:
            per_slot = float(mem_match.group(1)) / ncpus
            mem_per_slot = f"{per_slot:.1f}{mem_match.group(2) or 'g'}"
        else:
            mem_per_slot = memory
    else:
        mem_per_slot = memory

    job_ids = []
    overwrite_flag = "--overwrite" if overwrite else "--no-overwrite"
    for method_config in method_configs:
        print(f"Submitting array job for {method_config.name} with array={array_spec}")

        qsub_args = [
            QSUB, "-V", "-cwd", "-b", "y",
            "-q", "debian.q",
            "-N", f"est_{method_config.name}",
            "-pe", "threaded", str(ncpus),
            "-binding", f"linear:{ncpus}",
            "-l", f"h_vmem={mem_per_slot}",
            "-t", array_spec,
        ]
        if max_concurrent:
            qsub_args.extend(["-tc", str(max_concurrent)])

        qsub_args.extend([
            "uv", "run", "-m", "src.train_estimator",
            "--config", str(config_path.absolute()),
            "--method", method_config.name,
            overwrite_flag,
            "--timestamp", timestamp,
            *(["--n-jobs", str(n_jobs)] if n_jobs and n_jobs > 1 else [])
        ])

        result = subprocess.run(qsub_args, check=True, capture_output=True, text=True)
        output = result.stdout + result.stderr

        # Parse job ID
        match = re.search(r'Your job(?:-array)? (\d+)', output)
        if match:
            job_id = match.group(1)
            job_ids.append(job_id)
            print(f"  Submitted: {method_config.name} (job {job_id})")
        else:
            print(f"  Submitted: {method_config.name}")
            for line in output.split('\n'):
                if line.strip():
                    print(f"  {line.strip()}")

    print(f"\n{len(method_configs)} array jobs submitted successfully!")
    print(f"Total tasks: {len(method_configs) * n_batches}")
    print(f"Job IDs: {', '.join(job_ids)}")
    print(f"\nMonitor with: qstat")
    return job_ids


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

    overwrite_group = parser.add_mutually_exclusive_group(required=True)
    overwrite_group.add_argument("--overwrite", dest="overwrite", action="store_true",
                       help="Overwrite existing models")
    overwrite_group.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                       help="Skip training if model already exists")
    parser.add_argument("--grid-mem", type=str, default="8g",
                       help="Memory per job for cluster mode (e.g., '8g', '16g')")
    parser.add_argument("--grid-ncpus", type=int, default=None,
                       help="Number of CPUs per job for cluster mode (default: number of episode subsets)")
    parser.add_argument("--max-concurrent", type=int, default=None,
                       help="Maximum number of concurrent jobs for cluster mode")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    n_episode_subsets = len(config.value_estimators.training.episode_subsets or [])

    if args.mode == 'sequential':
        run_sequential(config, args.config, args.overwrite)
    elif args.mode == 'parallel':
        run_parallel(config, args.config, args.overwrite)
    elif args.mode == 'cluster':
        ncpus = args.grid_ncpus if args.grid_ncpus is not None else max(n_episode_subsets, 1)
        run_cluster(config, args.config, args.overwrite, memory=args.grid_mem,
                   ncpus=ncpus, n_jobs=ncpus, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
