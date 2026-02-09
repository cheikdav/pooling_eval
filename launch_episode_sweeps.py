#!/usr/bin/env python3
"""Launch W&B sweep for hyperparameter tuning.

New approach: Each hyperparameter set trains on all episode counts in a single run.
"""

import argparse
import yaml
from pathlib import Path
import subprocess
import sys
import wandb
import multiprocessing

from src.config import ExperimentConfig


def create_sweep_config(base_config: dict, config_path: str, exp_config: ExperimentConfig) -> dict:
    """Create sweep config with injected experiment config path and project name.

    Args:
        base_config: Base sweep configuration to use as template
        config_path: Path to experiment config file
        exp_config: Loaded experiment config (for environment name)

    Returns:
        Modified sweep configuration
    """
    import copy
    config = copy.deepcopy(base_config)

    if 'project' not in config:
        config['project'] = exp_config.logging.get_project_name(exp_config.environment.name)

    config['parameters']['config'] = {'value': config_path}

    if 'run_cap' not in config:
        config['run_cap'] = 30

    return config


def run_agent(sweep_id: str, config_path: Path, method: str, agent_idx: int):
    """Run a wandb agent in a subprocess (called via multiprocessing)."""
    import os

    config_temp = ExperimentConfig.from_yaml(config_path)

    sweep_id_hash = sweep_id.split('/')[-1] if '/' in sweep_id else sweep_id

    log_dir = config_temp.get_logs_dir() / "sweep" / config_temp.experiment_id / method / sweep_id_hash
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"agent_{agent_idx}.log"

    sys.stdout = open(log_file, 'w', buffering=1)
    sys.stderr = sys.stdout

    print(f"Sweep agent {agent_idx} starting")
    print(f"Sweep ID: {sweep_id}")
    print(f"Logging to: {log_file}\n")

    os.environ['WANDB_AGENT_IDX'] = str(agent_idx)

    wandb.agent(sweep_id)


def launch_sweep(config_dict: dict, method: str) -> str:
    """Launch a W&B sweep and return the sweep ID.

    Args:
        config_dict: Sweep configuration dictionary
        method: Method name

    Returns:
        Sweep ID
    """
    temp_config_path = Path(f".sweep_{method}.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_dict, f)

    try:
        result = subprocess.run(
            ['wandb', 'sweep', str(temp_config_path)],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            print(f"Error launching sweep (exit code {result.returncode}):")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return None

        output = result.stderr if result.stderr else result.stdout
        for line in output.split('\n'):
            if 'wandb agent' in line:
                sweep_id = line.split()[-1]
                return sweep_id

        print("Warning: Could not extract sweep ID from wandb output:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

    except Exception as e:
        print(f"Exception while launching sweep: {e}")
        return None

    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Launch W&B sweep for hyperparameter tuning")
    parser.add_argument("--config", type=Path, required=True,
                       help="Experiment config file")
    parser.add_argument("--method", type=str, required=True,
                       help="Method name (monte_carlo, dqn, etc.)")
    parser.add_argument("--sweep-config", type=Path, default=None,
                       help="Base sweep config file (default: configs/sweeps/sweep_<method>.yaml)")
    parser.add_argument("--dry-run", action='store_true',
                       help="Print sweep config without launching")
    parser.add_argument("--launch-agents", type=int, default=1,
                       help="Number of agents to launch (default: 1)")
    args = parser.parse_args()

    if args.sweep_config is None:
        args.sweep_config = Path(f"configs/sweeps/sweep_{args.method}.yaml")

    if not args.sweep_config.exists():
        print(f"Error: Sweep config not found: {args.sweep_config}")
        sys.exit(1)

    with open(args.sweep_config, 'r') as f:
        base_config = yaml.safe_load(f)

    if not args.config.exists():
        print(f"Error: Experiment config not found: {args.config}")
        sys.exit(1)

    exp_config = ExperimentConfig.from_yaml(args.config)

    sweep_config = create_sweep_config(base_config, str(args.config), exp_config)

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"Sweep config for {args.method}:")
        print(f"{'='*60}")
        print(yaml.dump(sweep_config))
        return

    print(f"Creating sweep for {args.method}...")
    print(f"Episode counts (from config): {exp_config.value_estimators.training.episode_subsets}")

    sweep_id = launch_sweep(sweep_config, args.method)

    if not sweep_id:
        print(f"Failed to create sweep")
        sys.exit(1)

    print(f"✓ Sweep created: {sweep_id}")

    if args.launch_agents > 0:
        print(f"\nLaunching {args.launch_agents} agent(s)...")

        processes = []
        for agent_idx in range(args.launch_agents):
            process = multiprocessing.Process(
                target=run_agent,
                args=(sweep_id, args.config, args.method, agent_idx),
                daemon=False
            )
            process.start()
            processes.append(process)

            sweep_id_hash = sweep_id.split('/')[-1] if '/' in sweep_id else sweep_id
            log_file = exp_config.get_logs_dir() / "sweep" / exp_config.experiment_id / args.method / sweep_id_hash / f"agent_{agent_idx}.log"
            print(f"  Agent {agent_idx} started (PID: {process.pid}, log: {log_file})")

        print(f"\n{'='*60}")
        print(f"All agents launched! Monitor progress:")
        print(f"{'='*60}\n")
        sweep_id_hash = sweep_id.split('/')[-1] if '/' in sweep_id else sweep_id
        for agent_idx in range(args.launch_agents):
            log_file = exp_config.get_logs_dir() / "sweep" / exp_config.experiment_id / args.method / sweep_id_hash / f"agent_{agent_idx}.log"
            print(f"tail -f {log_file}")
        print(f"\nOr view sweeps at: https://wandb.ai")
        print(f"\nTo stop agents, use: pkill -f 'wandb agent'")

        for process in processes:
            process.join()
    else:
        print(f"\nTo run agents manually:")
        print(f"wandb agent {sweep_id}")


if __name__ == "__main__":
    main()
