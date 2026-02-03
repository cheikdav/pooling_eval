#!/usr/bin/env python3
"""Launch separate W&B sweeps for each episode count.

Creates one sweep per episode count, each varying learning rate and batch size.
"""

import argparse
import yaml
from pathlib import Path
import subprocess
import sys
import wandb
import multiprocessing

from src.config import ExperimentConfig


def create_sweep_config(base_config: dict, method: str, episode_count: int, config_path: str) -> dict:
    """Create a sweep config for a specific episode count.

    Args:
        base_config: Base sweep configuration to use as template
        method: Method name (monte_carlo, dqn, etc.)
        episode_count: Number of episodes for this sweep
        config_path: Path to experiment config file

    Returns:
        Modified sweep configuration
    """
    import copy
    config = copy.deepcopy(base_config)

    # Set the experiment config path
    config['parameters']['config'] = {'value': config_path}

    # Fix the episode count for this sweep
    config['parameters']['num-episodes'] = {'value': episode_count}

    # Update run cap if needed (optional, can be adjusted)
    if 'run_cap' not in config:
        config['run_cap'] = 30

    return config


def run_agent(sweep_id: str, config_path: Path, method: str, episode_count: int, agent_idx: int):
    """Run a wandb agent in a subprocess (called via multiprocessing)."""
    import os

    # Load config early to get paths
    config_temp = ExperimentConfig.from_yaml(config_path)

    # Extract sweep ID hash from full sweep path (e.g., "user/project/abc123" -> "abc123")
    sweep_id_hash = sweep_id.split('/')[-1] if '/' in sweep_id else sweep_id

    # Redirect stdout/stderr to log file: logs/sweep/<exp_id>/<method>/<sweep_id>/agent_<episodes>ep_<idx>.log
    log_dir = config_temp.get_logs_dir() / "sweep" / config_temp.experiment_id / method / sweep_id_hash
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"agent_{episode_count}ep_{agent_idx}.log"

    sys.stdout = open(log_file, 'w', buffering=1)
    sys.stderr = sys.stdout

    print(f"Sweep agent starting for {episode_count} episodes (agent {agent_idx})")
    print(f"Sweep ID: {sweep_id}")
    print(f"Logging to: {log_file}\n")

    # Set environment variable so tune_hyperparameters.py can identify which agent it is
    os.environ['WANDB_AGENT_IDX'] = str(agent_idx)

    # Run the agent using wandb Python API
    wandb.agent(sweep_id)


def launch_sweep(config_dict: dict, method: str, episode_count: int) -> str:
    """Launch a W&B sweep and return the sweep ID.

    Args:
        config_dict: Sweep configuration dictionary
        method: Method name
        episode_count: Number of episodes

    Returns:
        Sweep ID
    """
    # Write config to temporary file
    temp_config_path = Path(f".sweep_{method}_{episode_count}ep.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_dict, f)

    try:
        # Launch sweep
        result = subprocess.run(
            ['wandb', 'sweep', str(temp_config_path)],
            capture_output=True,
            text=True,
            check=False  # Don't raise exception, handle errors manually
        )

        if result.returncode != 0:
            print(f"Error launching sweep (exit code {result.returncode}):")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return None

        # Extract sweep ID from output (wandb writes to stderr)
        output = result.stderr if result.stderr else result.stdout
        for line in output.split('\n'):
            if 'wandb agent' in line:
                sweep_id = line.split()[-1]
                return sweep_id

        # Fallback: couldn't find sweep ID in expected format
        print("Warning: Could not extract sweep ID from wandb output:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

    except Exception as e:
        print(f"Exception while launching sweep: {e}")
        return None

    finally:
        # Clean up temp file
        if temp_config_path.exists():
            temp_config_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Launch separate W&B sweeps for each episode count"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment config file (e.g., configs/config_humanoid.yaml)"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Method name (monte_carlo, dqn, etc.)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs='+',
        default=None,
        help="Episode counts to sweep over (default: use values from experiment config)"
    )
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=None,
        help="Base sweep config file (default: configs/sweeps/sweep_<method>.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action='store_true',
        help="Print sweep configs without launching"
    )
    parser.add_argument(
        "--launch-agents",
        type=int,
        default=0,
        help="Number of agents to launch per sweep (default: 0, just create sweeps)"
    )
    args = parser.parse_args()

    # Load base sweep config
    if args.sweep_config is None:
        args.sweep_config = Path(f"configs/sweeps/sweep_{args.method}.yaml")

    if not args.sweep_config.exists():
        print(f"Error: Sweep config not found: {args.sweep_config}")
        sys.exit(1)

    with open(args.sweep_config, 'r') as f:
        base_config = yaml.safe_load(f)

    # Verify experiment config exists
    if not args.config.exists():
        print(f"Error: Experiment config not found: {args.config}")
        sys.exit(1)

    # If no episodes specified, read from experiment config
    if args.episodes is None:
        exp_config = ExperimentConfig.from_yaml(args.config)
        args.episodes = exp_config.value_estimators.training.episode_subsets
        print(f"Using episode counts from {args.config}: {args.episodes}")

    print(f"Launching {len(args.episodes)} sweeps for {args.method}")
    print(f"Episode counts: {args.episodes}\n")

    sweep_ids = []

    for episode_count in args.episodes:
        sweep_config = create_sweep_config(base_config, args.method, episode_count, str(args.config))

        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"Sweep config for {episode_count} episodes:")
            print(f"{'='*60}")
            print(yaml.dump(sweep_config))
            continue

        print(f"Creating sweep for {episode_count} episodes...")
        sweep_id = launch_sweep(sweep_config, args.method, episode_count)

        if sweep_id:
            sweep_ids.append((episode_count, sweep_id))
            print(f"  ✓ Sweep created: {sweep_id}")
        else:
            print(f"  ✗ Failed to create sweep for {episode_count} episodes")

    if not args.dry_run and sweep_ids:
        print(f"\n{'='*60}")

        if args.launch_agents > 0:
            print(f"Launching {args.launch_agents} agent(s) per sweep...")
            print(f"{'='*60}\n")

            # Load config to get log directory path for display
            exp_config = ExperimentConfig.from_yaml(args.config)

            agent_processes = []
            for episode_count, sweep_id in sweep_ids:
                print(f"Launching {args.launch_agents} agent(s) for {episode_count} episodes (sweep: {sweep_id})")
                for agent_idx in range(args.launch_agents):
                    # Use multiprocessing to run agent in separate process
                    # This avoids all subprocess/shell/stdin issues
                    process = multiprocessing.Process(
                        target=run_agent,
                        args=(sweep_id, args.config, args.method, episode_count, agent_idx),
                        daemon=False  # Daemonize so it doesn't block parent exit
                    )
                    process.start()

                    # Calculate log path for display
                    sweep_id_hash = sweep_id.split('/')[-1] if '/' in sweep_id else sweep_id
                    log_file = exp_config.get_logs_dir() / "sweep" / exp_config.experiment_id / args.method / sweep_id_hash / f"agent_{episode_count}ep_{agent_idx}.log"
                    agent_processes.append((episode_count, agent_idx, sweep_id, log_file))
                    print(f"  Agent {agent_idx+1} started (PID: {process.pid}, log: {log_file})")

            print(f"\n{'='*60}")
            print(f"All agents launched! Monitor progress:")
            print(f"{'='*60}\n")
            for episode_count, agent_idx, sweep_id, log_file in agent_processes:
                print(f"tail -f {log_file}  # {episode_count}ep agent {agent_idx+1}")
            print(f"\nOr view sweeps at: https://wandb.ai")
            print(f"\nTo stop agents, use: pkill -f 'wandb agent'")

        else:
            print("All sweeps created! Run agents with:")
            print(f"{'='*60}\n")
            for episode_count, sweep_id in sweep_ids:
                print(f"# {episode_count} episodes:")
                print(f"wandb agent {sweep_id}\n")


if __name__ == "__main__":
    main()
