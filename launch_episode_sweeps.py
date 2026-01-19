#!/usr/bin/env python3
"""Launch separate W&B sweeps for each episode count.

Creates one sweep per episode count, each varying learning rate and batch size.
"""

import argparse
import yaml
from pathlib import Path
import subprocess
import sys


def create_sweep_config(base_config: dict, method: str, episode_count: int) -> dict:
    """Create a sweep config for a specific episode count.

    Args:
        base_config: Base sweep configuration to use as template
        method: Method name (monte_carlo, dqn, etc.)
        episode_count: Number of episodes for this sweep

    Returns:
        Modified sweep configuration
    """
    import copy
    config = copy.deepcopy(base_config)

    # Fix the episode count for this sweep
    config['parameters']['num-episodes'] = {'value': episode_count}

    # Update run cap if needed (optional, can be adjusted)
    if 'run_cap' not in config:
        config['run_cap'] = 30

    return config


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

        # Extract sweep ID from output
        for line in result.stdout.split('\n'):
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
        help="Episode counts to sweep over (default: use values from sweep config)"
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
    args = parser.parse_args()

    # Load base sweep config
    if args.sweep_config is None:
        args.sweep_config = Path(f"configs/sweeps/sweep_{args.method}.yaml")

    if not args.sweep_config.exists():
        print(f"Error: Sweep config not found: {args.sweep_config}")
        sys.exit(1)

    with open(args.sweep_config, 'r') as f:
        base_config = yaml.safe_load(f)

    # If no episodes specified, use values from sweep config
    if args.episodes is None:
        if 'num-episodes' in base_config.get('parameters', {}):
            num_episodes_param = base_config['parameters']['num-episodes']
            if 'values' in num_episodes_param:
                args.episodes = num_episodes_param['values']
            else:
                print(f"Error: num-episodes parameter in sweep config must have 'values' field")
                sys.exit(1)
        else:
            print(f"Error: No --episodes specified and num-episodes not found in sweep config")
            sys.exit(1)

    print(f"Launching {len(args.episodes)} sweeps for {args.method}")
    print(f"Episode counts: {args.episodes}\n")

    sweep_ids = []

    for episode_count in args.episodes:
        sweep_config = create_sweep_config(base_config, args.method, episode_count)

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
        print("All sweeps created! Run agents with:")
        print(f"{'='*60}\n")
        for episode_count, sweep_id in sweep_ids:
            print(f"# {episode_count} episodes:")
            print(f"wandb agent {sweep_id}\n")


if __name__ == "__main__":
    main()
