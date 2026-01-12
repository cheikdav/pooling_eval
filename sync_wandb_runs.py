#!/usr/bin/env python3
"""Sync all unsynced wandb offline runs.

This script finds all wandb offline run directories across all experiments
and syncs them to W&B. It skips runs that are already synced.
"""

import argparse
import subprocess
from pathlib import Path


def is_run_synced(run_dir: Path) -> bool:
    """Check if a wandb run has been synced.

    A run is considered synced if there's a .synced file matching the run name in the parent directory.
    wandb sync creates these files automatically.
    """
    parent_dir = run_dir.parent

    for file in run_dir.glob(f"*.synced"):
        return True
    return False


def sync_run(run_dir: Path, dry_run: bool = False) -> bool:
    """Sync a single wandb run directory.

    Args:
        run_dir: Path to the wandb run directory
        dry_run: If True, only print what would be synced

    Returns:
        True if sync succeeded, False otherwise
    """
    if dry_run:
        print(f"  [DRY RUN] Would sync: {run_dir}")
        return True

    try:
        result = subprocess.run(
            ["wandb", "sync", str(run_dir)],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✓ Synced: {run_dir.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {run_dir.name}")
        if e.stderr:
            print(f"    Error: {e.stderr.strip()}")
        return False


def find_wandb_runs(experiments_dir: Path) -> list[Path]:
    """Find all wandb offline run directories.

    Returns:
        List of paths to wandb run directories
    """
    runs = []

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        wandb_runs_dir = exp_dir / "wandb_offline" / "wandb"
        if not wandb_runs_dir.exists():
            continue

        for item in wandb_runs_dir.iterdir():
            if item.is_dir() and (item.name.startswith("run-") or item.name.startswith("offline-run-")):
                runs.append(item)

    return sorted(runs)


def main():
    parser = argparse.ArgumentParser(description="Sync unsynced wandb offline runs")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Path to experiments directory (default: experiments)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without actually syncing"
    )
    parser.add_argument(
        "--sync-all",
        action="store_true",
        help="Sync all runs, even those already synced"
    )
    args = parser.parse_args()

    if not args.experiments_dir.exists():
        print(f"Error: Experiments directory not found: {args.experiments_dir}")
        return 1

    print(f"Searching for wandb runs in {args.experiments_dir}...")
    runs = find_wandb_runs(args.experiments_dir)

    if not runs:
        print("No wandb offline runs found.")
        return 0

    print(f"Found {len(runs)} wandb run(s)\n")

    unsynced_runs = [r for r in runs if not is_run_synced(r)]
    synced_runs = [r for r in runs if is_run_synced(r)]

    print(f"Status:")
    print(f"  Already synced: {len(synced_runs)}")
    print(f"  Need syncing: {len(unsynced_runs)}")

    if args.sync_all:
        runs_to_sync = runs
        print(f"\n--sync-all flag set, syncing all {len(runs)} run(s)...\n")
    else:
        runs_to_sync = unsynced_runs
        if unsynced_runs:
            print(f"\nSyncing {len(unsynced_runs)} unsynced run(s)...\n")
        else:
            print("\nAll runs already synced!")
            return 0

    success_count = 0
    fail_count = 0

    for run_dir in runs_to_sync:
        experiment_name = run_dir.parent.parent.name
        print(f"[{experiment_name}] {run_dir.name}")

        if sync_run(run_dir, dry_run=args.dry_run):
            success_count += 1
        else:
            fail_count += 1


    print(f"\n{'='*60}")
    print(f"Summary:")
    if args.dry_run:
        print(f"  [DRY RUN] Would sync {success_count} run(s)")
    else:
        print(f"  Successfully synced: {success_count}")
        print(f"  Failed: {fail_count}")
    print(f"{'='*60}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
