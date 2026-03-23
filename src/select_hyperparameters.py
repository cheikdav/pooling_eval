"""Select best hyperparameters from sweep results and write to method config.

Reads sweep_results.csv, picks the best hyperparameter set using
mean-normalized-loss ranking, and updates the method's YAML config file.

Can be used as CLI or imported programmatically by the pipeline.

Usage:
    uv run -m src.select_hyperparameters --config configs/humanoid/config.yaml --method monte_carlo
    uv run -m src.select_hyperparameters --config configs/humanoid/config.yaml --method monte_carlo --dry-run
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from src.config import ExperimentConfig
from src.tune_hyperparameters import HYPERPARAM_KEYS


# Which YAML keys each hyperparam maps to (top-level vs nested under feature_extractor)
YAML_KEY_MAP = {
    'learning_rate': ('top', 'learning_rate'),
    'batch_size': ('top', 'batch_size'),
    'target_update_rate': ('top', 'target_update_rate'),
    'ridge_lambda': ('top', 'ridge_lambda'),
    'n_components': ('top', 'n_components'),
    'rbf_n_components': ('nested', 'feature_extractor', 'n_components'),
    'rbf_gamma': ('nested', 'feature_extractor', 'gamma'),
}

INT_FIELDS = {'batch_size', 'n_components', 'rbf_n_components'}

# Minimum number of sweep runs for reliable selection
MIN_RUNS_WARNING = 5
MIN_RUNS_ERROR = 2


@dataclass
class SelectionResult:
    """Result of hyperparameter selection, used by the pipeline."""
    success: bool
    best_row: Optional[dict] = None
    score: float = float('inf')
    n_runs: int = 0
    changes: dict = field(default_factory=dict)  # hp_key -> (old, new)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    ep_cols: list = field(default_factory=list)


def load_sweep_results(csv_path: Path) -> list[dict]:
    """Load sweep_results.csv into a list of row dicts with parsed numerics."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v == '' or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def get_episode_count_columns(rows: list[dict]) -> list[str]:
    """Extract episode count column names sorted by episode count."""
    ep_cols = [k for k in rows[0] if k.startswith('best_mc_loss_') and k.endswith('ep')]
    ep_cols.sort(key=lambda c: int(c.replace('best_mc_loss_', '').replace('ep', '')))
    return ep_cols


def _filter_valid_rows(rows: list[dict], ep_cols: list[str]) -> tuple[list[dict], list[str]]:
    """Filter out rows with NaN/Inf/missing losses. Returns (valid_rows, warnings)."""
    warnings = []
    valid = []
    for row in rows:
        losses = [row.get(col) for col in ep_cols]
        if any(v is None for v in losses):
            warnings.append(f"Run {row.get('run_id', '?')}: missing loss values, skipped")
            continue
        if any(not np.isfinite(v) for v in losses):
            warnings.append(f"Run {row.get('run_id', '?')}: NaN/Inf loss values, skipped")
            continue
        valid.append(row)

    # Deduplicate by run_id (keep last occurrence, which is the most recent)
    seen = {}
    for row in valid:
        seen[row.get('run_id', id(row))] = row
    deduped = list(seen.values())
    if len(deduped) < len(valid):
        warnings.append(f"Removed {len(valid) - len(deduped)} duplicate run_id entries")

    return deduped, warnings


def select_best(rows: list[dict], ep_cols: list[str]) -> tuple[dict, list[dict]]:
    """Select best hyperparameter set using mean-normalized-loss.

    For each episode count, divides each run's loss by the best loss seen
    for that count. Then picks the run with the lowest mean normalized score.
    Best possible score is 1.0 (best at every episode count).

    Returns (best_row, all_rows_with_scores) sorted by score.
    """
    loss_matrix = np.array([[row[col] for col in ep_cols] for row in rows])

    col_mins = loss_matrix.min(axis=0)
    col_mins = np.where(col_mins == 0, 1e-12, col_mins)

    normalized = loss_matrix / col_mins
    scores = normalized.mean(axis=1)

    scored_rows = []
    for i, row in enumerate(rows):
        row_copy = dict(row)
        row_copy['_score'] = scores[i]
        row_copy['_normalized'] = {col: normalized[i, j] for j, col in enumerate(ep_cols)}
        scored_rows.append(row_copy)

    scored_rows.sort(key=lambda r: r['_score'])
    return scored_rows[0], scored_rows


def _diagnose_results(scored_rows: list[dict], ep_cols: list[str]) -> list[str]:
    """Generate warnings about suspicious sweep results."""
    warnings = []

    if not scored_rows:
        return warnings

    best = scored_rows[0]
    worst = scored_rows[-1]

    # Check if best run is much worse than 1.0 (all episode counts)
    if best['_score'] > 1.5:
        warnings.append(
            f"Best score is {best['_score']:.2f} (ideal=1.0). "
            "No single hyperparameter set is good across all episode counts."
        )

    # Check if there's a huge gap between best and second-best
    if len(scored_rows) >= 2:
        second = scored_rows[1]
        gap = second['_score'] - best['_score']
        if gap > 0.5:
            warnings.append(
                f"Large gap between #1 (score={best['_score']:.3f}) and "
                f"#2 (score={second['_score']:.3f}). Best run may be an outlier."
            )

    # Check if any episode count has extreme normalized loss even for the best run
    for col in ep_cols:
        norm = best['_normalized'][col]
        if norm > 2.0:
            ep = col.replace('best_mc_loss_', '').replace('ep', '')
            warnings.append(
                f"Best run is {norm:.1f}x worse than optimal at {ep} episodes. "
                "Consider per-episode-count hyperparameters for this count."
            )

    # Check absolute loss spread across episode counts for best run
    losses = [best[col] for col in ep_cols]
    if max(losses) > 100 * min(losses):
        warnings.append(
            f"Extreme loss spread for best run: min={min(losses):.4g}, max={max(losses):.4g}. "
            "This is expected if episode counts span a wide range."
        )

    return warnings


def format_value(key: str, value):
    """Format a hyperparam value for YAML output."""
    if value is None:
        return None
    if key in INT_FIELDS:
        return int(value)
    return value


def compute_config_changes(yaml_path: Path, best_row: dict) -> tuple[dict, dict]:
    """Compute what would change in the YAML without writing.

    Returns (changes_dict, updated_config) where changes_dict maps
    hp_key -> (old_value, new_value).
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    changes = {}

    for hp_key in HYPERPARAM_KEYS:
        value = best_row.get(hp_key)
        if value is None:
            continue

        mapping = YAML_KEY_MAP[hp_key]
        formatted = format_value(hp_key, value)

        if mapping[0] == 'top':
            yaml_key = mapping[1]
            if yaml_key in config:
                old = config[yaml_key]
                if old != formatted:
                    changes[hp_key] = (old, formatted)
                    config[yaml_key] = formatted
        elif mapping[0] == 'nested':
            section = mapping[1]
            yaml_key = mapping[2]
            if section in config and isinstance(config[section], dict):
                if yaml_key in config[section]:
                    old = config[section][yaml_key]
                    if old != formatted:
                        changes[hp_key] = (old, formatted)
                        config[section][yaml_key] = formatted

    return changes, config


def write_method_yaml(yaml_path: Path, updated_config: dict):
    """Write updated config to YAML file."""
    with open(yaml_path, 'w') as f:
        yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)


def select_and_apply(config: ExperimentConfig, method_name: str,
                     config_path: Path, dry_run: bool = False) -> SelectionResult:
    """Full selection pipeline: load CSV, select best, apply to config.

    This is the programmatic API used by run_pipeline.py.

    Args:
        config: Loaded experiment config
        method_name: Method to select hyperparams for
        config_path: Path to the config directory (for finding method YAML)
        dry_run: If True, compute changes but don't write

    Returns:
        SelectionResult with all details
    """
    result = SelectionResult(success=False)

    # Find method config
    method_config = None
    for mc in config.value_estimators.method_configs:
        if mc.name == method_name:
            method_config = mc
            break
    if method_config is None:
        available = [mc.name for mc in config.value_estimators.method_configs]
        result.errors.append(f"Method '{method_name}' not found. Available: {available}")
        return result

    # Find sweep results CSV
    estimator_dir = config.get_estimator_dir(method_config)
    csv_path = estimator_dir / "sweeps" / "sweep_results.csv"
    if not csv_path.exists():
        result.errors.append(f"No sweep results found at {csv_path}")
        return result

    rows = load_sweep_results(csv_path)
    if not rows:
        result.errors.append(f"sweep_results.csv is empty at {csv_path}")
        return result

    ep_cols = get_episode_count_columns(rows)
    result.ep_cols = ep_cols

    if not ep_cols:
        result.errors.append("No episode count columns found in sweep_results.csv")
        return result

    # Filter out invalid rows
    valid_rows, filter_warnings = _filter_valid_rows(rows, ep_cols)
    result.warnings.extend(filter_warnings)
    result.n_runs = len(valid_rows)

    if len(valid_rows) < MIN_RUNS_ERROR:
        result.errors.append(
            f"Only {len(valid_rows)} valid runs (need >= {MIN_RUNS_ERROR}). "
            "Run more sweep agents before selecting."
        )
        return result

    if len(valid_rows) < MIN_RUNS_WARNING:
        result.warnings.append(
            f"Only {len(valid_rows)} valid runs. Consider running more sweep agents "
            f"for more reliable selection (recommended >= {MIN_RUNS_WARNING})."
        )

    # Select best
    best, scored = select_best(valid_rows, ep_cols)
    result.best_row = best
    result.score = best['_score']

    # Diagnostics
    result.warnings.extend(_diagnose_results(scored, ep_cols))

    # Compute config changes
    method_yaml = config_path / f"{method_name}.yaml"
    if not method_yaml.exists():
        result.errors.append(f"Method config file not found at {method_yaml}")
        return result

    changes, updated_config = compute_config_changes(method_yaml, best)
    result.changes = changes

    # Write if not dry run
    if not dry_run:
        write_method_yaml(method_yaml, updated_config)

    result.success = True
    return result


def _print_table(scored: list[dict], ep_cols: list[str], top_n: int):
    """Print ranked table of candidates."""
    n = min(top_n, len(scored))
    ep_labels = [c.replace('best_mc_loss_', '').replace('ep', '') for c in ep_cols]

    hp_present = [k for k in HYPERPARAM_KEYS if any(r.get(k) is not None for r in scored)]
    header = f"{'rank':<5} {'score':<8} {'run_id':<14}"
    for hp in hp_present:
        header += f" {hp:<18}"
    for label in ep_labels:
        header += f" {label+'ep':<14}"
    print(header)
    print("-" * len(header))

    for i, row in enumerate(scored[:n]):
        line = f"{i+1:<5} {row['_score']:<8.4f} {row['run_id']:<14}"
        for hp in hp_present:
            val = row.get(hp)
            if val is None:
                line += f" {'—':<18}"
            elif hp in INT_FIELDS:
                line += f" {int(val):<18}"
            else:
                line += f" {val:<18.6g}"

        for col in ep_cols:
            norm = row['_normalized'][col]
            line += f" {row[col]:<8.4f}({norm:.2f})"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Select best hyperparameters from sweep results")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to experiment config.yaml")
    parser.add_argument("--method", type=str, required=True,
                        help="Method name (e.g., monte_carlo)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Show top N candidates (default: 5)")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    # Run selection (always dry-run first for display)
    result = select_and_apply(config, args.method, args.config.parent, dry_run=True)

    # Print errors and exit if failed
    if result.errors:
        for err in result.errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    # Display results
    print(f"Loaded {result.n_runs} valid sweep runs")

    # Load raw data for table display
    method_config = next(mc for mc in config.value_estimators.method_configs if mc.name == args.method)
    estimator_dir = config.get_estimator_dir(method_config)
    csv_path = estimator_dir / "sweeps" / "sweep_results.csv"
    rows = load_sweep_results(csv_path)
    valid_rows, _ = _filter_valid_rows(rows, result.ep_cols)
    _, scored = select_best(valid_rows, result.ep_cols)

    print(f"\nTop {min(args.top_n, len(scored))} candidates (by mean normalized loss):\n")
    _print_table(scored, result.ep_cols, args.top_n)

    # Print warnings
    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    # Show selection
    print(f"\nSelected: run {result.best_row['run_id']} (score={result.score:.4f})")

    method_yaml = args.config.parent / f"{args.method}.yaml"
    if result.changes:
        print(f"\nChanges to {method_yaml}:")
        for hp, (old, new) in result.changes.items():
            print(f"  {hp}: {old} -> {new}")
    else:
        print("No changes needed — config already matches best hyperparams.")

    # Actually write if not dry run
    if not args.dry_run and result.changes:
        _, updated_config = compute_config_changes(method_yaml, result.best_row)
        write_method_yaml(method_yaml, updated_config)
        print(f"\nWritten to {method_yaml}")
    elif args.dry_run and result.changes:
        print("\n(dry run — no files modified)")

    # Exit with warning code if there were warnings
    if result.warnings and not result.success:
        sys.exit(2)


if __name__ == "__main__":
    main()
