"""Select best hyperparameters from sweep results and write them to the results tree.

Reads sweep_results.csv, picks the best hyperparameter set using
mean-normalized-loss ranking, and writes the selection as JSON into the
estimator's `sweeps/tuned_hyperparams.json`. The downstream training stage
reads this file at load time and merges the values into the method config;
the source config YAML is never mutated.

Usage:
    uv run -m src.select_hyperparameters \
        --config configs/humanoid/config.yaml \
        --method monte_carlo \
        --output experiments/Humanoid-v5/.../monte_carlo_estimator_001/sweeps/tuned_hyperparams.json
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import ExperimentConfig
from src.tune_hyperparameters import HYPERPARAM_KEYS


INT_FIELDS = {'batch_size', 'n_components', 'rbf_n_components'}

# Minimum number of sweep runs for reliable selection
MIN_RUNS_WARNING = 5
MIN_RUNS_ERROR = 2


@dataclass
class SelectionResult:
    """Result of hyperparameter selection."""
    success: bool
    best_row: Optional[dict] = None
    score: float = float('inf')
    n_runs: int = 0
    selected_hyperparams: dict = field(default_factory=dict)  # key -> value (flat dict)
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

    # Deduplicate by run_id (keep last occurrence, most recent)
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
    for that count. Picks the run with the lowest mean normalized score.
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

    if best['_score'] > 1.5:
        warnings.append(
            f"Best score is {best['_score']:.2f} (ideal=1.0). "
            "No single hyperparameter set is good across all episode counts."
        )

    if len(scored_rows) >= 2:
        second = scored_rows[1]
        gap = second['_score'] - best['_score']
        if gap > 0.5:
            warnings.append(
                f"Large gap between #1 (score={best['_score']:.3f}) and "
                f"#2 (score={second['_score']:.3f}). Best run may be an outlier."
            )

    for col in ep_cols:
        norm = best['_normalized'][col]
        if norm > 2.0:
            ep = col.replace('best_mc_loss_', '').replace('ep', '')
            warnings.append(
                f"Best run is {norm:.1f}x worse than optimal at {ep} episodes. "
                "Consider per-episode-count hyperparameters for this count."
            )

    losses = [best[col] for col in ep_cols]
    if max(losses) > 100 * min(losses):
        warnings.append(
            f"Extreme loss spread for best run: min={min(losses):.4g}, max={max(losses):.4g}. "
            "This is expected if episode counts span a wide range."
        )

    return warnings


def format_value(key: str, value):
    """Format a hyperparam value for JSON output (preserves int typing)."""
    if value is None:
        return None
    if key in INT_FIELDS:
        return int(value)
    return float(value)


def build_tuned_hyperparams(best_row: dict) -> dict:
    """Extract a flat {key: value} dict of tuned hyperparams from a selected row.

    Only includes keys in HYPERPARAM_KEYS that are present and non-None in
    best_row. The resulting dict is what gets serialized to tuned_hyperparams.json
    and applied by `src.train_estimator.apply_tuned_hyperparams`.
    """
    out = {}
    for key in HYPERPARAM_KEYS:
        value = best_row.get(key)
        if value is None:
            continue
        out[key] = format_value(key, value)
    return out


def write_tuned_hyperparams_json(path: Path, hyperparams: dict):
    """Write the flat hyperparam dict as JSON to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(hyperparams, f, indent=2, sort_keys=True)


def select_hyperparams_for_method(
    config: ExperimentConfig,
    method_name: str,
) -> SelectionResult:
    """Load sweep_results.csv for a method and return the best hyperparameter set.

    Does not write anything. Caller is responsible for persisting the result
    via `write_tuned_hyperparams_json`.
    """
    result = SelectionResult(success=False)

    method_config = None
    for mc in config.value_estimators.method_configs:
        if mc.name == method_name:
            method_config = mc
            break
    if method_config is None:
        available = [mc.name for mc in config.value_estimators.method_configs]
        result.errors.append(f"Method '{method_name}' not found. Available: {available}")
        return result

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

    best, scored = select_best(valid_rows, ep_cols)
    result.best_row = best
    result.score = best['_score']
    result.selected_hyperparams = build_tuned_hyperparams(best)
    result.warnings.extend(_diagnose_results(scored, ep_cols))

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
    parser = argparse.ArgumentParser(
        description="Select best hyperparameters from sweep results and write to tuned_hyperparams.json"
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to experiment config.yaml")
    parser.add_argument("--method", type=str, required=True,
                        help="Method name (e.g., monte_carlo)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Destination for tuned_hyperparams.json "
                             "(defaults to {estimator_dir}/sweeps/tuned_hyperparams.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show selection without writing the file")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Show top N candidates (default: 5)")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    result = select_hyperparams_for_method(config, args.method)

    if result.errors:
        for err in result.errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    # Print ranked table for human inspection
    method_config = next(mc for mc in config.value_estimators.method_configs if mc.name == args.method)
    estimator_dir = config.get_estimator_dir(method_config)
    csv_path = estimator_dir / "sweeps" / "sweep_results.csv"
    rows = load_sweep_results(csv_path)
    valid_rows, _ = _filter_valid_rows(rows, result.ep_cols)
    _, scored = select_best(valid_rows, result.ep_cols)

    print(f"Loaded {result.n_runs} valid sweep runs")
    print(f"\nTop {min(args.top_n, len(scored))} candidates (by mean normalized loss):\n")
    _print_table(scored, result.ep_cols, args.top_n)

    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    print(f"\nSelected: run {result.best_row['run_id']} (score={result.score:.4f})")
    print(f"Tuned hyperparams:")
    for k, v in sorted(result.selected_hyperparams.items()):
        print(f"  {k}: {v}")

    output_path = args.output or (estimator_dir / "sweeps" / "tuned_hyperparams.json")
    if args.dry_run:
        print(f"\n(dry run — would write to {output_path})")
    else:
        write_tuned_hyperparams_json(output_path, result.selected_hyperparams)
        print(f"\nWrote tuned hyperparams to {output_path}")


if __name__ == "__main__":
    main()
