"""Walk the experiment directory hierarchy and discover available results."""

import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional


NUMBERED_DIR = re.compile(r"^(.+?)_(\d+)$")
ESTIMATOR_DIR = re.compile(r"^(.+)_estimator_(\d+)$")


def _load_params(directory: Path) -> dict:
    params_file = directory / "params.json"
    if params_file.exists():
        with open(params_file) as f:
            return json.load(f)
    return {}


def _find_numbered_dirs(parent: Path, prefix: str) -> List[Path]:
    """Find dirs matching {prefix}_NNN, sorted by number."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    dirs = []
    for d in parent.iterdir():
        if d.is_dir() and pattern.match(d.name):
            dirs.append(d)
    return sorted(dirs, key=lambda d: d.name)


def _find_estimator_dirs(parent: Path) -> List[Path]:
    """Find dirs matching *_estimator_NNN."""
    return sorted(
        (d for d in parent.iterdir() if d.is_dir() and ESTIMATOR_DIR.match(d.name)),
        key=lambda d: d.name,
    )


def _extract_method_name(dirname: str) -> str:
    """'monte_carlo_estimator_001' -> 'monte_carlo'"""
    m = ESTIMATOR_DIR.match(dirname)
    return m.group(1) if m else dirname


def _add_policy_display_names(entries: List[dict]):
    """Add human-readable policy names like 'SAC #1', 'PPO #2'."""
    groups = {}
    for e in entries:
        key = (e["env_name"], e.get("policy_algorithm", "?"), e.get("policy_seed", 0))
        groups.setdefault(key, []).append(e)

    # Count distinct policies per (env, algorithm)
    algo_counts = {}
    for (env, algo, seed), group in groups.items():
        algo_counts.setdefault((env, algo), set()).add(seed)

    for (env, algo, seed), group in groups.items():
        n_policies = len(algo_counts[(env, algo)])
        if n_policies == 1:
            display = algo
        else:
            seeds = sorted(algo_counts[(env, algo)])
            idx = seeds.index(seed) + 1
            display = f"{algo} #{idx}"
        for e in group:
            e["policy_display_name"] = display


def discover_experiments(search_paths: List[Path]) -> pd.DataFrame:
    """Walk experiment hierarchy and return DataFrame of all discovered results.

    Each row represents one (method, n_episodes) combination with all paths resolved.
    """
    entries = []

    for base in search_paths:
        if not base.exists():
            continue
        for env_dir in sorted(base.iterdir()):
            if not env_dir.is_dir():
                continue

            for policy_dir in _find_numbered_dirs(env_dir, "policy"):
                policy_params = _load_params(policy_dir)

                for data_dir in _find_numbered_dirs(policy_dir, "data"):
                    data_params = _load_params(data_dir)
                    data_files_dir = data_dir / "data"

                    for est_dir in _find_estimator_dirs(data_dir):
                        method = _extract_method_name(est_dir.name)
                        est_params = _load_params(est_dir)

                        for eval_dir in _find_numbered_dirs(est_dir, "eval"):
                            eval_params = _load_params(eval_dir)
                            results_dir = eval_dir / "results"
                            method_results = results_dir / method

                            if not method_results.exists():
                                continue

                            gt_path = results_dir / "ground_truth" / "ground_truth_returns.parquet"

                            for n_ep_dir in sorted(method_results.iterdir()):
                                if not n_ep_dir.is_dir():
                                    continue
                                try:
                                    n_episodes = int(n_ep_dir.name)
                                except ValueError:
                                    continue

                                pred_file = n_ep_dir / "predictions.parquet"
                                if not pred_file.exists():
                                    continue

                                paired_file = n_ep_dir / "paired_predictions.parquet"

                                entries.append({
                                    "env_name": env_dir.name,
                                    "method": method,
                                    "n_episodes": n_episodes,
                                    "predictions_path": str(pred_file),
                                    "paired_predictions_path": str(paired_file) if paired_file.exists() else None,
                                    "ground_truth_path": str(gt_path) if gt_path.exists() else None,
                                    "data_dir": str(data_files_dir),
                                    "results_dir": str(results_dir),
                                    "policy_dir": str(policy_dir),
                                    "policy_algorithm": policy_params.get("algorithm", "?"),
                                    "policy_gamma": policy_params.get("gamma", 0.99),
                                    "policy_seed": policy_params.get("seed", 0),
                                    "eval_gamma": eval_params.get("gamma", policy_params.get("gamma", 0.99)),
                                })

    _add_policy_display_names(entries)

    if not entries:
        return pd.DataFrame()
    return pd.DataFrame(entries)
