"""Discover and parse metadata from experiment directories."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def discover_estimators(experiments_dir: Path = Path("experiments")) -> List[Dict[str, Any]]:
    """Discover all estimators with metadata in the experiments directory.

    Returns:
        List of dictionaries containing all metadata for each estimator.
        Each dictionary includes nested policy and data metadata (already embedded),
        plus a generated 'policy_display_name' field.
    """
    estimators = []

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        estimators_dir = exp_dir / "estimators"
        if not estimators_dir.exists():
            continue

        for metadata_path in estimators_dir.rglob("estimator_metadata.json"):
            try:
                with open(metadata_path, 'r') as f:
                    estimator_metadata = json.load(f)

                data_metadata = estimator_metadata.get('data_metadata', {})
                policy_metadata = data_metadata.get('policy_metadata', {})

                estimator = {
                    'experiment_id': exp_dir.name,
                    'estimator_path': str(metadata_path.parent / "estimator.pt"),
                    'metadata_path': str(metadata_path),
                }

                for k, v in estimator_metadata.items():
                    if k not in ['data_metadata', 'estimator_config', 'network_config']:
                        estimator[k] = v

                if 'estimator_config' in estimator_metadata:
                    for k, v in estimator_metadata['estimator_config'].items():
                        estimator[f'estimator_{k}'] = v

                if 'network_config' in estimator_metadata:
                    for k, v in estimator_metadata['network_config'].items():
                        estimator[f'network_{k}'] = v

                for k, v in data_metadata.items():
                    if k != 'policy_metadata':
                        estimator[f'data_{k}'] = v

                for k, v in policy_metadata.items():
                    estimator[f'policy_{k}'] = v

                estimators.append(estimator)

            except Exception as e:
                print(f"Warning: Could not load estimator metadata from {metadata_path}: {e}")

    _add_policy_display_names(estimators)

    return estimators


def _add_policy_display_names(estimators: List[Dict[str, Any]]) -> None:
    """Add policy_display_name field to all estimators based on unique policies.

    Modifies estimators in place, adding a 'policy_display_name' field.
    Display names are like 'PPO 1', 'PPO 2', 'DQN 1', etc.
    """
    policy_groups = {}
    for est in estimators:
        algo = est.get('policy_algorithm', 'unknown')
        seed = est.get('policy_seed', 0)
        exp_id = est.get('experiment_id', '')

        policy_key = (algo, seed, exp_id)

        if algo not in policy_groups:
            policy_groups[algo] = []

        if policy_key not in [pk for _, _, pk_list in policy_groups[algo] for pk in pk_list]:
            policy_groups[algo].append(policy_key)

    for algo in policy_groups:
        policy_groups[algo] = sorted(set(policy_groups[algo]))

    policy_display_map = {}
    for algo, policies in policy_groups.items():
        if len(policies) == 1:
            policy_display_map[policies[0]] = algo
        else:
            for idx, policy_key in enumerate(policies):
                policy_display_map[policy_key] = f"{algo} {idx + 1}"

    for est in estimators:
        algo = est.get('policy_algorithm', 'unknown')
        seed = est.get('policy_seed', 0)
        exp_id = est.get('experiment_id', '')
        policy_key = (algo, seed, exp_id)

        est['policy_display_name'] = policy_display_map.get(policy_key, algo)


def get_unique_values(estimators: List[Dict[str, Any]], key: str) -> List[Any]:
    """Get sorted list of unique values for a given key across all estimators."""
    values = set()
    for est in estimators:
        if key in est:
            val = est[key]
            if isinstance(val, (dict, list)):
                val = str(val)
            values.add(val)
    return sorted(values)


def filter_estimators(estimators: List[Dict[str, Any]], key: str, value: Any) -> List[Dict[str, Any]]:
    """Filter estimators by a specific key-value pair."""
    return [est for est in estimators if est.get(key) == value]


def get_selection_hierarchy(
    estimators: List[Dict[str, Any]],
    primary_keys: List[str],
    current_filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build a hierarchical selection structure based on primary keys.

    Args:
        estimators: All estimators
        primary_keys: List of keys to use for hierarchical filtering (in order)
        current_filters: Already applied filters {key: value}

    Returns:
        Dictionary with structure:
        {
            'current_estimators': [...],  # Estimators after current filters
            'next_key': str or None,      # Next key to filter by
            'available_values': [...],    # Available values for next key
            'additional_params': [...],   # Other varying parameters after all primary keys
        }
    """
    if current_filters is None:
        current_filters = {}

    filtered = estimators
    for key, value in current_filters.items():
        filtered = filter_estimators(filtered, key, value)

    next_key = None
    for key in primary_keys:
        if key not in current_filters:
            next_key = key
            break

    result = {
        'current_estimators': filtered,
        'next_key': next_key,
        'available_values': [],
        'additional_params': []
    }

    if next_key:
        result['available_values'] = get_unique_values(filtered, next_key)
    else:
        if len(filtered) > 1:
            all_keys = set(filtered[0].keys())

            varying_keys = []
            for key in all_keys:
                if key in primary_keys or key.endswith('_path'):
                    continue

                values = get_unique_values(filtered, key)
                if len(values) > 1:
                    varying_keys.append({
                        'key': key,
                        'values': values
                    })

            result['additional_params'] = varying_keys

    return result


