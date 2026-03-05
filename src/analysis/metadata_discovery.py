"""Discover and parse metadata from experiment directories."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def _is_hidden_path(path: Path, base_path: Path) -> bool:
    """Check if path contains any directory component starting with a dot.

    Args:
        path: Path to check
        base_path: Base path to compute relative path from

    Returns:
        True if any directory component (relative to base_path) starts with '.'
    """
    try:
        relative = path.relative_to(base_path)
        return any(part.startswith('.') for part in relative.parts)
    except ValueError:
        return False


def discover_predictions(experiments_dir: Path = Path("experiments")) -> List[Dict[str, Any]]:
    """Discover all predictions with metadata in the experiments/results directories.

    Returns:
        List of dictionaries containing all metadata for each predictions file.
        Each dictionary includes flattened policy/estimator/network metadata,
        plus a generated 'policy_display_name' field.
    """
    predictions_list = []

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        results_dir = exp_dir / "results"
        if not results_dir.exists():
            continue

        for metadata_path in results_dir.rglob("predictions_metadata.json"):
            # Skip hidden directories (those starting with '.')
            if _is_hidden_path(metadata_path, results_dir):
                continue
            try:
                with open(metadata_path, 'r') as f:
                    pred_metadata = json.load(f)

                predictions_file = metadata_path.parent / "predictions.parquet"
                if not predictions_file.exists():
                    print(f"Warning: Metadata found but predictions file missing: {predictions_file}")
                    continue

                prediction_record = {
                    'predictions_path': str(predictions_file),
                    'metadata_path': str(metadata_path),
                }

                # Flatten metadata structure
                for k, v in pred_metadata.items():
                    if k in ['estimator_config', 'network_config']:
                        continue
                    prediction_record[k] = v

                # Flatten estimator_config
                if 'estimator_config' in pred_metadata:
                    for k, v in pred_metadata['estimator_config'].items():
                        prediction_record[f'estimator_{k}'] = v

                # Flatten network_config
                if 'network_config' in pred_metadata:
                    for k, v in pred_metadata['network_config'].items():
                        prediction_record[f'network_{k}'] = v

                predictions_list.append(prediction_record)

            except Exception as e:
                print(f"Warning: Could not load predictions metadata from {metadata_path}: {e}")

    _add_policy_display_names(predictions_list)

    return predictions_list


def discover_estimators(experiments_dir: Path = Path("experiments")) -> List[Dict[str, Any]]:
    """Discover all estimators with metadata in the experiments directory.

    DEPRECATED: Use discover_predictions() instead for analysis workflows.

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
            # Skip hidden directories (those starting with '.')
            if _is_hidden_path(metadata_path, estimators_dir):
                continue
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


class SelectionTreeNode:
    """Node in the selection tree for parameter filtering."""

    def __init__(self, indices: List[int], branch_key: Optional[str] = None):
        """Initialize a tree node.

        Args:
            indices: List of indices into all_predictions for this node
            branch_key: The parameter key to branch on at this level (None for leaf)
        """
        self.indices = indices
        self.branch_key = branch_key
        self.children = {}  # Maps value -> child node

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (single element or no branching)."""
        return len(self.indices) <= 1 or not self.children

    def has_single_branch(self) -> bool:
        """Check if this node has exactly one child."""
        return len(self.children) == 1

    def get_single_child(self) -> tuple:
        """Get the single child (value, node) if has_single_branch is True."""
        if not self.has_single_branch():
            raise ValueError("Node does not have exactly one child")
        return next(iter(self.children.items()))


def build_selection_tree(
    all_predictions: List[Dict[str, Any]],
    primary_keys: List[str]
) -> SelectionTreeNode:
    """Build a tree structure for parameter selection.

    The tree branches on parameters in the following order:
    1. First, branch on all primary_keys in order
    2. Then, recursively branch on any remaining keys where values differ

    Args:
        all_predictions: List of all prediction metadata dicts
        primary_keys: List of keys to branch on first (in order)

    Returns:
        Root node of the selection tree
    """
    root_indices = list(range(len(all_predictions)))
    root = SelectionTreeNode(root_indices)

    # Build tree recursively
    _build_tree_recursive(root, all_predictions, primary_keys.copy())

    return root


def _build_tree_recursive(
    node: SelectionTreeNode,
    all_predictions: List[Dict[str, Any]],
    remaining_primary_keys: List[str]
):
    """Recursively build the selection tree."""

    # Base case: single element
    if len(node.indices) <= 1:
        return

    # Get predictions for this node
    node_predictions = [all_predictions[i] for i in node.indices]

    # Determine which key to branch on
    branch_key = None

    # First try remaining primary keys
    if remaining_primary_keys:
        branch_key = remaining_primary_keys[0]
        remaining_primary_keys = remaining_primary_keys[1:]
    else:
        # Find any key where values differ, excluding special keys
        exclude_keys = {'predictions_path', 'metadata_path', 'created_at'}

        for key in node_predictions[0].keys():
            if key in exclude_keys or key.endswith('_path'):
                continue

            # Check if this key has different values
            values = set()
            for pred in node_predictions:
                val = pred.get(key)
                # Convert unhashable types to string
                if isinstance(val, (dict, list)):
                    val = str(val)
                values.add(val)

            if len(values) > 1:
                branch_key = key
                break

    # If no key to branch on, this is a leaf
    if branch_key is None:
        return

    node.branch_key = branch_key

    # Group indices by value for this key
    value_to_indices = {}
    for idx in node.indices:
        val = all_predictions[idx].get(branch_key)
        # Convert unhashable types to string for dict key
        if isinstance(val, (dict, list)):
            val = str(val)

        if val not in value_to_indices:
            value_to_indices[val] = []
        value_to_indices[val].append(idx)

    # Create child nodes and recurse
    for value, child_indices in value_to_indices.items():
        child_node = SelectionTreeNode(child_indices)
        node.children[value] = child_node
        _build_tree_recursive(child_node, all_predictions, remaining_primary_keys.copy())


