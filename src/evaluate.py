"""Evaluate and compare value estimators."""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict
from datetime import datetime

from src.config import ExperimentConfig
from src.estimators.monte_carlo import MonteCarloEstimator
from src.estimators.dqn import DQNEstimator
from src.estimators.least_squares_mc import LeastSquaresMCEstimator
from src.estimators.least_squares_td import LeastSquaresTDEstimator


# Mapping from method names to estimator classes
ESTIMATOR_CLASSES = {
    'monte_carlo': MonteCarloEstimator,
    'dqn': DQNEstimator,
    'least_squares_mc': LeastSquaresMCEstimator,
    'least_squares_td': LeastSquaresTDEstimator,
}


def load_evaluation_batch(data_dir: Path) -> Dict:
    """Load the evaluation batch of data.

    Args:
        data_dir: Directory containing data files

    Returns:
        Dictionary containing evaluation batch data
    """
    eval_batch_path = data_dir / "batch_eval.npz"
    if not eval_batch_path.exists():
        raise FileNotFoundError(f"Evaluation batch not found at {eval_batch_path}")

    batch = np.load(eval_batch_path, allow_pickle=True)
    return dict(batch)


def load_estimator_model(model_path: Path, method_name: str, device: str = "cpu"):
    """Load a trained estimator model using the class's load_from_checkpoint method.

    Args:
        model_path: Path to model file
        method_name: Name of the estimation method (e.g., 'monte_carlo', 'dqn')
        device: Device to load model on

    Returns:
        Loaded estimator with predict() method
    """
    # Get the appropriate estimator class
    estimator_class = ESTIMATOR_CLASSES.get(method_name)
    if estimator_class is None:
        raise ValueError(f"Unknown method: {method_name}. Available methods: {list(ESTIMATOR_CLASSES.keys())}")

    # Load using the class method
    estimator = estimator_class.load_from_checkpoint(model_path, device=device)
    return estimator


def generate_predictions_for_n_episodes(config: ExperimentConfig,
                                        eval_batch: Dict, results_dir: Path,
                                        method_name: str, n_episodes: int,
                                        n_ep_dir: Path, device: str = "cpu"):
    """Generate predictions for a single method/n_episodes combination.

    Args:
        config: Experiment configuration
        eval_batch: Evaluation batch data
        results_dir: Root results directory
        method_name: Name of the method
        n_episodes: Number of episodes used for training
        n_ep_dir: Directory containing estimator models
        device: Device to use for inference

    Returns:
        Path to saved predictions file (relative to results_dir), or None if no predictions
    """
    eval_obs_list = eval_batch['observations']
    eval_obs_flat = np.concatenate(eval_obs_list, axis=0)
    n_states = len(eval_obs_flat)

    # Create episode index mapping for each state
    episode_indices = []
    for ep_idx, obs_array in enumerate(eval_obs_list):
        episode_indices.extend([ep_idx] * len(obs_array))

    # Load metadata from first batch
    first_batch_metadata_path = n_ep_dir / "batch_0" / "estimator_metadata.json"
    if not first_batch_metadata_path.exists():
        print(f"      No metadata found at {first_batch_metadata_path}, skipping")
        return None

    with open(first_batch_metadata_path, 'r') as f:
        sample_metadata = json.load(f)

    # Collect predictions for this n_episodes (process batches one at a time)
    predictions = []

    for batch_idx in range(config.data_generation.n_batches):
        batch_name = str(batch_idx)
        batch_dir = n_ep_dir / f"batch_{batch_name}"

        if not batch_dir.exists():
            continue

        model_path = batch_dir / "estimator.pt"
        if not model_path.exists():
            continue

        # Load model, generate predictions, then free memory
        estimator = load_estimator_model(model_path, method_name, device)

        # Use predict() method instead of direct value_net call
        values = estimator.predict(eval_obs_flat)

        for state_idx in range(n_states):
            predictions.append({
                'state_idx': state_idx,
                'episode_idx': episode_indices[state_idx],
                'batch_name': batch_name,
                'predicted_value': values[state_idx]
            })

        # Free memory
        del estimator
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not predictions:
        print(f"      No predictions generated, skipping")
        return None

    # Create output directory
    output_method_dir = results_dir / method_name / str(n_episodes)
    output_method_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    df = pd.DataFrame(predictions)
    predictions_file = output_method_dir / "predictions.parquet"
    df.to_parquet(predictions_file, index=False)

    # Create metadata
    data_metadata = sample_metadata.get('data_metadata', {})
    policy_metadata = data_metadata.get('policy_metadata', {})

    predictions_metadata = {
        'experiment_id': config.experiment_id,
        'method': method_name,
        'n_episodes': n_episodes,
        'n_batches': len(df['batch_name'].unique()),
        'n_states': len(df['state_idx'].unique()),
        'n_eval_episodes': len(eval_obs_list),
        'created_at': datetime.now().isoformat(),

        # Policy metadata
        'policy_environment': policy_metadata.get('environment'),
        'policy_algorithm': policy_metadata.get('algorithm'),
        'policy_seed': policy_metadata.get('seed'),
        'policy_learning_rate': policy_metadata.get('learning_rate'),
        'policy_gamma': policy_metadata.get('gamma'),
        'policy_total_timesteps': policy_metadata.get('total_timesteps'),
        'policy_average_reward': policy_metadata.get('average_reward'),

        # Estimator config
        'estimator_config': sample_metadata.get('estimator_config', {}),
        'network_config': sample_metadata.get('network_config', {}),

        # Data metadata
        'data_seed': data_metadata.get('seed'),
    }

    predictions_metadata['policy_display_name'] = None

    metadata_file = output_method_dir / "predictions_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(predictions_metadata, f, indent=2)

    print(f"      Saved: {predictions_file.relative_to(results_dir)}")
    print(f"        ({len(df)} predictions = {len(df['state_idx'].unique())} states × "
          f"{len(df['batch_name'].unique())} batches)")

    # Free memory
    del df
    del predictions

    return str(predictions_file.relative_to(results_dir))


def generate_predictions(experiment_dir: Path, config: ExperimentConfig,
                        eval_batch: Dict, results_dir: Path, device: str = "cpu"):
    """Generate predictions from all trained models with metadata.

    Processes one n_episodes at a time to avoid memory issues.

    Args:
        experiment_dir: Experiment directory
        config: Experiment configuration
        eval_batch: Evaluation batch data
        results_dir: Root results directory
        device: Device to use for inference

    Saves:
        For each method/n_episodes combination:
        - results/<method>/<n_episodes>/predictions.parquet
        - results/<method>/<n_episodes>/predictions_metadata.json
    """
    eval_obs_list = eval_batch['observations']
    eval_obs_flat = np.concatenate(eval_obs_list, axis=0)
    n_states = len(eval_obs_flat)

    print(f"\nGenerating predictions on {n_states} states from {len(eval_obs_list)} episodes...")

    prediction_files_created = []

    for method_config in config.value_estimators.method_configs:
        method_name = method_config.name
        print(f"\n  Processing method: {method_name}")

        # Get available n_episodes directories from estimators
        estimators_method_dir = experiment_dir / "estimators" / method_name
        if not estimators_method_dir.exists():
            print(f"    Method directory not found: {estimators_method_dir}, skipping")
            continue

        n_episodes_dirs = sorted([d for d in estimators_method_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                                key=lambda d: int(d.name))

        for n_ep_dir in n_episodes_dirs:
            n_episodes = int(n_ep_dir.name)
            print(f"    Processing n_episodes={n_episodes}")

            # Process this n_episodes and save immediately
            predictions_file = generate_predictions_for_n_episodes(
                config, eval_batch, results_dir,
                method_name, n_episodes, n_ep_dir, device
            )

            if predictions_file:
                prediction_files_created.append(predictions_file)

    return prediction_files_created


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare value estimators")
    parser.add_argument("--config", type=Path, required=True,
                       help="Path to config YAML file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    experiment_dir = Path("experiments") / config.experiment_id
    data_dir = experiment_dir / "data"
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating experiment: {config.experiment_id}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Methods: {[mc.name for mc in config.value_estimators.method_configs]}")
    print(f"Batches: {config.data_generation.n_batches}\n")

    print("Loading evaluation batch...")
    eval_batch = load_evaluation_batch(data_dir)
    print(f"Loaded evaluation batch with {len(eval_batch['observations'])} episodes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    prediction_files = generate_predictions(experiment_dir, config, eval_batch, results_dir, device)

    summary = {
        'experiment_id': config.experiment_id,
        'prediction_files': prediction_files,
        'n_files': len(prediction_files),
        'created_at': datetime.now().isoformat()
    }

    summary_file = results_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
