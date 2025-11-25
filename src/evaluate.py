"""Evaluate and compare value estimators."""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict

from src.config import ExperimentConfig
from src.estimators.base import ValueNetwork


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


def load_estimator_model(model_path: Path, device: str = "cpu"):
    """Load a trained estimator model.

    Args:
        model_path: Path to model file
        device: Device to load model on

    Returns:
        Loaded value network
    """
    checkpoint = torch.load(model_path, map_location=device)

    obs_dim = checkpoint['obs_dim']
    hidden_sizes = checkpoint['hidden_sizes']
    activation = checkpoint['activation']

    value_net = ValueNetwork(obs_dim, hidden_sizes, activation)
    value_net.load_state_dict(checkpoint['value_net_state_dict'])
    value_net.to(device)
    value_net.eval()

    return value_net


def generate_predictions(experiment_dir: Path, config: ExperimentConfig,
                        eval_batch: Dict, output_dir: Path, device: str = "cpu"):
    """Generate predictions from all trained models and save per n_episodes.

    Args:
        experiment_dir: Experiment directory
        config: Experiment configuration
        eval_batch: Evaluation batch data
        output_dir: Directory to save prediction files
        device: Device to use for inference

    Saves:
        One parquet file per n_episodes value: predictions_<n_episodes>.parquet
    """
    eval_obs_list = eval_batch['observations']
    eval_obs_flat = np.concatenate(eval_obs_list, axis=0)
    n_states = len(eval_obs_flat)

    # Create episode index mapping for each state
    episode_indices = []
    for ep_idx, obs_array in enumerate(eval_obs_list):
        episode_indices.extend([ep_idx] * len(obs_array))

    print(f"\nGenerating predictions on {n_states} states from {len(eval_obs_list)} episodes...")

    eval_obs_tensor = torch.FloatTensor(eval_obs_flat).to(device)

    # Organize predictions by n_episodes
    predictions_by_n_episodes = {}

    for method_config in config.value_estimators.method_configs:
        method_name = method_config.name
        print(f"\n  Processing method: {method_name}")

        # Get available n_episodes directories
        method_dir = experiment_dir / "estimators" / method_name
        if not method_dir.exists():
            print(f"    Method directory not found: {method_dir}, skipping")
            continue

        n_episodes_dirs = [d for d in method_dir.iterdir() if d.is_dir() and d.name.isdigit()]

        for n_ep_dir in n_episodes_dirs:
            n_episodes = int(n_ep_dir.name)

            if n_episodes not in predictions_by_n_episodes:
                predictions_by_n_episodes[n_episodes] = []

            print(f"    Processing n_episodes={n_episodes}")

            # Iterate through batch directories
            for batch_idx in range(config.data_generation.n_batches):
                batch_dir = n_ep_dir / f"batch_{batch_idx}"

                if not batch_dir.exists():
                    continue

                model_path = batch_dir / "estimator.pt"

                if not model_path.exists():
                    continue

                value_net = load_estimator_model(model_path, device)

                with torch.no_grad():
                    values = value_net(eval_obs_tensor).squeeze(-1).cpu().numpy()

                for state_idx in range(n_states):
                    predictions_by_n_episodes[n_episodes].append({
                        'state_idx': state_idx,
                        'episode_idx': episode_indices[state_idx],
                        'method': method_name,
                        'batch_idx': batch_idx,
                        'predicted_value': values[state_idx]
                    })

            print(f"      Batch {batch_idx}: Generated {n_states} predictions")

    # Save one file per n_episodes
    print("\nSaving prediction files:")
    for n_episodes, predictions in predictions_by_n_episodes.items():
        df = pd.DataFrame(predictions)
        output_file = output_dir / f"predictions_{n_episodes}.parquet"
        df.to_parquet(output_file, index=False)
        print(f"  {output_file.name}: {len(df)} predictions")
        print(f"    ({len(df['state_idx'].unique())} states × "
              f"{len(df['method'].unique())} methods × "
              f"{len(df['batch_idx'].unique())} batches)")

    return predictions_by_n_episodes.keys()


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare value estimators")
    parser.add_argument("--config", type=Path, required=True,
                       help="Path to config YAML file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    experiment_dir = Path("experiments") / config.experiment_id
    data_dir = experiment_dir / "data"
    output_dir = experiment_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating experiment: {config.experiment_id}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Methods: {[mc.name for mc in config.value_estimators.method_configs]}")
    print(f"Batches: {config.data_generation.n_batches}\n")

    print("Loading evaluation batch...")
    eval_batch = load_evaluation_batch(data_dir)
    print(f"Loaded evaluation batch with {len(eval_batch['observations'])} episodes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    n_episodes_list = generate_predictions(experiment_dir, config, eval_batch, output_dir, device)

    results = {
        'n_episodes_files': sorted(list(n_episodes_list)),
        'n_files': len(n_episodes_list)
    }

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
