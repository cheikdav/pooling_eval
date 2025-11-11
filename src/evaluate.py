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


def load_estimator_model(estimator_dir: Path, config: ExperimentConfig, device: str = "cpu"):
    """Load a trained estimator model.

    Args:
        estimator_dir: Directory containing trained estimator
        config: Experiment configuration
        device: Device to load model on

    Returns:
        Loaded value network
    """
    model_path = estimator_dir / "estimator_final.pt"
    if not model_path.exists():
        model_path = estimator_dir / "estimator_best.pt"
        if not model_path.exists():
            return None

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create network
    obs_dim = checkpoint['obs_dim']
    hidden_sizes = checkpoint['hidden_sizes']
    activation = checkpoint['activation']

    value_net = ValueNetwork(obs_dim, hidden_sizes, activation)
    value_net.load_state_dict(checkpoint['value_net_state_dict'])
    value_net.to(device)
    value_net.eval()

    return value_net


def generate_predictions(experiment_dir: Path, config: ExperimentConfig,
                        eval_batch: Dict, device: str = "cpu") -> pd.DataFrame:
    """Generate predictions from all trained models on the evaluation batch.

    Args:
        experiment_dir: Experiment directory
        config: Experiment configuration
        eval_batch: Evaluation batch data
        device: Device to use for inference

    Returns:
        DataFrame with columns: state_idx, method, batch_idx, predicted_value
    """
    # Flatten evaluation batch observations
    eval_obs_list = eval_batch['observations']
    eval_obs_flat = np.concatenate(eval_obs_list, axis=0)
    n_states = len(eval_obs_flat)

    print(f"\nGenerating predictions on {n_states} states from evaluation batch...")

    # Convert observations to torch tensor
    eval_obs_tensor = torch.FloatTensor(eval_obs_flat).to(device)

    # Collect all predictions
    predictions = []

    for method in config.value_estimators.methods:
        print(f"\n  Processing method: {method}")

        for batch_idx in range(config.data_generation.n_batches):
            estimator_dir = experiment_dir / "estimators" / method / f"batch_{batch_idx}"

            if not estimator_dir.exists():
                print(f"    Batch {batch_idx}: Directory not found, skipping")
                continue

            # Load model
            value_net = load_estimator_model(estimator_dir, config, device)

            if value_net is None:
                print(f"    Batch {batch_idx}: Model not found, skipping")
                continue

            # Generate predictions
            with torch.no_grad():
                values = value_net(eval_obs_tensor).squeeze(-1).cpu().numpy()

            # Store predictions for each state
            for state_idx in range(n_states):
                predictions.append({
                    'state_idx': state_idx,
                    'method': method,
                    'batch_idx': batch_idx,
                    'predicted_value': values[state_idx]
                })

            print(f"    Batch {batch_idx}: Generated {n_states} predictions")

    # Create DataFrame
    df = pd.DataFrame(predictions)

    print(f"\nTotal predictions: {len(df)}")
    print(f"Shape: {len(df['state_idx'].unique())} states × "
          f"{len(df['method'].unique())} methods × "
          f"{len(df['batch_idx'].unique())} batches")

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare value estimators")
    parser.add_argument("--config", type=Path, required=True,
                       help="Path to config YAML file")
    parser.add_argument("--experiment-dir", type=Path, default=None,
                       help="Experiment directory (default: experiments/<experiment_id>)")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory for results (default: <experiment_dir>/results)")
    args = parser.parse_args()

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)

    # Set default paths
    if args.experiment_dir is None:
        experiment_dir = Path("experiments") / config.experiment_id
    else:
        experiment_dir = args.experiment_dir

    if args.output_dir is None:
        output_dir = experiment_dir / "results"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating experiment: {config.experiment_id}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Methods: {config.value_estimators.methods}")
    print(f"Batches: {config.data_generation.n_batches}\n")

    # Load evaluation batch
    data_dir = experiment_dir / "data"
    print("Loading evaluation batch...")
    try:
        eval_batch = load_evaluation_batch(data_dir)
        print(f"Loaded evaluation batch with {len(eval_batch['observations'])} episodes")

        # Generate predictions
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        predictions_df = generate_predictions(experiment_dir, config, eval_batch, device)

        # Save predictions as CSV
        predictions_csv = output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_csv, index=False)
        print(f"\nPredictions saved to {predictions_csv}")

        # Also save as parquet for more efficient storage (optional)
        predictions_parquet = output_dir / "predictions.parquet"
        predictions_df.to_parquet(predictions_parquet, index=False)
        print(f"Predictions saved to {predictions_parquet}")

    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping prediction generation.")
        predictions_df = None

    # Save results
    results = {}

    if predictions_df is not None:
        results['predictions_summary'] = {
            'n_states': int(predictions_df['state_idx'].nunique()),
            'n_methods': int(predictions_df['method'].nunique()),
            'n_batches': int(predictions_df['batch_idx'].nunique()),
            'total_predictions': len(predictions_df)
        }

        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
