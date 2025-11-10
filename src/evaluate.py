"""Evaluate and compare value estimators."""

import argparse
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict

from src.config import ExperimentConfig


def load_training_stats(estimator_dir: Path) -> Dict:
    """Load training statistics from a trained estimator.

    Args:
        estimator_dir: Directory containing trained estimator

    Returns:
        Dictionary of training statistics
    """
    stats_file = estimator_dir / "training_stats.json"
    if not stats_file.exists():
        return None

    with open(stats_file, 'r') as f:
        return json.load(f)


def collect_all_stats(experiment_dir: Path, methods: List[str], n_batches: int) -> Dict:
    """Collect statistics from all trained estimators.

    Args:
        experiment_dir: Experiment directory
        methods: List of methods
        n_batches: Number of batches

    Returns:
        Nested dictionary: {method: {batch_idx: stats}}
    """
    all_stats = defaultdict(dict)

    for method in methods:
        for batch_idx in range(n_batches):
            estimator_dir = (experiment_dir / "estimators" / method /
                           f"batch_{batch_idx}")

            if estimator_dir.exists():
                stats = load_training_stats(estimator_dir)
                if stats:
                    all_stats[method][batch_idx] = stats

    return all_stats


def compute_aggregate_metrics(all_stats: Dict) -> Dict:
    """Compute aggregate metrics across batches for each method.

    Args:
        all_stats: Nested dictionary of statistics

    Returns:
        Dictionary of aggregate metrics per method
    """
    aggregate = {}

    for method, batch_stats in all_stats.items():
        if not batch_stats:
            continue

        final_losses = [stats['final_loss'] for stats in batch_stats.values()
                       if stats['final_loss'] is not None]
        best_losses = [stats['best_loss'] for stats in batch_stats.values()]
        final_epochs = [stats['final_epoch'] for stats in batch_stats.values()]
        converged = [stats['converged'] for stats in batch_stats.values()]

        aggregate[method] = {
            'mean_final_loss': np.mean(final_losses) if final_losses else None,
            'std_final_loss': np.std(final_losses) if final_losses else None,
            'mean_best_loss': np.mean(best_losses),
            'std_best_loss': np.std(best_losses),
            'mean_epochs': np.mean(final_epochs),
            'std_epochs': np.std(final_epochs),
            'convergence_rate': np.mean(converged),
            'n_batches': len(batch_stats),
        }

    return aggregate


def print_summary(aggregate: Dict):
    """Print summary of results.

    Args:
        aggregate: Aggregate metrics dictionary
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80 + "\n")

    for method, metrics in aggregate.items():
        print(f"\n{method.upper()}")
        print("-" * 40)
        print(f"  Batches completed: {metrics['n_batches']}")
        print(f"  Mean final loss: {metrics['mean_final_loss']:.6f} ± {metrics['std_final_loss']:.6f}")
        print(f"  Mean best loss: {metrics['mean_best_loss']:.6f} ± {metrics['std_best_loss']:.6f}")
        print(f"  Mean epochs: {metrics['mean_epochs']:.1f} ± {metrics['std_epochs']:.1f}")
        print(f"  Convergence rate: {metrics['convergence_rate']*100:.1f}%")

    print("\n" + "="*80 + "\n")


def plot_results(all_stats: Dict, aggregate: Dict, output_dir: Path):
    """Create visualization plots.

    Args:
        all_stats: All statistics
        aggregate: Aggregate metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(all_stats.keys())
    n_methods = len(methods)

    # Plot 1: Final loss comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(n_methods)
    means = [aggregate[m]['mean_final_loss'] for m in methods]
    stds = [aggregate[m]['std_final_loss'] for m in methods]

    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Method')
    ax.set_ylabel('Final Loss (MSE)')
    ax.set_title('Final Loss Comparison Across Methods')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'final_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Loss distribution per method (box plot)
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = []
    labels = []
    for method in methods:
        losses = [stats['final_loss'] for stats in all_stats[method].values()
                 if stats['final_loss'] is not None]
        if losses:
            data_to_plot.append(losses)
            labels.append(method)

    if data_to_plot:
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_xlabel('Method')
        ax.set_ylabel('Final Loss (MSE)')
        ax.set_title('Loss Distribution Across Batches')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'loss_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 3: Training epochs comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [aggregate[m]['mean_epochs'] for m in methods]
    stds = [aggregate[m]['std_epochs'] for m in methods]

    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='orange')
    ax.set_xlabel('Method')
    ax.set_ylabel('Number of Epochs')
    ax.set_title('Training Duration Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'epochs_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Per-batch performance
    fig, ax = plt.subplots(figsize=(12, 6))

    for method in methods:
        batch_indices = sorted(all_stats[method].keys())
        losses = [all_stats[method][idx]['final_loss'] for idx in batch_indices
                 if all_stats[method][idx]['final_loss'] is not None]
        ax.plot(batch_indices[:len(losses)], losses, marker='o', label=method, alpha=0.7)

    ax.set_xlabel('Batch Index')
    ax.set_ylabel('Final Loss (MSE)')
    ax.set_title('Performance Across Batches')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_batch_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}")


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

    # Collect all statistics
    print("Collecting training statistics...")
    all_stats = collect_all_stats(
        experiment_dir,
        config.value_estimators.methods,
        config.data_generation.n_batches
    )

    # Compute aggregate metrics
    aggregate = compute_aggregate_metrics(all_stats)

    # Print summary
    print_summary(aggregate)

    # Save results
    results = {
        'aggregate': aggregate,
        'all_stats': {method: dict(batch_stats)
                     for method, batch_stats in all_stats.items()},
    }

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    # Create plots
    print("\nGenerating plots...")
    plot_results(all_stats, aggregate, output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
