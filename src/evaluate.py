"""Evaluate and compare value estimators."""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from src.config import ExperimentConfig
from src.env_utils import ALGORITHM_MAP, ESTIMATOR_CLASSES


N_CRITIC_SAMPLES = 16


def compute_critic_values(policy_path: Path, algorithm: str, observations: np.ndarray) -> np.ndarray:
    """Estimate V(s) from the policy's critic network.

    For on-policy algorithms (PPO/A2C) that learn V(s), calls the value network directly.
    For off-policy algorithms (SAC/TD3) that learn Q(s,a), samples actions from the policy
    and averages Q(s, a_i) to approximate V(s).

    Returns array of shape (n_states,) with value estimates, or None if critic unavailable.
    """
    AlgorithmClass = ALGORITHM_MAP[algorithm]
    model = AlgorithmClass.load(policy_path, device="cpu")

    observations = np.array(observations, dtype=np.float32)
    obs_tensor = torch.as_tensor(observations)

    with torch.no_grad():
        if algorithm in ("PPO", "A2C"):
            values = model.policy.predict_values(obs_tensor)
            return values.squeeze(-1).numpy()

        # SAC/TD3: sample actions and average Q values
        q_sum = torch.zeros(len(observations))
        for _ in range(N_CRITIC_SAMPLES):
            actions, _ = model.predict(observations, deterministic=False)
            actions_tensor = torch.as_tensor(actions, dtype=torch.float32)
            # SB3 critic returns a tuple of Q-networks; average across them
            q_values = model.critic(obs_tensor, actions_tensor)
            q_mean = torch.stack(q_values).mean(dim=0).squeeze(-1)
            q_sum += q_mean
        return (q_sum / N_CRITIC_SAMPLES).numpy()


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


def load_paired_states(data_dir: Path) -> Dict:
    """Load paired states data.

    Args:
        data_dir: Directory containing data files

    Returns:
        Dictionary containing paired states data, or None if not found
    """
    paired_states_path = data_dir / "paired_states.npz"
    if not paired_states_path.exists():
        return None

    paired_data = np.load(paired_states_path, allow_pickle=True)
    return dict(paired_data)


def compute_ground_truth_returns(eval_batch: Dict, gamma: float) -> pd.DataFrame:
    """Compute ground truth discounted returns for each state.

    For each state s_t at timestep t in an episode, computes:
        Return_t = Σ γ^(τ-t) * r_τ for τ from t to end of episode

    Args:
        eval_batch: Evaluation batch data with rewards
        gamma: Discount factor

    Returns:
        DataFrame with columns: state_idx, episode_idx, timestep_in_episode,
                               episode_length, is_truncated, ground_truth_return
    """
    eval_obs_list = eval_batch['observations']
    rewards_list = eval_batch['rewards']
    truncated_array = eval_batch.get('truncated', np.zeros(len(eval_obs_list), dtype=bool))

    ground_truth_data = []
    state_idx = 0

    for ep_idx, (obs_array, rewards_array) in enumerate(zip(eval_obs_list, rewards_list)):
        episode_length = len(rewards_array)
        is_truncated = bool(truncated_array[ep_idx])

        # Compute discounted return from each timestep
        for t in range(episode_length):
            # Return_t = sum of discounted future rewards from timestep t
            discounted_return = 0.0
            for tau in range(t, episode_length):
                discounted_return += (gamma ** (tau - t)) * rewards_array[tau]

            ground_truth_data.append({
                'state_idx': state_idx,
                'episode_idx': ep_idx,
                'timestep_in_episode': t,
                'episode_length': episode_length,
                'is_truncated': is_truncated,
                'ground_truth_return': float(discounted_return)
            })
            state_idx += 1

    return pd.DataFrame(ground_truth_data)


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
    truncated_array = eval_batch.get('truncated', np.zeros(len(eval_obs_list), dtype=bool))

    # Create episode metadata mapping for each state
    episode_indices = []
    timesteps_in_episode = []
    episode_lengths = []
    is_truncated_per_state = []

    for ep_idx, obs_array in enumerate(eval_obs_list):
        ep_length = len(obs_array)
        is_truncated = bool(truncated_array[ep_idx])

        episode_indices.extend([ep_idx] * ep_length)
        timesteps_in_episode.extend(list(range(ep_length)))
        episode_lengths.extend([ep_length] * ep_length)
        is_truncated_per_state.extend([is_truncated] * ep_length)

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

        # predict() automatically adds reward_offset for centered models
        values = estimator.predict(eval_obs_flat)

        for state_idx in range(n_states):
            predictions.append({
                'state_idx': state_idx,
                'episode_idx': episode_indices[state_idx],
                'timestep_in_episode': timesteps_in_episode[state_idx],
                'episode_length': episode_lengths[state_idx],
                'is_truncated': is_truncated_per_state[state_idx],
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

        # Directory paths for dashboard navigation
        'data_dir': str(config.get_data_dir()),
        'results_dir': str(results_dir),
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


def generate_paired_predictions_for_n_episodes(config: ExperimentConfig,
                                               paired_data: Dict, results_dir: Path,
                                               method_name: str, n_episodes: int,
                                               n_ep_dir: Path, device: str = "cpu"):
    """Generate predictions for paired states from a single method/n_episodes combination.

    Args:
        config: Experiment configuration
        paired_data: Paired states data
        results_dir: Root results directory
        method_name: Name of the method
        n_episodes: Number of episodes used for training
        n_ep_dir: Directory containing estimator models
        device: Device to use for inference

    Returns:
        Path to saved predictions file (relative to results_dir), or None if no predictions
    """
    state1_obs = paired_data['state1_obs']
    state2_obs = paired_data['state2_obs']
    n_pairs = len(state1_obs)

    # Stack object arrays into proper 2D arrays (same as concatenate for eval_obs_list)
    state1_obs_stacked = np.vstack(state1_obs)
    state2_obs_stacked = np.vstack(state2_obs)

    # Combine all states (s1 and s2) into a single array for batch prediction
    all_states = np.vstack([state1_obs_stacked, state2_obs_stacked])

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

        # predict() automatically adds reward_offset for centered models
        values = estimator.predict(all_states)

        # Split predictions back into s1 and s2
        s1_values = values[:n_pairs]
        s2_values = values[n_pairs:]

        # Store predictions for each pair
        for pair_idx in range(n_pairs):
            predictions.append({
                'pair_idx': pair_idx,
                'batch_name': batch_name,
                's1_predicted': s1_values[pair_idx],
                's2_predicted': s2_values[pair_idx],
                'diff_predicted': s1_values[pair_idx] - s2_values[pair_idx]
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
    predictions_file = output_method_dir / "paired_predictions.parquet"
    df.to_parquet(predictions_file, index=False)

    # Create metadata
    data_metadata = sample_metadata.get('data_metadata', {})
    policy_metadata = data_metadata.get('policy_metadata', {})

    predictions_metadata = {
        'experiment_id': config.experiment_id,
        'method': method_name,
        'n_episodes': n_episodes,
        'n_batches': len(df['batch_name'].unique()),
        'n_pairs': n_pairs,
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

        # Directory paths for dashboard navigation
        'data_dir': str(config.get_data_dir()),
        'results_dir': str(results_dir),
    }

    predictions_metadata['policy_display_name'] = None

    metadata_file = output_method_dir / "paired_predictions_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(predictions_metadata, f, indent=2)

    print(f"      Saved: {predictions_file.relative_to(results_dir)}")
    print(f"        ({len(df)} predictions = {n_pairs} pairs × "
          f"{len(df['batch_name'].unique())} batches)")

    # Free memory
    del df
    del predictions

    return str(predictions_file.relative_to(results_dir))


def _predictions_worker(task):
    """Worker: load eval batch from disk and generate predictions for one (method, n_episodes)."""
    config, data_dir, results_dir, method_name, n_episodes, n_ep_dir, device = task
    eval_batch = dict(np.load(data_dir / "batch_eval.npz", allow_pickle=True))
    return generate_predictions_for_n_episodes(
        config, eval_batch, results_dir, method_name, n_episodes, n_ep_dir, device
    )


def generate_predictions(config: ExperimentConfig, eval_batch: Dict,
                        device: str = "cpu", n_jobs: int = 1):
    """Generate predictions from all trained models with metadata."""
    eval_obs_list = eval_batch['observations']
    n_states = len(np.concatenate(eval_obs_list, axis=0))

    # Build task list across all methods
    tasks = []
    for method_config in config.value_estimators.method_configs:
        method_name = method_config.name
        estimator_dir = config.get_estimator_dir(method_config)
        results_dir = config.get_eval_dir(method_config) / "results"

        if not estimator_dir.exists():
            print(f"  Estimator directory not found: {estimator_dir}, skipping")
            continue
        n_episodes_dirs = sorted(
            [d for d in estimator_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name)
        )
        for n_ep_dir in n_episodes_dirs:
            tasks.append((config, config.get_data_dir(), results_dir, method_name, int(n_ep_dir.name), n_ep_dir, device))

    print(f"\nGenerating predictions on {n_states} states from {len(eval_obs_list)} episodes...")
    print(f"  {len(tasks)} (method, n_episodes) combinations, n_jobs={n_jobs}")

    if n_jobs == 1:
        results = [_predictions_worker(t) for t in tqdm(tasks, desc="Processing tasks")]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(tqdm(executor.map(_predictions_worker, tasks),
                               total=len(tasks),
                               desc="Processing tasks"))

    return [r for r in results if r is not None]


def _paired_predictions_worker(task):
    """Worker: load paired data from disk and generate predictions for one (method, n_episodes)."""
    config, data_dir, results_dir, method_name, n_episodes, n_ep_dir, device = task
    paired_data = dict(np.load(data_dir / "paired_states.npz", allow_pickle=True))
    return generate_paired_predictions_for_n_episodes(
        config, paired_data, results_dir, method_name, n_episodes, n_ep_dir, device
    )


def generate_paired_predictions(config: ExperimentConfig,
                                paired_data: Dict, device: str = "cpu",
                                n_jobs: int = 1):
    """Generate predictions for paired states from all trained models."""
    n_pairs = len(paired_data['pair_indices'])

    # Build task list across all methods
    tasks = []
    for method_config in config.value_estimators.method_configs:
        method_name = method_config.name
        estimator_dir = config.get_estimator_dir(method_config)
        results_dir = config.get_eval_dir(method_config) / "results"

        if not estimator_dir.exists():
            print(f"  Estimator directory not found: {estimator_dir}, skipping")
            continue
        n_episodes_dirs = sorted(
            [d for d in estimator_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name)
        )
        for n_ep_dir in n_episodes_dirs:
            tasks.append((config, config.get_data_dir(), results_dir, method_name, int(n_ep_dir.name), n_ep_dir, device))

    print(f"\nGenerating paired state predictions for {n_pairs} pairs...")
    print(f"  {len(tasks)} (method, n_episodes) combinations, n_jobs={n_jobs}")

    if n_jobs == 1:
        results = [_paired_predictions_worker(t) for t in tqdm(tasks, desc="Processing paired tasks")]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(tqdm(executor.map(_paired_predictions_worker, tasks),
                               total=len(tasks),
                               desc="Processing paired tasks"))

    return [r for r in results if r is not None]


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare value estimators")
    parser.add_argument("--config", type=Path, required=True,
                       help="Path to config YAML file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval-only", action="store_true",
                      help="Only run evaluation on the standard test set")
    mode.add_argument("--paired-only", action="store_true",
                      help="Only run evaluation on the paired states dataset")
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of parallel workers for prediction generation (default: 1)")
    args = parser.parse_args()

    run_eval = not args.paired_only
    run_paired = not args.eval_only

    config = ExperimentConfig.from_yaml(args.config)

    data_dir = config.get_data_dir()

    print(f"\nEvaluating experiment: {config.experiment_id}")
    print(f"Data directory: {data_dir}")
    print(f"Methods: {[mc.name for mc in config.value_estimators.method_configs]}")
    print(f"Batches: {config.data_generation.n_batches}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    prediction_files = []
    if run_eval:
        print("\nLoading evaluation batch...")
        eval_batch = load_evaluation_batch(data_dir)
        print(f"Loaded evaluation batch with {len(eval_batch['observations'])} episodes")

        print("\nComputing ground truth returns...")
        gamma = config.value_estimators.training.gamma
        ground_truth_df = compute_ground_truth_returns(eval_batch, gamma)

        # Compute critic values from the trained policy
        policy_path = config.get_policy_dir() / "policy_final.zip"
        if policy_path.exists():
            print("\nComputing critic values from trained policy...")
            eval_obs_flat = np.concatenate(eval_batch['observations'], axis=0)
            critic_values = compute_critic_values(policy_path, config.policy.algorithm, eval_obs_flat)
            if critic_values is not None:
                ground_truth_df['critic_value'] = critic_values
                print(f"  Added critic values (mean={critic_values.mean():.4f})")
        else:
            print(f"\nPolicy not found at {policy_path}, skipping critic values")

        # Save ground truth to each method's eval dir
        for method_config in config.value_estimators.method_configs:
            results_dir = config.get_eval_dir(method_config) / "results"
            ground_truth_dir = results_dir / "ground_truth"
            ground_truth_dir.mkdir(parents=True, exist_ok=True)
            ground_truth_df.to_parquet(ground_truth_dir / "ground_truth_returns.parquet", index=False)

        print(f"Saved ground truth returns ({len(ground_truth_df)} states, {ground_truth_df['episode_idx'].nunique()} episodes, gamma={gamma})")

        prediction_files = generate_predictions(config, eval_batch, device, n_jobs=args.n_jobs)

    paired_prediction_files = []
    if run_paired:
        print("\n" + "="*80)
        print("Checking for paired states data...")
        paired_data = load_paired_states(data_dir)

        if paired_data is not None:
            print(f"Found paired states data with {len(paired_data['pair_indices'])} pairs")
            paired_prediction_files = generate_paired_predictions(config, paired_data, device, n_jobs=args.n_jobs)
            print(f"\nGenerated {len(paired_prediction_files)} paired prediction files")
        else:
            print("No paired states data found. Skipping paired state evaluation.")
            print("To generate paired states data, run: uv run -m src.generate_data --config <config> --generate-paired")

    # Save summary to each method's eval dir
    for method_config in config.value_estimators.method_configs:
        results_dir = config.get_eval_dir(method_config) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            'experiment_id': config.experiment_id,
            'prediction_files': prediction_files,
            'n_files': len(prediction_files),
            'paired_prediction_files': paired_prediction_files,
            'n_paired_files': len(paired_prediction_files),
            'created_at': datetime.now().isoformat()
        }
        with open(results_dir / "evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
