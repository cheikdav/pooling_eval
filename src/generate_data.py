"""Generate episode data using a trained policy."""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import gymnasium as gym
import scipy.stats as stats
import multiprocessing as mp
from functools import partial

from src.config import ExperimentConfig
from src.env_utils import ALGORITHM_MAP, create_vec_env


def collect_episodes_parallel(env, model, n_episodes: int, deterministic: bool = False,
                              use_vec_normalize: bool = False, seed: int = None) -> List[Dict[str, np.ndarray]]:
    """Collect multiple episodes in parallel using vectorized environments.

    Runs n_envs episodes in parallel, waits for all to complete, then repeats.
    n_envs is capped at n_episodes to avoid bias toward short episodes.

    Args:
        env: VecEnv with n_envs parallel environments
        model: Trained SB3 model
        n_episodes: Total number of episodes to collect
        deterministic: Whether to use deterministic actions
        use_vec_normalize: Whether VecNormalize is used
        seed: If set, seed the environment on the first reset for reproducibility

    Returns:
        List of episode dictionaries, each containing:
            - observations: (T, obs_dim) array
            - actions: (T,) or (T, act_dim) array
            - rewards: (T,) array
            - dones: (T,) array
            - next_observations: (T, obs_dim) array
    """
    n_envs = env.num_envs
    completed_episodes = []
    first_reset = True

    while len(completed_episodes) < n_episodes:
        # Use at most as many envs as episodes still needed
        n_active = min(n_envs, n_episodes - len(completed_episodes))

        episode_data = [
            {'observations': [], 'actions': [], 'rewards': [], 'dones': [], 'next_observations': []}
            for _ in range(n_active)
        ]
        active = [True] * n_active

        if first_reset and seed is not None:
            env.seed(seed)
            first_reset = False
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Run until all n_active envs have completed one episode
        while any(active):
            if use_vec_normalize:
                original_obs = env.get_original_obs()
            else:
                original_obs = obs

            for i in range(n_active):
                if active[i]:
                    episode_data[i]['observations'].append(original_obs[i].copy())

            action, _ = model.predict(obs, deterministic=deterministic)

            step_result = env.step(action)
            next_obs = step_result[0]
            rewards = step_result[1]
            dones = step_result[2]
            infos = step_result[3] if len(step_result) > 3 else [{} for _ in range(n_envs)]

            if use_vec_normalize:
                original_next_obs = env.get_original_obs()
            else:
                original_next_obs = next_obs

            for i in range(n_active):
                if active[i]:
                    episode_data[i]['actions'].append(action[i].copy() if isinstance(action[i], np.ndarray) else action[i])
                    episode_data[i]['rewards'].append(float(rewards[i]))
                    episode_data[i]['dones'].append(bool(dones[i]))
                    episode_data[i]['next_observations'].append(original_next_obs[i].copy())

                    if dones[i]:
                        info = infos[i] if isinstance(infos, (list, tuple)) else {}
                        truncated = info.get('TimeLimit.truncated', False)
                        completed_episodes.append({
                            'observations': np.array(episode_data[i]['observations']),
                            'actions': np.array(episode_data[i]['actions']),
                            'rewards': np.array(episode_data[i]['rewards']),
                            'dones': np.array(episode_data[i]['dones']),
                            'next_observations': np.array(episode_data[i]['next_observations']),
                            'truncated': truncated,
                        })
                        active[i] = False

            obs = next_obs

    np.random.shuffle(completed_episodes)
    return completed_episodes[:n_episodes]


def episodes_to_batch(episodes: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Convert a list of episode dicts into batch dict format."""
    batch_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'next_observations': [],
        'truncated': [],
        'episode_lengths': [],
        'episode_returns': [],
    }

    for episode in episodes:
        for key in episode:
            batch_data[key].append(episode[key])
        batch_data['episode_lengths'].append(len(episode['rewards']))
        batch_data['episode_returns'].append(episode['rewards'].sum())

    for key in batch_data.keys():
        if key not in ['episode_lengths', 'episode_returns', 'truncated']:
            batch_data[key] = np.array(batch_data[key], dtype=object)
    batch_data['truncated'] = np.array(batch_data['truncated'], dtype=bool)
    batch_data['episode_lengths'] = np.array(batch_data['episode_lengths'])
    batch_data['episode_returns'] = np.array(batch_data['episode_returns'])
    return batch_data


def save_and_log_batch(batch_data: dict, output_path: Path, batch_name: str) -> dict:
    """Save batch data and return statistics.

    Args:
        batch_data: Dictionary containing batch data
        output_path: Path to save the NPZ file
        batch_name: Name for logging

    Returns:
        Dictionary of statistics
    """
    np.savez_compressed(output_path, **batch_data)

    mean_return = batch_data['episode_returns'].mean()
    std_return = batch_data['episode_returns'].std()
    mean_length = batch_data['episode_lengths'].mean()

    print(f"  Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"  Mean episode length: {mean_length:.1f}\n")

    return {
        'name': batch_name,
        'mean_return': mean_return,
        'std_return': std_return,
        'mean_length': mean_length,
    }


def is_classic_control_env(env_name: str) -> bool:
    """Check if environment is Classic Control (supports state setting from observation)."""
    classic_control = ['CartPole', 'Acrobot', 'MountainCar', 'Pendulum']
    return any(name in env_name for name in classic_control)


def is_mujoco_env(env_name: str) -> bool:
    """Check if environment is MuJoCo-based."""
    mujoco_envs = ['Hopper', 'Walker2d', 'HalfCheetah', 'Ant', 'Humanoid', 'Swimmer', 'Reacher']
    return any(name in env_name for name in mujoco_envs)


def get_full_state(env):
    """Get full environment state that can be used for restoration.

    Returns:
        For Classic Control: observation (which is the full state)
        For MuJoCo: tuple of (qpos, qvel)
    """
    if hasattr(env.unwrapped, 'state'):
        # Classic Control - observation is the state
        return env.unwrapped.state.copy()
    elif hasattr(env.unwrapped, 'data'):
        # MuJoCo - need full simulator state
        return (env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy())
    else:
        raise ValueError(f"Environment {type(env)} does not support state extraction")


def restore_full_state(env, state):
    """Restore environment to a saved state.

    Args:
        env: Gymnasium environment
        state: For Classic Control - observation array
               For MuJoCo - tuple of (qpos, qvel)
    """
    if hasattr(env.unwrapped, 'state'):
        # Classic Control
        env.unwrapped.state = state.copy() if isinstance(state, np.ndarray) else state
    elif hasattr(env.unwrapped, 'data'):
        # MuJoCo
        qpos, qvel = state
        env.unwrapped.set_state(qpos, qvel)
    else:
        raise ValueError(f"Environment {type(env)} does not support state restoration")


def get_obs_from_env(env) -> np.ndarray:
    """Get current observation from env after state restoration."""
    if hasattr(env.unwrapped, '_get_obs'):
        return env.unwrapped._get_obs()
    elif hasattr(env.unwrapped, 'state'):
        state = env.unwrapped.state
        return state.copy() if hasattr(state, 'copy') else np.array(state)
    else:
        raise ValueError(f"Cannot get observation from {type(env)}")


def generate_trajectory_from_state(env, model, full_state, gamma: float = 0.99, deterministic: bool = False, vec_normalize=None, max_steps: int = None) -> float:
    """Generate a single trajectory from an initial state and return discounted return.

    Trajectories are capped at max_steps (default: 10/(1-gamma)) since further
    steps contribute < e^{-10} ≈ 0.005% to the discounted return.
    """
    if max_steps is None:
        max_steps = int(10 / (1 - gamma))

    env.reset()  # reset TimeLimit counter and all wrapper state
    restore_full_state(env, full_state)

    # Get obs fresh from env after restoration to avoid any stale-obs mismatch
    obs = get_obs_from_env(env)
    if vec_normalize is not None:
        obs = vec_normalize.normalize_obs(obs[None])[0]

    done = False
    truncated = False
    episode_return = 0.0
    undiscounted_return = 0.0
    discount = 1.0
    step = 0

    while not (done or truncated) and step < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, _ = env.step(action)
        if vec_normalize is not None:
            obs = vec_normalize.normalize_obs(obs[None])[0]
        episode_return += discount * reward
        undiscounted_return += reward
        discount *= gamma
        step += 1

    return episode_return, undiscounted_return


def sample_state_pairs(env, config: ExperimentConfig):
    """Sample state pairs by resetting environment.

    Args:
        env: Gymnasium environment
        config: Experiment configuration

    Returns:
        List of tuples: (obs1, obs2, state1, state2)
        where obs is the observation and state is the full restorable state
    """
    pairs = []

    for _ in range(config.evaluation.paired_states_n_pairs):
        # Reset twice to get two different initial states
        reset_result = env.reset()
        obs1 = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        state1 = get_full_state(env)

        reset_result = env.reset()
        obs2 = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        state2 = get_full_state(env)

        pairs.append((obs1, obs2, state1, state2))

    return pairs


def compute_confidence_interval(returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval using t-distribution.

    Args:
        returns: Array of returns
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound)
    """
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    n = len(returns)

    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * std / np.sqrt(n)

    return mean - margin, mean + margin


def _process_single_state(state_data: Tuple) -> Dict:
    """Worker function to process a single state in parallel.

    Args:
        state_data: Tuple containing (pair_idx, state_idx, obs, full_state, env_name,
                   policy_path, algorithm, max_episode_steps, n_trajectories, gamma,
                   vec_normalize_path)

    Returns:
        Dictionary with results for this state
    """
    pair_idx, state_idx, obs, full_state, env_name, policy_path, algorithm, max_episode_steps, n_trajectories, gamma, vec_normalize_path = state_data

    # Create environment for this worker
    env = gym.make(env_name, max_episode_steps=max_episode_steps)

    # Load model for this worker
    AlgorithmClass = ALGORITHM_MAP[algorithm]
    model = AlgorithmClass.load(policy_path)

    # Load VecNormalize if available
    vec_normalize = None
    if vec_normalize_path is not None and Path(vec_normalize_path).exists():
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        dummy_env = DummyVecEnv([lambda: gym.make(env_name, max_episode_steps=max_episode_steps)])
        vec_normalize = VecNormalize.load(str(vec_normalize_path), dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    # Generate trajectories from this state
    returns = []
    total_undiscounted = 0.0
    for _ in range(n_trajectories):
        ret, undiscount_ret = generate_trajectory_from_state(env, model, full_state, gamma=gamma, deterministic=False, vec_normalize=vec_normalize)
        returns.append(ret)
        total_undiscounted += undiscount_ret

    returns = np.array(returns)
    avg_undiscounted = total_undiscounted / n_trajectories

    # Compute statistics
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    ci_lower, ci_upper = compute_confidence_interval(returns)

    env.close()

    return {
        'pair_idx': pair_idx,
        'state_idx': state_idx,
        'obs': obs,
        'returns': returns,
        'mean': mean_ret,
        'std': std_ret,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'undiscounted': avg_undiscounted,
    }


def generate_paired_states(config: ExperimentConfig, output_dir: Path, gamma: float, vec_normalize_path: Path = None, policy_path: Path = None, n_workers: int = None):
    """Generate paired state evaluations with ground truth confidence intervals.

    Args:
        config: Experiment configuration
        output_dir: Directory where to save results
        gamma: Discount factor for value computation
        vec_normalize_path: Optional path to VecNormalize file
        policy_path: Optional path to policy file (defaults to experiments/<exp_id>/policy/policy_final.zip)
        n_workers: Number of parallel workers (None = use all CPUs)
    """
    if not (is_classic_control_env(config.environment.name) or is_mujoco_env(config.environment.name)):
        print(f"\nSkipping paired state generation: {config.environment.name} is not a supported environment")
        return

    print(f"\nGenerating paired state evaluations")
    print(f"Environment: {config.environment.name}")
    print(f"Number of pairs: {config.evaluation.paired_states_n_pairs}")
    print(f"Trajectories per state: {config.evaluation.paired_states_n_trajectories}")

    # Sample state pairs
    env = gym.make(config.environment.name, max_episode_steps=config.environment.max_episode_steps)
    env.reset(seed=config.evaluation.paired_states_seed)

    print("Sampling state pairs by resetting environment...")
    state_pairs = sample_state_pairs(env, config)
    env.close()

    # Use default policy path if not provided
    if policy_path is None:
        policy_path = config.get_policy_dir() / "policy_final.zip"

    # Flatten state pairs into individual states for parallel processing
    all_states = []
    for pair_idx, (obs1, obs2, state1, state2) in enumerate(state_pairs):
        all_states.append((
            pair_idx, 0, obs1, state1, config.environment.name,
            str(policy_path),
            config.policy.algorithm, config.environment.max_episode_steps,
            config.evaluation.paired_states_n_trajectories, gamma,
            str(vec_normalize_path) if vec_normalize_path else None
        ))
        all_states.append((
            pair_idx, 1, obs2, state2, config.environment.name,
            str(policy_path),
            config.policy.algorithm, config.environment.max_episode_steps,
            config.evaluation.paired_states_n_trajectories, gamma,
            str(vec_normalize_path) if vec_normalize_path else None
        ))

    print(f"Generating trajectories from {len(all_states)} states in parallel (n_workers={n_workers or 'all CPUs'})...")

    # Process all states in parallel
    with mp.Pool(processes=n_workers) as pool:
        state_results = list(tqdm(
            pool.imap(_process_single_state, all_states),
            total=len(all_states),
            desc="Processing states"
        ))

    # Reconstruct paired results
    results = {
        'pair_indices': [],
        'state1_obs': [],
        'state2_obs': [],
        's1_returns': [],
        's2_returns': [],
        's1_mean': [],
        's1_std': [],
        's1_ci_lower': [],
        's1_ci_upper': [],
        's2_mean': [],
        's2_std': [],
        's2_ci_lower': [],
        's2_ci_upper': [],
        'diff_mean': [],
        'diff_std': [],
        'diff_ci_lower': [],
        'diff_ci_upper': [],
    }
    undiscounted_returns = []

    # Group results by pair
    for pair_idx in range(config.evaluation.paired_states_n_pairs):
        s1_result = state_results[pair_idx * 2]
        s2_result = state_results[pair_idx * 2 + 1]

        # Compute difference statistics
        diff_returns = s1_result['returns'] - s2_result['returns']
        diff_mean = np.mean(diff_returns)
        diff_std = np.std(diff_returns, ddof=1)
        diff_ci_lower, diff_ci_upper = compute_confidence_interval(diff_returns)

        results['pair_indices'].append(pair_idx)
        results['state1_obs'].append(s1_result['obs'])
        results['state2_obs'].append(s2_result['obs'])
        results['s1_returns'].append(s1_result['returns'])
        results['s2_returns'].append(s2_result['returns'])
        results['s1_mean'].append(s1_result['mean'])
        results['s1_std'].append(s1_result['std'])
        results['s1_ci_lower'].append(s1_result['ci_lower'])
        results['s1_ci_upper'].append(s1_result['ci_upper'])
        results['s2_mean'].append(s2_result['mean'])
        results['s2_std'].append(s2_result['std'])
        results['s2_ci_lower'].append(s2_result['ci_lower'])
        results['s2_ci_upper'].append(s2_result['ci_upper'])
        results['diff_mean'].append(diff_mean)
        results['diff_std'].append(diff_std)
        results['diff_ci_lower'].append(diff_ci_lower)
        results['diff_ci_upper'].append(diff_ci_upper)

        undiscounted_returns.append(s1_result['undiscounted'])
        undiscounted_returns.append(s2_result['undiscounted'])

    for key in results:
        if key in ['s1_returns', 's2_returns']:
            results[key] = np.array(results[key], dtype=object)
        else:
            results[key] = np.array(results[key])

    output_path = output_dir / "paired_states.npz"
    np.savez_compressed(output_path, **results)

    print(f"\nPaired state data saved to {output_path}")
    print(f"Sample statistics:")
    print(f"  S1 mean discounted returns: {np.mean(results['s1_mean']):.2f} ± {np.std(results['s1_mean']):.2f}")
    print(f"  S2 mean discounted returns: {np.mean(results['s2_mean']):.2f} ± {np.std(results['s2_mean']):.2f}")
    print(f"  Mean undiscounted returns (averaged over trajectories): {np.mean(undiscounted_returns):.2f} ± {np.std(undiscounted_returns):.2f}")
    print(f"  Mean difference (V(s1) - V(s2)): {np.mean(results['diff_mean']):.2f} ± {np.std(results['diff_mean']):.2f}")
    print(f"  Average CI width for differences: {np.mean(results['diff_ci_upper'] - results['diff_ci_lower']):.2f}")


def generate_data(config: ExperimentConfig, policy_path: Path, output_dir: Path, start_batch_idx: int = 0, end_batch_idx: int = None, generate_paired: bool = False, eval_only: bool = False, n_workers: int = None):
    """Generate n batches of k episodes using trained policy.

    Args:
        config: Experiment configuration
        policy_path: Path to trained policy (.zip file)
        output_dir: Directory to save generated data
        start_batch_idx: Skip batches before this index (for resuming interrupted runs)
        end_batch_idx: Stop after this index (exclusive, None = generate all batches)
        generate_paired: Whether to generate paired state evaluations
        eval_only: If True, only generate the eval batch (skips all other batches)
        n_workers: Number of parallel workers for paired state generation (None = use all CPUs)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_metadata = {}
    policy_metadata_path = policy_path.parent / "policy_metadata.json"
    if policy_metadata_path.exists():
        with open(policy_metadata_path, 'r') as f:
            policy_metadata = json.load(f)

    if config.policy.algorithm not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {config.policy.algorithm}")

    AlgorithmClass = ALGORITHM_MAP[config.policy.algorithm]
    model = AlgorithmClass.load(policy_path)

    vec_normalize_path = policy_path.parent / "vec_normalize.pkl" if config.policy.use_vec_normalize else None
    env, use_vec_normalize = create_vec_env(
        config,
        n_envs=config.data_generation.n_envs,
        use_monitor=False,
        vec_normalize_path=vec_normalize_path
    )

    if use_vec_normalize:
        print(f"Loaded VecNormalize stats from {vec_normalize_path}")

    print(f"\nGenerating data using policy from {policy_path}")
    print(f"Algorithm: {config.policy.algorithm}")
    print(f"Environment: {config.environment.name}")
    print(f"VecNormalize: {use_vec_normalize}")
    print(f"Deterministic policy: {config.data_generation.deterministic_policy}")
    print(f"Output directory: {output_dir}\n")

    all_batch_stats = []
    current_seed = config.seed
    val_eps = config.data_generation.validation_episodes_per_batch

    def collect_and_save(batch_name, n_train, n_val, seed, skip=False):
        """Collect train + validation episodes together, split, and save."""
        if skip or (eval_only and batch_name != "batch_eval"):
            print(f"  Skipping {batch_name}\n")
            return n_train + n_val

        total = n_train + n_val
        print(f"Collecting {batch_name} ({n_train}+{n_val} val = {total} episodes)")
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        episodes = collect_episodes_parallel(env, model, total,
                                             config.data_generation.deterministic_policy,
                                             use_vec_normalize, seed=seed)

        train_data = episodes_to_batch(episodes[:n_train])
        stats = save_and_log_batch(train_data, output_dir / f"{batch_name}.npz", batch_name)
        stats['batch_seed'] = seed
        all_batch_stats.append(stats)

        if n_val > 0:
            val_name = f"{batch_name}_validation"
            val_data = episodes_to_batch(episodes[n_train:])
            val_stats = save_and_log_batch(val_data, output_dir / f"{val_name}.npz", val_name)
            val_stats['batch_seed'] = seed
            all_batch_stats.append(val_stats)

        return total

    total_episodes = 0

    # Tuning batch (+ validation)
    if config.data_generation.tuning_episodes > 0:
        total_episodes += collect_and_save(
            "batch_tuning", config.data_generation.tuning_episodes, val_eps, current_seed)
        current_seed += 1

    # Regular batches (+ validation each)
    for i in range(config.data_generation.n_batches):
        skip = i < start_batch_idx or (end_batch_idx is not None and i >= end_batch_idx)
        total_episodes += collect_and_save(
            f"batch_{i}", config.data_generation.episodes_per_batch, val_eps, current_seed, skip=skip)
        current_seed += 1

    # Eval batch (no validation pair)
    if config.evaluation.eval_episodes > 0:
        total_episodes += collect_and_save(
            "batch_eval", config.evaluation.eval_episodes, 0, current_seed)
        current_seed += 1

    stats_file = output_dir / "data_statistics.npz"
    np.savez(
        stats_file,
        batch_stats=all_batch_stats,
        config_seed=config.seed,
        total_episodes=total_episodes,
    )

    # Save data generation metadata
    data_metadata = {
        'policy_path': str(policy_path),
        'deterministic_policy': config.data_generation.deterministic_policy,
        'total_episodes': int(total_episodes),
        'n_batches': config.data_generation.n_batches,
        'episodes_per_batch': config.data_generation.episodes_per_batch,
        'tuning_episodes': config.data_generation.tuning_episodes,
        'validation_episodes_per_batch': config.data_generation.validation_episodes_per_batch,
        'eval_episodes': config.evaluation.eval_episodes,
        'seed': config.seed,
        'policy_metadata': policy_metadata,
    }

    metadata_path = output_dir / "data_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(data_metadata, f, indent=2)
    print(f"Data generation metadata saved to {metadata_path}")

    print(f"Data generation complete! Files saved to {output_dir}")
    print(f"Total episodes: {total_episodes}")

    env.close()

    # Generate paired state evaluations if requested
    if generate_paired:
        generate_paired_states(config, output_dir, gamma=config.value_estimators.training.gamma, vec_normalize_path=vec_normalize_path, policy_path=policy_path, n_workers=n_workers)


def main():
    parser = argparse.ArgumentParser(description="Generate episode data using trained policy")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument("--policy-path", type=Path, default=None,
                       help="Path to trained policy (default: experiments/<experiment_id>/policy/policy_final.zip)")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (default: experiments/<experiment_id>/data)")
    parser.add_argument("--start-batch-idx", type=int, default=0,
                       help="Skip batches before this index (for resuming interrupted runs)")
    parser.add_argument("--end-batch-idx", type=int, default=None,
                       help="Stop after this batch index (exclusive, default: generate all batches)")
    parser.add_argument("--generate-paired", action="store_true",
                       help="Generate paired state evaluations with ground truth CIs (overrides config)")
    parser.add_argument("--no-generate-paired", action="store_true",
                       help="Skip paired state generation (overrides config)")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only generate the eval batch (skips tuning, regular, and validation batches)")
    parser.add_argument("--n-workers", type=int, default=None,
                       help="Number of parallel workers for paired state generation (default: use all CPUs)")
    args = parser.parse_args()

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)

    # Determine whether to generate paired states
    # Command-line flags override config setting
    if args.generate_paired:
        generate_paired = True
    elif args.no_generate_paired:
        generate_paired = False
    else:
        generate_paired = config.evaluation.paired_states_n_pairs > 0

    # Set default paths
    if args.policy_path is None:
        policy_path = config.get_policy_dir() / "policy_final.zip"
    else:
        policy_path = args.policy_path

    if args.output_dir is None:
        output_dir = config.get_data_dir()
    else:
        output_dir = args.output_dir

    # Save config to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Generate data
    generate_data(config, policy_path, output_dir,
                 start_batch_idx=args.start_batch_idx,
                 end_batch_idx=args.end_batch_idx,
                 generate_paired=generate_paired,
                 eval_only=args.eval_only,
                 n_workers=args.n_workers)


if __name__ == "__main__":
    main()
