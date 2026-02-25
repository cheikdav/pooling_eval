"""Generate episode data using a trained policy."""

import argparse
import json
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, A2C, SAC, TD3
from tqdm import tqdm
from typing import Dict, List, Tuple
import gymnasium as gym
import scipy.stats as stats

from src.config import ExperimentConfig
from src.env_utils import create_vec_env


ALGORITHM_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


def collect_episodes_parallel(env, model, n_episodes: int, deterministic: bool = False,
                              use_vec_normalize: bool = False) -> List[Dict[str, np.ndarray]]:
    """Collect multiple episodes in parallel using vectorized environments.

    Args:
        env: VecEnv with n_envs parallel environments
        model: Trained SB3 model
        n_episodes: Total number of episodes to collect
        deterministic: Whether to use deterministic actions
        use_vec_normalize: Whether VecNormalize is used

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

    episode_data = {
        i: {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'next_observations': []
        }
        for i in range(n_envs)
    }

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    while len(completed_episodes) < n_episodes:
        if use_vec_normalize:
            original_obs = env.get_original_obs()
        else:
            original_obs = obs

        for i in range(n_envs):
            episode_data[i]['observations'].append(original_obs[i].copy())

        action, _ = model.predict(obs, deterministic=deterministic)

        step_result = env.step(action)
        next_obs = step_result[0]
        rewards = step_result[1]
        dones = step_result[2]

        if use_vec_normalize:
            original_next_obs = env.get_original_obs()
        else:
            original_next_obs = next_obs

        for i in range(n_envs):
            if episode_data[i]['observations']:
                episode_data[i]['actions'].append(action[i].copy() if isinstance(action[i], np.ndarray) else action[i])
                episode_data[i]['rewards'].append(float(rewards[i]))
                episode_data[i]['dones'].append(bool(dones[i]))
                episode_data[i]['next_observations'].append(original_next_obs[i].copy())

                if dones[i] and len(completed_episodes) < n_episodes:
                    completed_episodes.append({
                        'observations': np.array(episode_data[i]['observations']),
                        'actions': np.array(episode_data[i]['actions']),
                        'rewards': np.array(episode_data[i]['rewards']),
                        'dones': np.array(episode_data[i]['dones']),
                        'next_observations': np.array(episode_data[i]['next_observations']),
                    })

                    episode_data[i] = {
                        'observations': [],
                        'actions': [],
                        'rewards': [],
                        'dones': [],
                        'next_observations': []
                    }

        obs = next_obs

    return completed_episodes[:n_episodes]


def collect_batch(env, model, n_episodes: int, deterministic: bool = False,
                  batch_seed: int = None, use_vec_normalize: bool = False) -> Dict[str, List[np.ndarray]]:
    """Collect a batch of episodes using parallel environments.

    Args:
        env: VecEnv (SubprocVecEnv, optionally wrapped with VecNormalize)
        model: Trained SB3 model
        n_episodes: Number of episodes to collect
        deterministic: Whether to use deterministic actions
        batch_seed: Random seed for this batch
        use_vec_normalize: Whether VecNormalize is used

    Returns:
        Dictionary containing lists of arrays (one per episode):
            - observations: List of (T_i, obs_dim) arrays
            - actions: List of (T_i,) or (T_i, act_dim) arrays
            - rewards: List of (T_i,) arrays
            - dones: List of (T_i,) arrays
            - next_observations: List of (T_i, obs_dim) arrays
            - episode_lengths: (n_episodes,) array of episode lengths
            - episode_returns: (n_episodes,) array of total returns
    """
    if batch_seed is not None:
        np.random.seed(batch_seed)

    episodes = collect_episodes_parallel(env, model, n_episodes, deterministic, use_vec_normalize)

    batch_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'next_observations': [],
        'episode_lengths': [],
        'episode_returns': [],
    }

    for episode in episodes:
        for key in episode:
            batch_data[key].append(episode[key])

        batch_data['episode_lengths'].append(len(episode['rewards']))
        batch_data['episode_returns'].append(episode['rewards'].sum())

    for key in batch_data.keys():
        if key not in ['episode_lengths', 'episode_returns']:
            batch_data[key] = np.array(batch_data[key], dtype=object)
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


def generate_trajectory_from_state(env, model, full_state, gamma: float = 0.99, deterministic: bool = False, vec_normalize=None) -> float:
    """Generate a single trajectory from an initial state and return discounted return."""
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

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, _ = env.step(action)
        if vec_normalize is not None:
            obs = vec_normalize.normalize_obs(obs[None])[0]
        episode_return += discount * reward
        undiscounted_return += reward
        discount *= gamma

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

    for _ in range(config.paired_state.n_pairs):
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


def generate_paired_states(config: ExperimentConfig, model, output_dir: Path, gamma: float, vec_normalize_path: Path = None):
    """Generate paired state evaluations with ground truth confidence intervals.

    Args:
        config: Experiment configuration
        model: Trained SB3 model
        output_dir: Directory where to save results
        gamma: Discount factor for value computation
    """
    if not (is_classic_control_env(config.environment.name) or is_mujoco_env(config.environment.name)):
        print(f"\nSkipping paired state generation: {config.environment.name} is not a supported environment")
        return

    print(f"\nGenerating paired state evaluations")
    print(f"Environment: {config.environment.name}")
    print(f"Number of pairs: {config.paired_state.n_pairs}")
    print(f"Trajectories per state: {config.paired_state.n_trajectories_per_state}")

    # Load VecNormalize if available (needed for policies trained with observation normalization)
    vec_normalize = None
    if vec_normalize_path is not None and vec_normalize_path.exists():
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        dummy_env = DummyVecEnv([lambda: gym.make(config.environment.name)])
        vec_normalize = VecNormalize.load(str(vec_normalize_path), dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        print(f"Loaded VecNormalize from {vec_normalize_path}")

    env = gym.make(config.environment.name)
    env.reset(seed=config.paired_state.seed)

    print("Sampling state pairs by resetting environment...")
    state_pairs = sample_state_pairs(env, config)

    print("Generating trajectories from sampled states...")

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

    for pair_idx, (obs1, obs2, state1, state2) in enumerate(tqdm(state_pairs)):
        s1_returns = []
        total_undiscounted_return = 0.0
        for _ in range(config.paired_state.n_trajectories_per_state):
            ret, undiscount_ret = generate_trajectory_from_state(env, model, state1, gamma=gamma, deterministic=False, vec_normalize=vec_normalize)
            s1_returns.append(ret)
            total_undiscounted_return += undiscount_ret
            
        undiscounted_returns.append(total_undiscounted_return / config.paired_state.n_trajectories_per_state)

        s2_returns = []
        total_undiscounted_return = 0.0
        for _ in range(config.paired_state.n_trajectories_per_state):
            ret, undiscount_ret = generate_trajectory_from_state(env, model, state2, gamma=gamma, deterministic=False, vec_normalize=vec_normalize)
            s2_returns.append(ret)
            total_undiscounted_return += undiscount_ret
            
        undiscounted_returns.append(total_undiscounted_return / config.paired_state.n_trajectories_per_state)

        s1_returns = np.array(s1_returns)
        s2_returns = np.array(s2_returns)

        s1_mean = np.mean(s1_returns)
        s1_std = np.std(s1_returns, ddof=1)
        s1_ci_lower, s1_ci_upper = compute_confidence_interval(s1_returns)

        s2_mean = np.mean(s2_returns)
        s2_std = np.std(s2_returns, ddof=1)
        s2_ci_lower, s2_ci_upper = compute_confidence_interval(s2_returns)

        diff_returns = s1_returns - s2_returns
        diff_mean = np.mean(diff_returns)
        diff_std = np.std(diff_returns, ddof=1)
        diff_ci_lower, diff_ci_upper = compute_confidence_interval(diff_returns)

        results['pair_indices'].append(pair_idx)
        results['state1_obs'].append(obs1)
        results['state2_obs'].append(obs2)
        results['s1_returns'].append(s1_returns)
        results['s2_returns'].append(s2_returns)
        results['s1_mean'].append(s1_mean)
        results['s1_std'].append(s1_std)
        results['s1_ci_lower'].append(s1_ci_lower)
        results['s1_ci_upper'].append(s1_ci_upper)
        results['s2_mean'].append(s2_mean)
        results['s2_std'].append(s2_std)
        results['s2_ci_lower'].append(s2_ci_lower)
        results['s2_ci_upper'].append(s2_ci_upper)
        results['diff_mean'].append(diff_mean)
        results['diff_std'].append(diff_std)
        results['diff_ci_lower'].append(diff_ci_lower)
        results['diff_ci_upper'].append(diff_ci_upper)

    for key in results:
        if key in ['s1_returns', 's2_returns']:
            # Returns are variable-length trajectories, need object dtype
            results[key] = np.array(results[key], dtype=object)
        else:
            # Everything else (including state1_obs, state2_obs) can be regular arrays
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

    env.close()


def generate_data(config: ExperimentConfig, policy_path: Path, output_dir: Path, start_batch_idx: int = 0, end_batch_idx: int = None, generate_paired: bool = False):
    """Generate n batches of k episodes using trained policy.

    Args:
        config: Experiment configuration
        policy_path: Path to trained policy (.zip file)
        output_dir: Directory to save generated data
        start_batch_idx: Skip batches before this index (for resuming interrupted runs)
        end_batch_idx: Stop after this index (exclusive, None = generate all batches)
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
    print(f"Environment: {config.environment.name}")
    print(f"VecNormalize: {use_vec_normalize}")
    print(f"Deterministic policy: {config.data_generation.deterministic_policy}")
    print(f"Output directory: {output_dir}\n")

    all_batch_stats = []
    current_seed = config.seed


    batches_to_generate = []
    if config.data_generation.tuning_episodes > 0:
        batches_to_generate.append(("batch_tuning", config.data_generation.tuning_episodes))
        if config.data_generation.validation_episodes_per_batch > 0:
            batches_to_generate.append(("batch_tuning_validation", config.data_generation.validation_episodes_per_batch))

    # Regular batches (each with optional validation set)
    for i in range(config.data_generation.n_batches):
        if i < start_batch_idx or (end_batch_idx is not None and i >= end_batch_idx):
            batches_to_generate.append((f"skip", 0))
            if config.data_generation.validation_episodes_per_batch > 0:
                batches_to_generate.append((f"skip", 0))
            continue
        batches_to_generate.append((f"batch_{i}", config.data_generation.episodes_per_batch))
        if config.data_generation.validation_episodes_per_batch > 0:
            batches_to_generate.append((f"batch_{i}_validation", config.data_generation.validation_episodes_per_batch))

    # Eval batch
    if config.data_generation.eval_episodes > 0:
        batches_to_generate.append(("batch_eval", config.data_generation.eval_episodes))

    # Generate all batches
    for batch_name, n_episodes in batches_to_generate:
        print(f"Collecting {batch_name} ({n_episodes} episodes)")
        if batch_name == "skip":
            print(f"  Skipping batch {batch_name}\n")
            current_seed += 1
            continue
        batch_data = collect_batch(
            env=env,
            model=model,
            n_episodes=n_episodes,
            deterministic=config.data_generation.deterministic_policy,
            batch_seed=current_seed,
            use_vec_normalize=use_vec_normalize
        )

        stats = save_and_log_batch(batch_data, output_dir / f"{batch_name}.npz", batch_name)
        stats['batch_seed'] = current_seed
        all_batch_stats.append(stats)

        current_seed += 1

    # Save overall statistics
    total_episodes = sum(n_eps for _, n_eps in batches_to_generate)

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
        'eval_episodes': config.data_generation.eval_episodes,
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
        generate_paired_states(config, model, output_dir, gamma=config.value_estimators.training.gamma, vec_normalize_path=vec_normalize_path)


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
        generate_paired = config.paired_state.enabled

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
                 generate_paired=generate_paired)


if __name__ == "__main__":
    main()
