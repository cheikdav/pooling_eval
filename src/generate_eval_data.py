"""Generate ground-truth evaluation data.

Two independent modes (each enabled if its config block is non-trivial):

1. Trajectories: N on-policy rollouts saved as `trajectories.npz`. Used for
   the trajectory-viz mode and the PPO-gradient mode (which needs many
   trajectories for a low-variance gradient estimate).

2. Reset-states: sample states along a few seed trajectories, then for each
   state do many resets to estimate V(s), Q(s, a), A(s, a) = Q - V. Output
   saved as `reset_states.npz` with mean/std/stderr per estimate.

Output dir: {data_NNN}/eval_NNN/ (resolved via the registry from eval params).
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from src.config import ExperimentConfig
from src.env_utils import ALGORITHM_MAP, create_vec_env
from src.generate_data import collect_episodes_parallel, episodes_to_batch
from src.policy_gradient import compute_mc_returns, compute_surrogate_gradient


def get_full_state(env) -> Tuple[np.ndarray, np.ndarray]:
    return env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()


def restore_full_state(env, qpos: np.ndarray, qvel: np.ndarray):
    env.unwrapped.set_state(qpos, qvel)


def get_obs(env) -> np.ndarray:
    return env.unwrapped._get_obs()


# --- Mode 1: trajectories ---------------------------------------------------

def generate_trajectories(config: ExperimentConfig, model, env,
                          use_vec_normalize: bool, output_path: Path):
    n_total = config.evaluation.trajectories.n_total
    seed = config.evaluation.seed
    print(f"\nGenerating {n_total} trajectories (seed={seed})")

    np.random.seed(seed)
    torch.manual_seed(seed)

    episodes = collect_episodes_parallel(
        env, model, n_total,
        deterministic=config.data_generation.deterministic_policy,
        use_vec_normalize=use_vec_normalize,
        seed=seed,
    )
    batch = episodes_to_batch(episodes)
    np.savez_compressed(output_path, **batch)

    print(f"  Saved {n_total} trajectories to {output_path}")
    print(f"  Mean return: {batch['episode_returns'].mean():.2f} ± {batch['episode_returns'].std():.2f}")
    print(f"  Mean length: {batch['episode_lengths'].mean():.1f}")


# --- Mode 2: reset states ---------------------------------------------------

def generate_seed_trajectory(env_name: str, max_episode_steps: int,
                             policy_path: str, algorithm: str,
                             deterministic: bool, seed: int) -> Dict[str, np.ndarray]:
    """Roll out one episode under the policy, capturing per-step (qpos, qvel, obs, action)."""
    AlgorithmClass = ALGORITHM_MAP[algorithm]
    model = AlgorithmClass.load(policy_path)
    env = gym.make(env_name, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)

    qposs, qvels, obss, actions = [], [], [], []
    obs = get_obs(env)
    done = truncated = False
    while not (done or truncated):
        qposs.append(env.unwrapped.data.qpos.copy())
        qvels.append(env.unwrapped.data.qvel.copy())
        obss.append(obs.copy())
        action, _ = model.predict(obs, deterministic=deterministic)
        actions.append(np.asarray(action).copy())
        obs, _, done, truncated, _ = env.step(action)

    env.close()
    return {
        'qpos': np.array(qposs),
        'qvel': np.array(qvels),
        'obs': np.array(obss),
        'actions': np.array(actions),
    }


def sample_states(trajectories: List[Dict[str, np.ndarray]],
                  stride: int, n_states_max: int) -> List[Dict[str, np.ndarray]]:
    """Subsample states with given stride; cap at n_states_max via uniform thinning."""
    states = []
    for traj in trajectories:
        T = len(traj['obs'])
        for t in range(0, T, stride):
            states.append({
                'qpos': traj['qpos'][t],
                'qvel': traj['qvel'][t],
                'obs': traj['obs'][t],
                'action': traj['actions'][t],
            })
    if n_states_max > 0 and len(states) > n_states_max:
        idx = np.linspace(0, len(states) - 1, n_states_max).astype(int)
        states = [states[i] for i in idx]
    return states


def _rollout(env, model, qpos, qvel, gamma, max_steps, deterministic,
             vec_normalize, initial_action=None, record=False):
    """Reset to (qpos, qvel), optionally play initial_action, then play policy until done.

    Returns the discounted return (and the per-step trajectory if record=True).
    """
    env.reset()
    restore_full_state(env, qpos, qvel)
    obs = get_obs(env)
    if vec_normalize is not None:
        obs = vec_normalize.normalize_obs(obs[None])[0]

    total = 0.0
    discount = 1.0
    step = 0
    done = truncated = False

    if record:
        obs_list, act_list, rew_list, done_list = [], [], [], []

    if initial_action is not None:
        if record:
            obs_list.append(obs.copy())
            act_list.append(np.asarray(initial_action).copy())
        obs, reward, done, truncated, _ = env.step(initial_action)
        if vec_normalize is not None:
            obs = vec_normalize.normalize_obs(obs[None])[0]
        total += reward
        if record:
            rew_list.append(float(reward))
            done_list.append(bool(done or truncated))
        discount *= gamma
        step = 1

    while not (done or truncated) and step < max_steps:
        if record:
            obs_list.append(obs.copy())
        action, _ = model.predict(obs, deterministic=deterministic)
        if record:
            act_list.append(np.asarray(action).copy())
        obs, reward, done, truncated, _ = env.step(action)
        if vec_normalize is not None:
            obs = vec_normalize.normalize_obs(obs[None])[0]
        total += discount * reward
        if record:
            rew_list.append(float(reward))
            done_list.append(bool(done or truncated))
        discount *= gamma
        step += 1

    if record:
        return total, {
            'observations': np.array(obs_list),
            'actions': np.array(act_list),
            'rewards': np.array(rew_list),
            'dones': np.array(done_list),
        }
    return total


def _process_state(args):
    """Worker: V and Q rollouts for one state. Records first n_rollouts_keep Q-rollouts."""
    (state_idx, qpos, qvel, action, env_name, policy_path, algorithm,
     max_episode_steps, n_rollouts, n_rollouts_keep, gamma, deterministic,
     vec_normalize_path) = args

    AlgorithmClass = ALGORITHM_MAP[algorithm]
    model = AlgorithmClass.load(policy_path)
    env = gym.make(env_name, max_episode_steps=max_episode_steps)

    vec_normalize = None
    if vec_normalize_path is not None and Path(vec_normalize_path).exists():
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        dummy = DummyVecEnv([lambda: gym.make(env_name, max_episode_steps=max_episode_steps)])
        vec_normalize = VecNormalize.load(str(vec_normalize_path), dummy)
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    max_steps = int(10 / (1 - gamma))

    v_returns = np.array([
        _rollout(env, model, qpos, qvel, gamma, max_steps, deterministic, vec_normalize)
        for _ in range(n_rollouts)
    ])

    q_returns = []
    kept_trajs = []
    for k in range(n_rollouts):
        if k < n_rollouts_keep:
            ret, traj = _rollout(env, model, qpos, qvel, gamma, max_steps, deterministic,
                                 vec_normalize, initial_action=action, record=True)
            kept_trajs.append(traj)
        else:
            ret = _rollout(env, model, qpos, qvel, gamma, max_steps, deterministic,
                           vec_normalize, initial_action=action)
        q_returns.append(ret)
    q_returns = np.array(q_returns)

    env.close()
    return state_idx, v_returns, q_returns, kept_trajs


def generate_reset_states(config: ExperimentConfig, policy_path: Path,
                          output_path: Path, rollouts_output_path: Path,
                          n_workers: int = None):
    rs = config.evaluation.reset_states
    gamma = config.value_estimators.training.gamma
    env_params = config.get_data_env_params()
    n_keep = min(rs.n_rollouts_keep, rs.n_rollouts)

    stride = rs.state_stride
    if stride == "auto":
        stride = max(1, int(0.1 / (1 - gamma)))
    stride = int(stride)

    print(f"\nReset-states mode: stride={stride}, n_seed_trajectories={rs.n_seed_trajectories}, "
          f"n_states_max={rs.n_states_max}, n_rollouts={rs.n_rollouts}, n_rollouts_keep={n_keep}")

    print("Generating seed trajectories...")
    seed_trajs = []
    for i in range(rs.n_seed_trajectories):
        traj = generate_seed_trajectory(
            config.environment.name, env_params['max_episode_steps'],
            str(policy_path), config.policy.algorithm,
            deterministic=config.data_generation.deterministic_policy,
            seed=config.evaluation.seed + 1 + i,
        )
        seed_trajs.append(traj)
        print(f"  Trajectory {i}: T={len(traj['obs'])}")

    states = sample_states(seed_trajs, stride, rs.n_states_max)
    print(f"Sampled {len(states)} states from seed trajectories")

    vec_normalize_path = (
        policy_path.parent / "vec_normalize.pkl"
        if config.policy.use_vec_normalize else None
    )
    args_list = [
        (i, s['qpos'], s['qvel'], s['action'],
         config.environment.name, str(policy_path), config.policy.algorithm,
         env_params['max_episode_steps'], rs.n_rollouts, n_keep, gamma,
         config.data_generation.deterministic_policy,
         str(vec_normalize_path) if vec_normalize_path else None)
        for i, s in enumerate(states)
    ]

    print(f"Computing V/Q ({rs.n_rollouts} rollouts each) for {len(states)} states "
          f"in parallel (n_workers={n_workers or 'all CPUs'})...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_process_state, args_list),
                            total=len(args_list), desc="States"))

    results.sort(key=lambda r: r[0])
    n_states = len(results)
    n = rs.n_rollouts

    out = {
        'states': np.stack([s['obs'] for s in states]),
        'qpos': np.stack([s['qpos'] for s in states]),
        'qvel': np.stack([s['qvel'] for s in states]),
        'actions': np.stack([np.atleast_1d(s['action']) for s in states]),
        'v_mean': np.zeros(n_states),
        'v_std': np.zeros(n_states),
        'v_stderr': np.zeros(n_states),
        'q_mean': np.zeros(n_states),
        'q_std': np.zeros(n_states),
        'q_stderr': np.zeros(n_states),
        'advantage_mean': np.zeros(n_states),
        'advantage_stderr': np.zeros(n_states),
    }

    rollouts_obs, rollouts_act, rollouts_rew, rollouts_done = [], [], [], []
    rollouts_state_idx, rollouts_idx = [], []

    for state_idx, v, q, kept_trajs in results:
        out['v_mean'][state_idx] = v.mean()
        out['v_std'][state_idx] = v.std(ddof=1)
        out['v_stderr'][state_idx] = out['v_std'][state_idx] / np.sqrt(n)
        out['q_mean'][state_idx] = q.mean()
        out['q_std'][state_idx] = q.std(ddof=1)
        out['q_stderr'][state_idx] = out['q_std'][state_idx] / np.sqrt(n)
        out['advantage_mean'][state_idx] = out['q_mean'][state_idx] - out['v_mean'][state_idx]
        out['advantage_stderr'][state_idx] = np.sqrt(
            out['v_stderr'][state_idx] ** 2 + out['q_stderr'][state_idx] ** 2
        )
        for k, traj in enumerate(kept_trajs):
            rollouts_obs.append(traj['observations'])
            rollouts_act.append(traj['actions'])
            rollouts_rew.append(traj['rewards'])
            rollouts_done.append(traj['dones'])
            rollouts_state_idx.append(state_idx)
            rollouts_idx.append(k)

    np.savez_compressed(output_path, **out)
    print(f"Saved reset-states ground truth to {output_path}")
    print(f"  V: mean={out['v_mean'].mean():.3f}, mean_stderr={out['v_stderr'].mean():.3f}")
    print(f"  Q: mean={out['q_mean'].mean():.3f}, mean_stderr={out['q_stderr'].mean():.3f}")
    print(f"  A: mean={out['advantage_mean'].mean():.3f}, mean_stderr={out['advantage_stderr'].mean():.3f}")

    if n_keep > 0:
        rollouts_out = {
            'state_idx': np.array(rollouts_state_idx),
            'rollout_idx': np.array(rollouts_idx),
            'observations': np.array(rollouts_obs, dtype=object),
            'actions': np.array(rollouts_act, dtype=object),
            'rewards': np.array(rollouts_rew, dtype=object),
            'dones': np.array(rollouts_done, dtype=object),
        }
        np.savez_compressed(rollouts_output_path, **rollouts_out)
        print(f"Saved {len(rollouts_obs)} Q-rollout trajectories ({n_keep} per state) "
              f"to {rollouts_output_path}")


# --- Reference gradient (eval-data level) -----------------------------------

def generate_reference_gradient(config: ExperimentConfig, policy_path: Path,
                                trajectories_path: Path, output_path: Path):
    """Compute g_true_ref = ∇θ E[A · log π_θ] with unbiased A on all trajectories.

    A_t = G_t − b(s_t) where b is a state-only baseline (any baseline preserves
    unbiasedness; we use the configured choice for variance reduction only).
    """
    gamma = config.value_estimators.training.gamma
    baseline_type = config.evaluation.gradient_reference.baseline

    print(f"\nComputing reference gradient (baseline={baseline_type}, gamma={gamma})...")
    AlgorithmClass = ALGORITHM_MAP[config.policy.algorithm]
    model = AlgorithmClass.load(policy_path, device='cpu')
    for p in model.policy.parameters():
        p.requires_grad_(True)

    batch = np.load(trajectories_path, allow_pickle=True)
    obs_list = list(batch['observations'])
    act_list = list(batch['actions'])
    rew_list = list(batch['rewards'])

    G_per_traj = compute_mc_returns(rew_list, gamma)
    G = np.concatenate(G_per_traj, axis=0)
    obs_flat = np.concatenate(obs_list, axis=0).astype(np.float32)
    act_flat = np.concatenate(act_list, axis=0).astype(np.float32)

    if baseline_type == "batch_mean":
        b = float(G.mean())
    elif baseline_type == "zero":
        b = 0.0
    else:
        raise ValueError(f"Unknown baseline: {baseline_type}")
    A = (G - b).astype(np.float32)

    grad = compute_surrogate_gradient(model, obs_flat, act_flat, A)

    np.savez_compressed(output_path,
                        grad=grad,
                        baseline=baseline_type,
                        baseline_value=np.float64(b),
                        n_trajectories=np.int64(len(obs_list)),
                        n_transitions=np.int64(len(obs_flat)),
                        param_count=np.int64(len(grad)))
    print(f"Saved reference gradient ({len(grad)} params, baseline_value={b:.3f}) to {output_path}")


# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate ground-truth evaluation data")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override eval data dir (default: registry-resolved {data_NNN}/eval_NNN/)")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Workers for reset-state rollouts (default: all CPUs)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "trajectories", "reset_states", "reference_gradient"])
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    policy_path = args.policy_path or config.get_policy_dir() / "policy_final.zip"
    output_dir = args.output_dir or config.get_eval_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Eval data output dir: {output_dir}")

    do_traj = args.mode in ("all", "trajectories") and config.evaluation.trajectories.n_total > 0
    do_reset = args.mode in ("all", "reset_states") and config.evaluation.reset_states.n_seed_trajectories > 0
    do_grad = args.mode in ("all", "reference_gradient") and config.evaluation.trajectories.n_total > 0

    if do_traj:
        AlgorithmClass = ALGORITHM_MAP[config.policy.algorithm]
        model = AlgorithmClass.load(policy_path)
        vec_normalize_path = (
            policy_path.parent / "vec_normalize.pkl"
            if config.policy.use_vec_normalize else None
        )
        env_params = config.get_data_env_params()
        env, use_vec_normalize = create_vec_env(
            config, n_envs=config.data_generation.n_envs,
            use_monitor=False, vec_normalize_path=vec_normalize_path,
            seed=config.evaluation.seed, **env_params,
        )
        generate_trajectories(config, model, env, use_vec_normalize,
                              output_dir / "trajectories.npz")
        env.close()

    if do_reset:
        generate_reset_states(config, policy_path,
                              output_dir / "reset_states.npz",
                              output_dir / "reset_states_rollouts.npz",
                              args.n_workers)

    if do_grad:
        generate_reference_gradient(config, policy_path,
                                    output_dir / "trajectories.npz",
                                    output_dir / "reference_gradient.npz")

    metadata = {
        'experiment_id': config.experiment_id,
        'gamma': config.value_estimators.training.gamma,
        'evaluation': {
            'trajectories': {
                'n_total': config.evaluation.trajectories.n_total,
                'n_save': config.evaluation.trajectories.n_save,
            },
            'reset_states': {
                'n_seed_trajectories': config.evaluation.reset_states.n_seed_trajectories,
                'state_stride': config.evaluation.reset_states.state_stride,
                'n_states_max': config.evaluation.reset_states.n_states_max,
                'n_rollouts': config.evaluation.reset_states.n_rollouts,
                'n_rollouts_keep': config.evaluation.reset_states.n_rollouts_keep,
            },
            'gradient_reference': {
                'baseline': config.evaluation.gradient_reference.baseline,
            },
            'seed': config.evaluation.seed,
        },
    }
    with open(output_dir / "eval_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDone. Metadata at {output_dir / 'eval_metadata.json'}")


if __name__ == "__main__":
    main()
