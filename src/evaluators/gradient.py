"""Gradient-mode evaluation.

For one (V_est, λ):
- Split N trajectories into K disjoint chunks.
- For each chunk, walk through trajectories accumulating data; at each
  transition checkpoint, compute the PPO surrogate gradient using GAE-λ on V_est.
- Compute g_ref_est = surrogate gradient on all N trajectories with the same V_est + λ.
- Compute cos(g_chunk_t, g_ref_est) for each (chunk, checkpoint) — sampling variance.
- Compute cos(g_ref_est, g_true_ref) once — estimator bias (vs unbiased reference).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.env_utils import ALGORITHM_MAP
from src.evaluators._common import gae_advantage, lam_dir, load_v_estimator, results_root
from src.policy_gradient import compute_surrogate_gradient, cosine_sim


def chunked(traj_list, K):
    """Round-robin K-way split: chunk k = trajectories [k, k+K, k+2K, ...]."""
    return [traj_list[k::K] for k in range(K)]


def precompute_advantages(traj_chunk, estimator, gamma, lam):
    """For each trajectory in chunk: compute V_est on obs, then GAE-λ. Returns list of (obs, act, adv)."""
    out = []
    for traj in traj_chunk:
        obs = traj['observations'].astype(np.float32)
        act = traj['actions'].astype(np.float32)
        rew = traj['rewards']
        values = np.asarray(estimator.predict(obs)).reshape(-1)
        advs = gae_advantage(rew, values, gamma, lam, last_value=0.0).astype(np.float32)
        out.append((obs, act, advs))
    return out


def gradient_at_checkpoints(model, advantage_data, checkpoints):
    """For each checkpoint t_max, compute surrogate gradient on prefix containing ≥ t_max transitions.

    Returns a list of (checkpoint, n_transitions_used, grad).
    """
    cumul = 0
    obs_acc, act_acc, adv_acc = [], [], []
    grads = []
    cp_idx = 0
    sorted_cps = sorted(checkpoints)
    for obs, act, adv in advantage_data:
        if cp_idx >= len(sorted_cps):
            break
        obs_acc.append(obs)
        act_acc.append(act)
        adv_acc.append(adv)
        cumul += len(adv)
        while cp_idx < len(sorted_cps) and cumul >= sorted_cps[cp_idx]:
            obs_f = np.concatenate(obs_acc, axis=0)
            act_f = np.concatenate(act_acc, axis=0)
            adv_f = np.concatenate(adv_acc, axis=0)
            grad = compute_surrogate_gradient(model, obs_f, act_f, adv_f)
            grads.append((int(sorted_cps[cp_idx]), int(cumul), grad))
            cp_idx += 1
    return grads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--batch-idx", type=int, required=True)
    parser.add_argument("--lam", type=float, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    gamma = config.value_estimators.training.gamma
    grad_cfg = config.evaluation.modes.gradient
    K = grad_cfg.n_chunks
    checkpoints = sorted(grad_cfg.transition_checkpoints)

    eval_data_dir = config.get_eval_data_dir()
    traj_path = eval_data_dir / "trajectories.npz"
    ref_path = eval_data_dir / "reference_gradient.npz"

    batch = np.load(traj_path, allow_pickle=True)
    n_traj = len(batch['observations'])
    traj_list = [
        {
            'observations': batch['observations'][i],
            'actions': batch['actions'][i],
            'rewards': batch['rewards'][i],
        }
        for i in range(n_traj)
    ]

    estimator = load_v_estimator(config, args.method, args.n_episodes, args.batch_idx)
    AlgorithmClass = ALGORITHM_MAP[config.policy.algorithm]
    policy_path = config.get_policy_dir() / "policy_final.zip"
    model = AlgorithmClass.load(policy_path, device='cpu')
    for p in model.policy.parameters():
        p.requires_grad_(True)

    chunks = chunked(traj_list, K)
    rows = []
    all_obs, all_act, all_adv = [], [], []

    for k, chunk in enumerate(chunks):
        adv_data = precompute_advantages(chunk, estimator, gamma, args.lam)
        chunk_grads = gradient_at_checkpoints(model, adv_data, checkpoints)
        for cp, ntrans, g in chunk_grads:
            rows.append({
                'batch_name': str(args.batch_idx),
                'lambda': args.lam,
                'chunk_idx': str(k),
                'n_transitions_checkpoint': cp,
                'n_transitions_actual': ntrans,
                'cos_sim': None,   # filled in below once g_ref_est is known
                'grad_norm': float(np.linalg.norm(g)),
            })
            rows[-1]['_grad'] = g  # stash for cos_sim computation

        for obs, act, adv in adv_data:
            all_obs.append(obs)
            all_act.append(act)
            all_adv.append(adv)

    obs_full = np.concatenate(all_obs, axis=0)
    act_full = np.concatenate(all_act, axis=0)
    adv_full = np.concatenate(all_adv, axis=0)
    g_ref_est = compute_surrogate_gradient(model, obs_full, act_full, adv_full)

    for row in rows:
        row['cos_sim'] = cosine_sim(row.pop('_grad'), g_ref_est)

    grt = np.load(ref_path, allow_pickle=False)
    g_true_ref = grt['grad']
    rows.append({
        'batch_name': str(args.batch_idx),
        'lambda': args.lam,
        'chunk_idx': 'ref',
        'n_transitions_checkpoint': int(len(adv_full)),
        'n_transitions_actual': int(len(adv_full)),
        'cos_sim': cosine_sim(g_ref_est, g_true_ref),
        'grad_norm': float(np.linalg.norm(g_ref_est)),
    })

    df = pd.DataFrame(rows)
    out_dir = (results_root(config, args.method) / "gradient" / lam_dir(args.lam)
               / str(args.n_episodes) / f"batch_{args.batch_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
