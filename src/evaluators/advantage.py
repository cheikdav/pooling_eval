"""Advantage-mode evaluation: A_GAE-λ on each Q-rollout vs ground-truth A."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.evaluators._common import gae_advantage, lam_dir, load_v_estimator, results_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--batch-idx", type=int, required=True)
    parser.add_argument("--lam", type=float, required=True, help="GAE lambda")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    gamma = config.value_estimators.training.gamma

    rollouts_path = config.get_eval_data_dir() / "reset_states_rollouts.npz"
    if not rollouts_path.exists():
        raise FileNotFoundError(
            f"reset_states_rollouts.npz not found at {rollouts_path}. "
            f"Set evaluation.reset_states.n_rollouts_keep > 0 and rerun generate_eval_data."
        )
    r = np.load(rollouts_path, allow_pickle=True)
    state_idx = r['state_idx']
    rollout_idx = r['rollout_idx']
    obs_list = list(r['observations'])
    rew_list = list(r['rewards'])

    estimator = load_v_estimator(config, args.method, args.n_episodes, args.batch_idx)

    # A_pred at the *initial* (s, a) of each rollout = advs[0] from GAE-λ.
    a_preds = np.empty(len(obs_list), dtype=np.float64)
    for k, (obs, rew) in enumerate(zip(obs_list, rew_list)):
        obs_f = obs.astype(np.float32)
        values = np.asarray(estimator.predict(obs_f)).reshape(-1)
        advs = gae_advantage(rew, values, gamma, args.lam, last_value=0.0)
        a_preds[k] = advs[0]

    df = pd.DataFrame({
        'batch_name': str(args.batch_idx),
        'lambda': args.lam,
        'state_idx': state_idx.astype(np.int64),
        'rollout_idx': rollout_idx.astype(np.int64),
        'a_pred': a_preds,
    })

    out_dir = (results_root(config, args.method) / "advantage" / lam_dir(args.lam)
               / str(args.n_episodes) / f"batch_{args.batch_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
