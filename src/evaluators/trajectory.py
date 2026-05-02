"""Trajectory-mode evaluation: V_est at every step of saved trajectories + MC return."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.evaluators._common import load_v_estimator, mc_returns, results_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--batch-idx", type=int, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    gamma = config.value_estimators.training.gamma
    n_save = config.evaluation.trajectories.n_save

    traj_path = config.get_eval_data_dir() / "trajectories.npz"
    batch = np.load(traj_path, allow_pickle=True)
    obs_list = list(batch['observations'])[:n_save]
    rew_list = list(batch['rewards'])[:n_save]

    estimator = load_v_estimator(config, args.method, args.n_episodes, args.batch_idx)

    rows = []
    for ep_idx, (obs, rew) in enumerate(zip(obs_list, rew_list)):
        obs_f = obs.astype(np.float32)
        v_pred = np.asarray(estimator.predict(obs_f)).reshape(-1)
        g = mc_returns(rew, gamma)
        for t in range(len(obs)):
            rows.append({
                'episode_idx': ep_idx,
                't': t,
                'batch_name': str(args.batch_idx),
                'v_pred': float(v_pred[t]),
                'mc_return': float(g[t]),
            })

    df = pd.DataFrame(rows)
    out_dir = results_root(config, args.method) / "trajectory" / str(args.n_episodes) / f"batch_{args.batch_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
