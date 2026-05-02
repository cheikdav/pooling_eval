"""Value-mode evaluation: V_est on `reset_states.npz` points vs ground-truth V."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.evaluators._common import load_v_estimator, results_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--batch-idx", type=int, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    rs_path = config.get_eval_data_dir() / "reset_states.npz"
    if not rs_path.exists():
        raise FileNotFoundError(f"Ground-truth eval data not found: {rs_path}")
    rs = np.load(rs_path, allow_pickle=False)
    states = rs['states'].astype(np.float32)

    estimator = load_v_estimator(config, args.method, args.n_episodes, args.batch_idx)
    v_pred = np.asarray(estimator.predict(states)).reshape(-1)

    df = pd.DataFrame({
        'state_idx': np.arange(len(states), dtype=np.int64),
        'batch_name': str(args.batch_idx),
        'v_pred': v_pred,
    })

    out_dir = results_root(config, args.method) / "value" / str(args.n_episodes) / f"batch_{args.batch_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
