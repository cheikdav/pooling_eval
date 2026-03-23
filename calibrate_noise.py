"""Sweep over (reset_noise_scale, action_noise_std) to calibrate noise levels.

Metrics per config:
  Episode-level (undiscounted returns):
    - IQR: robust diversity measure
    - IQR/median: scale-free robust diversity
    - Skewness: asymmetry (negative = heavy left tail / bad outliers)
    - Excess kurtosis: tail weight (0 = normal-like)
    - Outlier fraction: fraction outside Tukey fences

  State-level (discounted returns at every timestep):
    - cross_state_std: std of G_t across all states and episodes
    - cross_state_iqr: IQR of G_t across all states and episodes
    - intra_traj_std: mean per-episode std of G_t (how much value varies within a trajectory)
"""

import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import replace

from src.config import ExperimentConfig
from src.env_utils import ALGORITHM_MAP, create_vec_env
from src.generate_data import collect_episodes_parallel


def collect_episodes(config, policy, reset_noise_scale, action_noise_std, n_episodes, n_envs, use_vec_normalize):
    """Collect full episodes using collect_episodes_parallel (unbiased)."""
    env, _ = create_vec_env(
        config, n_envs=n_envs, use_monitor=False, seed=config.data_generation.seed,
        max_episode_steps=config.environment.max_episode_steps,
        reset_noise_scale=reset_noise_scale,
        action_noise_std=action_noise_std if action_noise_std > 0 else None,
    )
    episodes = collect_episodes_parallel(env, policy, n_episodes,
                                         deterministic=False,
                                         use_vec_normalize=use_vec_normalize)
    env.close()
    return episodes


def compute_discounted_returns(rewards, gamma):
    """Compute discounted return G_t at every timestep (reverse cumsum)."""
    T = len(rewards)
    G = np.zeros(T)
    G[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    return G


def compute_metrics(episodes, gamma):
    # Episode-level: undiscounted returns
    returns = np.array([ep['rewards'].sum() for ep in episodes])
    q1, q3 = np.percentile(returns, [25, 75])
    iqr = q3 - q1
    median = np.median(returns)
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_frac = np.mean((returns < lower_fence) | (returns > upper_fence))

    # State-level: discounted returns at every timestep
    all_G = []
    intra_stds = []
    for ep in episodes:
        G = compute_discounted_returns(ep['rewards'], gamma)
        all_G.append(G)
        intra_stds.append(G.std())

    all_G_flat = np.concatenate(all_G)
    gq1, gq3 = np.percentile(all_G_flat, [25, 75])

    return {
        'mean': returns.mean(),
        'std': returns.std(),
        'iqr': iqr,
        'iqr_over_median': iqr / abs(median) if abs(median) > 1e-8 else float('inf'),
        'skewness': float(stats.skew(returns)),
        'excess_kurtosis': float(stats.kurtosis(returns)),
        'outlier_frac': outlier_frac,
        'p5': np.percentile(returns, 5),
        'p95': np.percentile(returns, 95),
        'cross_state_std': all_G_flat.std(),
        'cross_state_iqr': gq3 - gq1,
        'intra_traj_std': np.mean(intra_stds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--n-episodes', type=int, default=200)
    parser.add_argument('--n-envs', type=int, default=32)
    parser.add_argument('--reset-noise-scales', type=float, nargs='+', default=[0.01, 0.1, 0.2, 0.4, 0.6])
    parser.add_argument('--action-noise-stds', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument('--output', type=str, default='noise_calibration.png')
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    gamma = config.value_estimators.training.gamma
    policy_path = config.get_policy_dir() / 'policy_final.zip'
    algo_cls = ALGORITHM_MAP[config.policy.algorithm]
    policy = algo_cls.load(policy_path)
    use_vec_normalize = config.policy.use_vec_normalize
    print(f"Loaded policy from {policy_path}")

    configs = list(itertools.product(args.reset_noise_scales, args.action_noise_stds))
    print(f"Running {len(configs)} configs × {args.n_episodes} episodes each...\n")

    rows = []
    all_returns = {}
    for reset_noise, action_noise in configs:
        label = f"rn={reset_noise}, an={action_noise}"
        print(f"  {label} ...", end=' ', flush=True)
        episodes = collect_episodes(
            config, policy,
            reset_noise_scale=reset_noise,
            action_noise_std=action_noise,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            use_vec_normalize=use_vec_normalize,
        )
        metrics = compute_metrics(episodes, gamma)
        print(f"mean={metrics['mean']:.0f}, IQR={metrics['iqr']:.0f}, IQR/med={metrics['iqr_over_median']:.3f}, "
              f"skew={metrics['skewness']:.2f}, kurt={metrics['excess_kurtosis']:.2f}, "
              f"outliers={metrics['outlier_frac']:.2%}, intra_std={metrics['intra_traj_std']:.0f}")
        rows.append({'reset_noise': reset_noise, 'action_noise': action_noise, **metrics})
        all_returns[(reset_noise, action_noise)] = np.array([ep['rewards'].sum() for ep in episodes])

    df = pd.DataFrame(rows)

    # --- Plotting ---
    rn_vals = args.reset_noise_scales
    an_vals = args.action_noise_stds
    metric_names = ['iqr', 'iqr_over_median', 'skewness', 'excess_kurtosis', 'outlier_frac', 'cross_state_std', 'cross_state_iqr', 'intra_traj_std']
    metric_labels = ['IQR (diversity ↑)', 'IQR/median (scale-free ↑)', 'Skewness (→ 0)',
                     'Excess Kurtosis (→ 0)', 'Outlier Fraction (↓)',
                     'Cross-state std (↑)', 'Cross-state IQR (↑)', 'Intra-traj std (↑)']

    n_rows = len(metric_names) + 1  # metrics heatmaps + return histograms
    n_cols = max(len(rn_vals), len(an_vals))
    fig = plt.figure(figsize=(4 * len(an_vals), 4 * n_rows))
    fig.suptitle(f"Noise calibration — {config.environment.name}", fontsize=14, y=1.01)

    # Heatmaps: rows = reset_noise, cols = action_noise
    for mi, (metric, mlabel) in enumerate(zip(metric_names, metric_labels)):
        ax = fig.add_subplot(n_rows, 1, mi + 1)
        matrix = df.pivot(index='reset_noise', columns='action_noise', values=metric).values
        higher_is_better = metric in ('iqr', 'iqr_over_median', 'cross_state_iqr', 'intra_traj_std')
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn' if higher_is_better else 'RdYlGn_r')
        ax.set_xticks(range(len(an_vals)))
        ax.set_xticklabels([f"{v}" for v in an_vals])
        ax.set_yticks(range(len(rn_vals)))
        ax.set_yticklabels([f"{v}" for v in rn_vals])
        ax.set_xlabel('action_noise_std')
        ax.set_ylabel('reset_noise_scale')
        ax.set_title(mlabel)
        plt.colorbar(im, ax=ax)
        for i in range(len(rn_vals)):
            for j in range(len(an_vals)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', fontsize=8)

    # Return histograms: one subplot per action_noise, overlaying all reset_noise values
    ax_hist = fig.add_subplot(n_rows, 1, n_rows)
    for (rn, an), returns in all_returns.items():
        ax_hist.hist(returns, bins=30, alpha=0.3, label=f"rn={rn},an={an}", density=True)
    ax_hist.set_xlabel('Undiscounted return')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Return distributions (all configs)')
    ax_hist.legend(fontsize=6, ncol=3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=120, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    # Print summary table
    print("\nSummary table:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == '__main__':
    main()
