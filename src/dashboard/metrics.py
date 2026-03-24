"""Metric functions and registry.

Each metric is a pure function operating on numpy arrays from MethodPredictions.
Metrics declare which ground truth types they accept and whether they need a baseline.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Type

from src.dashboard.data import MethodPredictions, GroundTruthStore, TrajectoryEval, TemporalDiffEval, PairedEval


@dataclass
class Metric:
    name: str
    description: str
    fn: Callable
    accepts: List[Type] = field(default_factory=list)  # empty = no ground truth needed
    comparison: bool = False
    plot_type: str = "histogram"  # histogram, bar, scatter_timestep
    reference_line: Optional[float] = None
    reference_label: Optional[str] = None


# --- Pure metric functions ---
# Signature: fn(mp, gt_store, baseline=None, **params) -> np.ndarray

def variance(mp, gt_store, **kw):
    return mp.variance

def log_variance(mp, gt_store, epsilon=1e-2, **kw):
    return np.log(mp.variance + epsilon)

def mean_value(mp, gt_store, **kw):
    return mp.mean

def bias(mp, gt_store, **kw):
    gt = gt_store.get(mp.ground_truth_key)
    return mp.mean - gt.returns

def bias_squared(mp, gt_store, **kw):
    gt = gt_store.get(mp.ground_truth_key)
    return (mp.mean - gt.returns) ** 2

def mse(mp, gt_store, **kw):
    gt = gt_store.get(mp.ground_truth_key)
    return mp.variance + (mp.mean - gt.returns) ** 2

def log_variance_ratio(mp, gt_store, baseline=None, epsilon=1e-2, **kw):
    return np.log(mp.variance + epsilon) - np.log(baseline.variance + epsilon)

def log_mean_ratio(mp, gt_store, baseline=None, epsilon=1e-2, **kw):
    return np.log(np.abs(mp.mean) + epsilon) - np.log(np.abs(baseline.mean) + epsilon)

def variance_by_decile(mp, gt_store, n_buckets=10, **kw):
    """Mean variance per value decile. Returns DataFrame with bucket and metric_value."""
    buckets = pd.qcut(mp.mean, n_buckets, labels=False, duplicates="drop")
    df = pd.DataFrame({"bucket": buckets, "variance": mp.variance})
    return df.groupby("bucket")["variance"].mean().reset_index().rename(
        columns={"variance": "metric_value"}
    )

def normalized_variance_by_decile(mp, gt_store, n_buckets=10, **kw):
    """Mean(variance / mean^2) per value decile."""
    safe_mean_sq = mp.mean ** 2
    safe_mean_sq[safe_mean_sq < 1e-10] = 1e-10
    norm_var = mp.variance / safe_mean_sq
    buckets = pd.qcut(mp.mean, n_buckets, labels=False, duplicates="drop")
    df = pd.DataFrame({"bucket": buckets, "norm_variance": norm_var})
    return df.groupby("bucket")["norm_variance"].mean().reset_index().rename(
        columns={"norm_variance": "metric_value"}
    )


# --- Registry ---

METRICS = {
    "variance": Metric(
        "Variance", "Var[V̂(s)] across batches", variance,
    ),
    "log_variance": Metric(
        "Log Variance", "log(Var + ε)", log_variance,
    ),
    "mean": Metric(
        "Mean Value", "E[V̂(s)] across batches", mean_value,
    ),
    "bias": Metric(
        "Bias", "E[V̂] - V*", bias,
        accepts=[TrajectoryEval, TemporalDiffEval, PairedEval],
    ),
    "bias_squared": Metric(
        "Bias²", "(E[V̂] - V*)²", bias_squared,
        accepts=[TrajectoryEval, TemporalDiffEval, PairedEval],
    ),
    "mse": Metric(
        "MSE", "Var + Bias² (mean squared error)", mse,
        accepts=[TrajectoryEval, TemporalDiffEval, PairedEval],
    ),
    "log_variance_ratio": Metric(
        "Log Variance Ratio", "log(Var_method / Var_baseline)",
        log_variance_ratio,
        comparison=True, reference_line=0.0, reference_label="Equal variance",
    ),
    "log_mean_ratio": Metric(
        "Log Mean Ratio", "log(|E_method| / |E_baseline|)",
        log_mean_ratio,
        comparison=True, reference_line=0.0, reference_label="Equal mean",
    ),
    "variance_by_decile": Metric(
        "Variance by Decile", "Avg variance per value bucket",
        variance_by_decile, plot_type="bar",
    ),
    "normalized_variance_by_decile": Metric(
        "Normalized Var by Decile", "Avg(Var/Mean²) per value bucket",
        normalized_variance_by_decile, plot_type="bar",
    ),
    "variance_vs_timestep": Metric(
        "Variance vs Timestep", "Var[V̂(s)] by position in episode",
        variance, accepts=[TrajectoryEval], plot_type="scatter_timestep",
    ),
}


def available_metrics(gt_store: GroundTruthStore, gt_key: Optional[str]) -> dict:
    """Filter METRICS to those whose ground truth requirements are met."""
    gt = gt_store.get(gt_key) if gt_key else None
    result = {}
    for key, metric in METRICS.items():
        if not metric.accepts:
            result[key] = metric
        elif gt is not None and type(gt) in metric.accepts:
            result[key] = metric
    return result


# --- Bootstrap ---

def bootstrap_metric_stderr(
    mp: MethodPredictions,
    gt_store: GroundTruthStore,
    metric_fn: Callable,
    baseline: Optional[MethodPredictions] = None,
    n_bootstrap: int = 200,
    seed: int = 42,
    **metric_params,
) -> float:
    """Compute standard error of a metric's mean via batch resampling.

    Resamples batch columns of the pivot table, recomputes mean/variance,
    evaluates the metric, and returns stderr of the metric's mean across resamples.
    """
    rng = np.random.RandomState(seed)
    n_batches = mp.pivot.shape[1]
    pivot_values = mp.pivot.values
    baseline_values = baseline.pivot.values if baseline is not None else None

    metric_means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_batches, n_batches, replace=True)

        resampled = pivot_values[:, idx]
        resampled_mp = MethodPredictions(
            method=mp.method,
            n_episodes=mp.n_episodes,
            pivot=pd.DataFrame(resampled),
            mean=resampled.mean(axis=1),
            variance=resampled.var(axis=1, ddof=1),
            ground_truth_key=mp.ground_truth_key,
        )

        baseline_mp = None
        if baseline_values is not None:
            bl_resampled = baseline_values[:, idx]
            baseline_mp = MethodPredictions(
                method=baseline.method,
                n_episodes=baseline.n_episodes,
                pivot=pd.DataFrame(bl_resampled),
                mean=bl_resampled.mean(axis=1),
                variance=bl_resampled.var(axis=1, ddof=1),
                ground_truth_key=baseline.ground_truth_key,
            )

        vals = metric_fn(resampled_mp, gt_store, baseline=baseline_mp, **metric_params)
        if isinstance(vals, pd.DataFrame):
            metric_means.append(vals["metric_value"].mean())
        else:
            metric_means.append(np.nanmean(vals))

    return np.std(metric_means, ddof=1)
