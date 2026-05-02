"""Estimator-vs-ground-truth evaluation entrypoints.

One module per evaluation mode (value, trajectory, advantage, gradient).
Each is invoked as `uv run -m src.evaluators.<mode> --config ... --method ... ...`
and writes a single parquet under `{method}_estimator_NNN/eval_NNN/results/<mode>/`.
"""
