"""Snakemake workflow for the pooling-eval experiment pipeline.

Replaces the previous src/run_pipeline.py + src/run_all_estimators.py duo.
Each logical unit of work is a separate rule, and Snakemake handles
dependency resolution, state tracking, retry, and cluster dispatch.

Invocation:

    # Local, 8 parallel tasks
    snakemake -j 8 --config experiment_config=configs/test_mini/config.yaml

    # SGE cluster via the cluster-generic executor
    snakemake \\
        --executor cluster-generic \\
        --cluster-generic-submit-cmd \\
            "qsub -V -cwd -pe threaded {threads} -l h_vmem={resources.mem_mb}M" \\
        --jobs 50 \\
        --config experiment_config=configs/hopper/config.yaml

    # Dry run — prints the DAG without submitting anything
    snakemake -n --config experiment_config=configs/hopper/config.yaml

    # DAG visualization
    snakemake --dag --config experiment_config=configs/hopper/config.yaml \\
        | dot -Tpng > dag.png

The workflow is parameterised by one config value: `experiment_config` is the
path to a pooling-eval experiment YAML (e.g., `configs/hopper/config.yaml`).
All result directories are resolved once at preamble time via `src.registry`
(parameter-based content-addressed numbering).
"""

import hashlib
import json
import re
from pathlib import Path

from src.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Preamble: load experiment config, pre-resolve all content-addressed paths.
# ---------------------------------------------------------------------------

if "experiment_config" not in config:
    raise WorkflowError(
        "Missing --config experiment_config=<path>. "
        "Invoke snakemake with e.g. --config experiment_config=configs/hopper/config.yaml"
    )

CONFIG_PATH = Path(config["experiment_config"]).resolve()
if not CONFIG_PATH.exists():
    raise WorkflowError(f"Experiment config not found: {CONFIG_PATH}")

EXP = ExperimentConfig.from_yaml(CONFIG_PATH)

METHODS = {mc.name: mc for mc in EXP.value_estimators.method_configs}
METHOD_NAMES = list(METHODS.keys())
EPISODE_SUBSETS = sorted(EXP.value_estimators.training.episode_subsets or [])
N_BATCHES = EXP.data_generation.n_batches

# Pre-resolve content-addressed result directories once. After this, rules
# use concrete absolute paths and only fan out over {method}, {n_ep}, {batch}.
POLICY_DIR = Path(EXP.get_policy_dir()).resolve()
DATA_DIR = Path(EXP.get_data_dir()).resolve()
EVAL_DATA_DIR = Path(EXP.get_eval_data_dir()).resolve()
EST_DIRS = {name: Path(EXP.get_estimator_dir(mc)).resolve() for name, mc in METHODS.items()}
EVAL_DIRS = {name: Path(EXP.get_eval_dir(mc)).resolve() for name, mc in METHODS.items()}

# Reverse lookup: est_dir (as string) -> method name. Used by rules whose
# wildcards match one of the known estimator directory paths.
METHOD_BY_EST_DIR = {str(d): name for name, d in EST_DIRS.items()}
EST_DIR_ALTERNATION = "|".join(re.escape(str(d)) for d in EST_DIRS.values())

METHOD_BY_EVAL_DIR = {str(d): name for name, d in EVAL_DIRS.items()}
EVAL_DIR_ALTERNATION = "|".join(re.escape(str(d)) for d in EVAL_DIRS.values())

ADV_LAMBDAS = list(EXP.evaluation.modes.advantage.lambdas)
GRAD_LAMBDAS = list(EXP.evaluation.modes.gradient.lambdas)

# Run identifier for application log paths. Same parameters -> same hash ->
# retries of the same cell accumulate in one log directory. Different
# parameters (same experiment_id) -> different hash -> different directories.
PARAMS_HASH = hashlib.md5(
    json.dumps(EXP.to_dict(), sort_keys=True, default=str).encode()
).hexdigest()[:8]

LOGS_ROOT = Path(EXP.get_logs_dir()).resolve() / EXP.experiment_id / PARAMS_HASH


def needs_sweep(method: str) -> bool:
    """True if the method config opts into the hyperparameter sweep path."""
    return METHODS[method].tuning is not None


def tuned_hp_path(method: str) -> str:
    return str(EST_DIRS[method] / "sweeps" / "tuned_hyperparams.json")


def estimator_ckpt(method: str, n_ep: int, batch: int) -> str:
    return str(EST_DIRS[method] / str(n_ep) / f"batch_{batch}" / "estimator.pt")


def value_parquet(method: str, n_ep: int, batch: int) -> str:
    return str(EVAL_DIRS[method] / "results" / "value" / str(n_ep) / f"batch_{batch}" / "predictions.parquet")


def trajectory_parquet(method: str, n_ep: int, batch: int) -> str:
    return str(EVAL_DIRS[method] / "results" / "trajectory" / str(n_ep) / f"batch_{batch}" / "predictions.parquet")


def advantage_parquet(method: str, n_ep: int, batch: int, lam: float) -> str:
    return str(EVAL_DIRS[method] / "results" / "advantage" / f"lambda_{lam:g}"
               / str(n_ep) / f"batch_{batch}" / "predictions.parquet")


def gradient_parquet(method: str, n_ep: int, batch: int, lam: float) -> str:
    return str(EVAL_DIRS[method] / "results" / "gradient" / f"lambda_{lam:g}"
               / str(n_ep) / f"batch_{batch}" / "metrics.parquet")


# ---------------------------------------------------------------------------
# Terminal target
# ---------------------------------------------------------------------------

_eval_data_outputs = []
if EXP.evaluation.trajectories.n_total > 0:
    _eval_data_outputs.append(str(EVAL_DATA_DIR / "trajectories.npz"))
    _eval_data_outputs.append(str(EVAL_DATA_DIR / "reference_gradient.npz"))
if EXP.evaluation.reset_states.n_seed_trajectories > 0:
    _eval_data_outputs.append(str(EVAL_DATA_DIR / "reset_states.npz"))
    if EXP.evaluation.reset_states.n_rollouts_keep > 0:
        _eval_data_outputs.append(str(EVAL_DATA_DIR / "reset_states_rollouts.npz"))

_value_outputs = (
    [value_parquet(m, n, b) for m in METHOD_NAMES for n in EPISODE_SUBSETS for b in range(N_BATCHES)]
    if EXP.evaluation.modes.value else []
)
_trajectory_outputs = (
    [trajectory_parquet(m, n, b) for m in METHOD_NAMES for n in EPISODE_SUBSETS for b in range(N_BATCHES)]
    if EXP.evaluation.modes.trajectory else []
)
_advantage_outputs = [
    advantage_parquet(m, n, b, lam)
    for m in METHOD_NAMES for n in EPISODE_SUBSETS for b in range(N_BATCHES)
    for lam in ADV_LAMBDAS
]
_gradient_outputs = [
    gradient_parquet(m, n, b, lam)
    for m in METHOD_NAMES for n in EPISODE_SUBSETS for b in range(N_BATCHES)
    for lam in GRAD_LAMBDAS
]


rule all:
    input:
        [estimator_ckpt(m, n_ep, b)
         for m in METHOD_NAMES for n_ep in EPISODE_SUBSETS for b in range(N_BATCHES)],
        _eval_data_outputs,
        _value_outputs,
        _trajectory_outputs,
        _advantage_outputs,
        _gradient_outputs,


# ---------------------------------------------------------------------------
# Policy training
# ---------------------------------------------------------------------------

rule train_policy:
    output:
        policy=str(POLICY_DIR / "policy_final.zip"),
    resources:
        mem_mb=16000,
        runtime="24h",
    threads: max(EXP.policy.n_envs, 4)
    log:
        str(LOGS_ROOT / "policy" / "train_policy.log"),
    shell:
        "uv run -m src.train_policy --config {CONFIG_PATH:q} > {log} 2>&1"


# ---------------------------------------------------------------------------
# Data generation (one rule per phase — tuning, training, eval)
# ---------------------------------------------------------------------------

_tuning_outputs = [str(DATA_DIR / "batch_tuning.npz")]
if EXP.data_generation.validation_episodes_per_batch > 0:
    _tuning_outputs.append(str(DATA_DIR / "batch_tuning_validation.npz"))

rule generate_tuning_data:
    input:
        rules.train_policy.output.policy,
    output:
        _tuning_outputs,
    resources:
        mem_mb=200000,
        runtime="12h",
    threads: min(max(EXP.data_generation.n_envs, 4), 64)
    log:
        str(LOGS_ROOT / "data" / "generate_tuning.log"),
    shell:
        "uv run -m src.generate_data --config {CONFIG_PATH:q} "
        "--phase tuning --n-workers {threads} > {log} 2>&1"


_training_outputs = [str(DATA_DIR / f"batch_{i}.npz") for i in range(N_BATCHES)]
if EXP.data_generation.validation_episodes_per_batch > 0:
    _training_outputs += [
        str(DATA_DIR / f"batch_{i}_validation.npz") for i in range(N_BATCHES)
    ]

rule generate_training_data:
    input:
        rules.train_policy.output.policy,
    output:
        _training_outputs,
    resources:
        mem_mb=200000,
        runtime="12h",
    threads: min(max(EXP.data_generation.n_envs, 4), 64)
    log:
        str(LOGS_ROOT / "data" / "generate_training.log"),
    shell:
        "uv run -m src.generate_data --config {CONFIG_PATH:q} "
        "--phase training --n-workers {threads} > {log} 2>&1"


rule generate_trajectories:
    input:
        rules.train_policy.output.policy,
    output:
        str(EVAL_DATA_DIR / "trajectories.npz"),
    resources:
        mem_mb=200000,
        runtime="6h",
    threads: min(max(EXP.data_generation.n_envs, 4), 64)
    log:
        str(LOGS_ROOT / "data" / "generate_trajectories.log"),
    shell:
        "uv run -m src.generate_eval_data --config {CONFIG_PATH:q} "
        "--mode trajectories --n-workers {threads} > {log} 2>&1"


_reset_states_outputs = [str(EVAL_DATA_DIR / "reset_states.npz")]
if EXP.evaluation.reset_states.n_rollouts_keep > 0:
    _reset_states_outputs.append(str(EVAL_DATA_DIR / "reset_states_rollouts.npz"))


rule generate_reset_states:
    input:
        rules.train_policy.output.policy,
    output:
        _reset_states_outputs,
    resources:
        mem_mb=200000,
        runtime="6h",
    threads: min(max(EXP.data_generation.n_envs, 4), 64)
    log:
        str(LOGS_ROOT / "data" / "generate_reset_states.log"),
    shell:
        "uv run -m src.generate_eval_data --config {CONFIG_PATH:q} "
        "--mode reset_states --n-workers {threads} > {log} 2>&1"


rule generate_reference_gradient:
    input:
        policy=rules.train_policy.output.policy,
        traj=str(EVAL_DATA_DIR / "trajectories.npz"),
    output:
        str(EVAL_DATA_DIR / "reference_gradient.npz"),
    resources:
        mem_mb=32000,
        runtime="2h",
    threads: 1
    log:
        str(LOGS_ROOT / "data" / "generate_reference_gradient.log"),
    shell:
        "uv run -m src.generate_eval_data --config {CONFIG_PATH:q} "
        "--mode reference_gradient > {log} 2>&1"


# ---------------------------------------------------------------------------
# Hyperparameter sweep (per method, optional)
# ---------------------------------------------------------------------------

rule sweep:
    input:
        tuning=str(DATA_DIR / "batch_tuning.npz"),
        tuning_val=(
            str(DATA_DIR / "batch_tuning_validation.npz")
            if EXP.data_generation.validation_episodes_per_batch > 0
            else []
        ),
    output:
        csv="{est_dir}/sweeps/sweep_results.csv",
    wildcard_constraints:
        est_dir=EST_DIR_ALTERNATION,
    params:
        method=lambda wc: METHOD_BY_EST_DIR[wc.est_dir],
    resources:
        mem_mb=100000,
        runtime="12h",
    threads: 8
    log:
        "{est_dir}/sweeps/sweep.log",
    shell:
        "uv run launch_episode_sweeps.py --config {CONFIG_PATH:q} "
        "--method {params.method} --launch-agents {threads} > {log} 2>&1"


rule select_tuned_hyperparams:
    input:
        csv="{est_dir}/sweeps/sweep_results.csv",
    output:
        json="{est_dir}/sweeps/tuned_hyperparams.json",
    wildcard_constraints:
        est_dir=EST_DIR_ALTERNATION,
    params:
        method=lambda wc: METHOD_BY_EST_DIR[wc.est_dir],
    log:
        "{est_dir}/sweeps/select_hyperparams.log",
    shell:
        "uv run -m src.select_hyperparameters --config {CONFIG_PATH:q} "
        "--method {params.method} --output {output.json} > {log} 2>&1"


# ---------------------------------------------------------------------------
# Per-cell estimator training: one job per (method × n_episodes × batch)
# ---------------------------------------------------------------------------

def _estimator_inputs(wildcards):
    """Inputs for train_estimator: data batch plus optional tuned hyperparams."""
    batch = int(wildcards.batch)
    deps = {
        "batch": str(DATA_DIR / f"batch_{batch}.npz"),
    }
    if EXP.data_generation.validation_episodes_per_batch > 0:
        deps["batch_val"] = str(DATA_DIR / f"batch_{batch}_validation.npz")

    method = METHOD_BY_EST_DIR[wildcards.est_dir]
    if needs_sweep(method):
        deps["tuned"] = tuned_hp_path(method)
    return deps


rule train_estimator:
    input:
        unpack(_estimator_inputs),
    output:
        ckpt="{est_dir}/{n_ep}/batch_{batch}/estimator.pt",
        stats="{est_dir}/{n_ep}/batch_{batch}/training_stats.json",
        meta="{est_dir}/{n_ep}/batch_{batch}/estimator_metadata.json",
    wildcard_constraints:
        est_dir=EST_DIR_ALTERNATION,
        n_ep=r"\d+",
        batch=r"\d+",
    params:
        method=lambda wc: METHOD_BY_EST_DIR[wc.est_dir],
    resources:
        mem_mb=32000,
        runtime="6h",
    threads: 1
    retries: 2
    log:
        "{est_dir}/{n_ep}/batch_{batch}/train.log",
    shell:
        "uv run -m src.train_estimator --config {CONFIG_PATH:q} "
        "--method {params.method} --batch-idx {wildcards.batch} "
        "--n-episodes {wildcards.n_ep} > {log} 2>&1"


# ---------------------------------------------------------------------------
# Per-cell evaluation: one job per (method × n_episodes × batch [× lambda]).
# ---------------------------------------------------------------------------

rule evaluate_value:
    input:
        ckpt=lambda wc: estimator_ckpt(METHOD_BY_EVAL_DIR[wc.eval_dir], int(wc.n_ep), int(wc.batch)),
        rs=str(EVAL_DATA_DIR / "reset_states.npz"),
    output:
        "{eval_dir}/results/value/{n_ep}/batch_{batch}/predictions.parquet",
    wildcard_constraints:
        eval_dir=EVAL_DIR_ALTERNATION,
        n_ep=r"\d+",
        batch=r"\d+",
    params:
        method=lambda wc: METHOD_BY_EVAL_DIR[wc.eval_dir],
    priority: 100
    resources:
        mem_mb=8000,
        runtime="30m",
    threads: 1
    log:
        "{eval_dir}/results/value/{n_ep}/batch_{batch}/evaluate.log",
    shell:
        "uv run -m src.evaluators.value --config {CONFIG_PATH:q} "
        "--method {params.method} --n-episodes {wildcards.n_ep} "
        "--batch-idx {wildcards.batch} > {log} 2>&1"


rule evaluate_trajectory:
    input:
        ckpt=lambda wc: estimator_ckpt(METHOD_BY_EVAL_DIR[wc.eval_dir], int(wc.n_ep), int(wc.batch)),
        traj=str(EVAL_DATA_DIR / "trajectories.npz"),
    output:
        "{eval_dir}/results/trajectory/{n_ep}/batch_{batch}/predictions.parquet",
    wildcard_constraints:
        eval_dir=EVAL_DIR_ALTERNATION,
        n_ep=r"\d+",
        batch=r"\d+",
    params:
        method=lambda wc: METHOD_BY_EVAL_DIR[wc.eval_dir],
    priority: 90
    resources:
        mem_mb=8000,
        runtime="30m",
    threads: 1
    log:
        "{eval_dir}/results/trajectory/{n_ep}/batch_{batch}/evaluate.log",
    shell:
        "uv run -m src.evaluators.trajectory --config {CONFIG_PATH:q} "
        "--method {params.method} --n-episodes {wildcards.n_ep} "
        "--batch-idx {wildcards.batch} > {log} 2>&1"


rule evaluate_advantage:
    input:
        ckpt=lambda wc: estimator_ckpt(METHOD_BY_EVAL_DIR[wc.eval_dir], int(wc.n_ep), int(wc.batch)),
        rollouts=str(EVAL_DATA_DIR / "reset_states_rollouts.npz"),
    output:
        "{eval_dir}/results/advantage/lambda_{lam}/{n_ep}/batch_{batch}/predictions.parquet",
    wildcard_constraints:
        eval_dir=EVAL_DIR_ALTERNATION,
        lam=r"[\d.]+",
        n_ep=r"\d+",
        batch=r"\d+",
    params:
        method=lambda wc: METHOD_BY_EVAL_DIR[wc.eval_dir],
    priority: 50
    resources:
        mem_mb=80000,
        runtime="30m",
    threads: 1
    log:
        "{eval_dir}/results/advantage/lambda_{lam}/{n_ep}/batch_{batch}/evaluate.log",
    shell:
        "uv run -m src.evaluators.advantage --config {CONFIG_PATH:q} "
        "--method {params.method} --n-episodes {wildcards.n_ep} "
        "--batch-idx {wildcards.batch} --lam {wildcards.lam} > {log} 2>&1"


rule evaluate_gradient:
    input:
        ckpt=lambda wc: estimator_ckpt(METHOD_BY_EVAL_DIR[wc.eval_dir], int(wc.n_ep), int(wc.batch)),
        traj=str(EVAL_DATA_DIR / "trajectories.npz"),
        ref=str(EVAL_DATA_DIR / "reference_gradient.npz"),
    output:
        "{eval_dir}/results/gradient/lambda_{lam}/{n_ep}/batch_{batch}/metrics.parquet",
    wildcard_constraints:
        eval_dir=EVAL_DIR_ALTERNATION,
        lam=r"[\d.]+",
        n_ep=r"\d+",
        batch=r"\d+",
    params:
        method=lambda wc: METHOD_BY_EVAL_DIR[wc.eval_dir],
    priority: 10
    resources:
        mem_mb=16000,
        runtime="1h",
    threads: 1
    log:
        "{eval_dir}/results/gradient/lambda_{lam}/{n_ep}/batch_{batch}/evaluate.log",
    shell:
        "uv run -m src.evaluators.gradient --config {CONFIG_PATH:q} "
        "--method {params.method} --n-episodes {wildcards.n_ep} "
        "--batch-idx {wildcards.batch} --lam {wildcards.lam} > {log} 2>&1"
