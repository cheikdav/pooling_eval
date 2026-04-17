I# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Rules

- **Simplicity**: This is a research project and code does not need to be production level. Simplicity on the other hand is key to iterate quickly. Make sure to keep code as simple and readable as possible.
- **Factored**: Factor code as much as possible, in the limit of readability.
- **Comments**: Write only high level comments and no need for detailed documentation.
- **Git Commits**: Never include Claude Code references, attributions, or Co-Authored-By tags in commit messages. Keep commits clean, professional, and concise (short subject line with optional bulleted details).
- **Python Execution**: Always use `uv run` to execute Python scripts, never use `python` directly. Example: `uv run -m src.train_policy` instead of `python -m src.train_policy`.


## Project Overview

This is a research framework for evaluating value function estimation methods in reinforcement learning. The core workflow:

1. Train a policy using Stable Baselines 3 (PPO/A2C/SAC/TD3)
2. Generate batches of episodes using the trained policy
3. Train multiple value estimators (Monte Carlo, TD) on the same data
4. Compare estimation performance across methods

All experiments are reproducible via configuration files and logged with Weights & Biases.

## Essential Commands

### Install Dependencies
```bash
uv sync                    # Recommended (uses pyproject.toml)
pip install -e .          # Alternative
```

### Estimate Storage Requirements
```bash
# Calculate episodes per batch needed for target storage
uv run -m src.estimate_storage --config configs/example_config.yaml --target-gb 10

# Custom number of batches
uv run -m src.estimate_storage --config configs/example_config.yaml --target-gb 5 --n-batches 100

# Use more samples for better estimate
uv run -m src.estimate_storage --config configs/example_config.yaml --target-gb 10 --sample-episodes 50
```

This tool generates a small batch of episodes, measures storage size, and calculates the optimal `episodes_per_batch` to achieve your target total storage across all batches. Useful for planning experiments with storage constraints.

### Run Complete Experiment Pipeline (Snakemake)

The pipeline is orchestrated by a Snakefile at the repository root. Snakemake
handles DAG resolution, state tracking, retry, and cluster dispatch. The same
workflow runs locally, on SGE, on SLURM, or in the cloud with a CLI flag
change.

```bash
# Local, 4 parallel jobs
uv run snakemake --cores 4 --config experiment_config=configs/hopper/config.yaml

# Convenience wrapper (defaults: config=configs/test_mini/config.yaml, cores=4)
./run_experiment.sh configs/hopper/config.yaml

# Dry-run: print the DAG without executing anything
uv run snakemake -n --config experiment_config=configs/hopper/config.yaml

# Visualize the DAG as a PNG (requires graphviz)
uv run snakemake --dag --config experiment_config=configs/hopper/config.yaml \
    | dot -Tpng > dag.png

# Re-run a specific stage explicitly
uv run snakemake --forcerun train_estimator \
    --cores 4 --config experiment_config=configs/hopper/config.yaml
```

The workflow has one required config entry: `experiment_config` is the path
to a pooling-eval experiment YAML. All other state (which batches to train,
which methods to sweep, which episode counts to evaluate) is read from the
experiment YAML at Snakefile preamble time.

### Running on a Cluster

SGE via the generic cluster executor:
```bash
uv run snakemake \
    --executor cluster-generic \
    --cluster-generic-submit-cmd \
        "qsub -V -cwd -pe threaded {threads} -l h_vmem={resources.mem_mb}M" \
    --jobs 50 \
    --config experiment_config=configs/hopper/config.yaml
```

SLURM via the first-class slurm plugin (requires `snakemake-executor-plugin-slurm`):
```bash
uv run snakemake --executor slurm --jobs 50 \
    --config experiment_config=configs/hopper/config.yaml
```

`threads:`, `resources: mem_mb=`, and `resources: runtime=` on each rule are
translated automatically to the scheduler's resource request.

### Running Individual Stages

Each stage script still has its own CLI and can be invoked directly for
debugging. Snakemake is not required for ad-hoc single-stage runs.
```bash
uv run -m src.train_policy --config configs/hopper/config.yaml
uv run -m src.generate_data --config configs/hopper/config.yaml --phase training
uv run -m src.train_estimator --config configs/hopper/config.yaml \
    --method monte_carlo --batch-idx 3 --n-episodes 50
uv run -m src.evaluate --config configs/hopper/config.yaml
```

`src.train_estimator` trains **one estimator per invocation** — one method on
one batch at one episode count. Fan-out over the `(method × batch × n_episodes)`
cross-product is Snakemake's job. This makes per-cell retry possible: if a
single estimator fails, only that cell is re-trained, not the whole method's
batch.

### Retry and Resume

Snakemake resume is automatic: if a pipeline is interrupted, re-running
`snakemake` on the same config picks up from where it stopped based on which
output files already exist.

Per-rule retries are declared in the Snakefile (`retries: 2` on `train_estimator`).
Transient failures of individual cells are automatically retried.

If the previous run was killed hard, Snakemake may leave a lock file:
```bash
uv run snakemake --unlock --config experiment_config=configs/hopper/config.yaml
uv run snakemake --rerun-incomplete \
    --cores 4 --config experiment_config=configs/hopper/config.yaml
```

### Evaluate Results
`src.evaluate` runs as part of the Snakemake pipeline via the `evaluate_all`
rule. To run it standalone (e.g. re-evaluating without retraining):
```bash
uv run -m src.evaluate --config configs/hopper/config.yaml --n-jobs 1
```
Note: evaluate.py's ProcessPoolExecutor path (`--n-jobs > 1`) has a pre-existing
hang with torch CUDA init on some systems. The Snakemake rule uses `--n-jobs 1`
by default to work around this.

```bash
# Launch interactive dashboard for analysis
uv run streamlit run src/analysis/app.py
# Access at http://localhost:8501
```

### Disable Weights & Biases
```bash
uv run -m src.train_estimator ... --no-wandb
# Or set use_wandb: false in config
```

### Hyperparameter Tuning with W&B Sweeps

Methods opt into the sweep path by adding a `tuning:` field to their method
config YAML. Methods without `tuning:` skip the sweep stage and train with
their fixed hyperparams. The Snakemake workflow automatically schedules
`sweep → select_tuned_hyperparams → train_estimator` for opt-in methods.

```yaml
# configs/hopper/monte_carlo.yaml
name: monte_carlo
type: monte_carlo
learning_rate: 0.001  # fallback if tuning is skipped
batch_size: 128
tuning: {}            # marker — enables sweep path
```

The tuned hyperparameters are written to
`{estimator_dir}/sweeps/tuned_hyperparams.json` (inside the results tree).
`train_estimator` reads this JSON before constructing the estimator and
merges the tuned values into the in-memory method config. **The source YAML
is never mutated** — this is the key change from the pre-Snakemake workflow.

Invocations (each sweep trial calls `train_one_estimator` once per episode
count in the experiment config):
```bash
# Standalone sweep (Snakemake would call this automatically as a rule)
uv run launch_episode_sweeps.py --config configs/hopper/config.yaml --method monte_carlo --launch-agents 4

# Standalone selection: writes tuned_hyperparams.json to the results tree
uv run -m src.select_hyperparameters --config configs/hopper/config.yaml --method monte_carlo
```

**Sweep mechanics:**
- One W&B sweep per method, N parallel agents (configured via `--launch-agents`)
- Each trial trains on ALL `episode_subsets` and reports
  `final/mean_val_mc_loss` as the sweep objective
- Results CSV: `{estimator_dir}/sweeps/sweep_results.csv`
- Selection picks best trial by mean-normalized-loss across episode counts
- Selected hyperparams land in `{estimator_dir}/sweeps/tuned_hyperparams.json`
- Sweep configs in `configs/sweeps/sweep_*.yaml` define the search space
- **Monte Carlo / TD**: tunes learning_rate, batch_size
- **Least Squares (MC/TD)**: tunes ridge_lambda
- Uses `batch_tuning.npz` for training, `batch_tuning_validation.npz` for validation
- Set `wandb-mode: offline` in sweep config to avoid rate limits

## Architecture

### Project Structure

The codebase is organized into two main parts:

**Training Pipeline** (`src/`):
- `config.py`: Configuration system (dataclasses, YAML parsing)
- `registry.py`: Parameter-based directory resolution (find-or-create numbered dirs)
- `train_policy.py`: Policy training with SB3
- `generate_data.py`: Episode data generation
- `train_estimator.py`: Value estimator training (one estimator per invocation)
- `tune_hyperparameters.py`: W&B hyperparameter sweeps
- `evaluate.py`: Model evaluation and prediction generation
- `estimators/`: Value estimator implementations (Monte Carlo, TD, etc.)

**Result Analysis** (`src/analysis/`):
- `app.py`: Streamlit dashboard for interactive visualization
- `metrics.py`: Metric definitions and computation functions
- `analyze_variance_ratios.ipynb`: Jupyter notebook for variance analysis

### Configuration System (src/config.py)

All experiments are defined by YAML config files organized in environment-specific directories. Each experiment has:

**Directory Structure:**
```
configs/
  <environment_name>/
    config.yaml              # Main config (shared parameters)
    monte_carlo.yaml         # Method-specific parameters
    td.yaml
    least_squares_mc.yaml
    least_squares_td.yaml
```

**Main Config (config.yaml):**
- **ExperimentConfig**: Top-level container
  - **EnvironmentConfig**: Gymnasium environment name
  - **PolicyConfig**: SB3 algorithm and hyperparameters
  - **DataGenerationConfig**: Number of batches and episodes per batch
  - **ValueEstimatorsConfig**: Shared training parameters + list of methods to load
  - **NetworkConfig**: Neural network architecture (hidden sizes, activation)
  - **LoggingConfig**: Weights & Biases settings
  - **EvaluationConfig**: Eval batch size + paired state settings
  - `data_root`: Root directory for all experiment outputs (default: ".")
  - `logs_root`: Root directory for logs (defaults to data_root)

**Code Versions (`code_versions.yaml` at project root):**
- Standalone file tracking code version numbers per pipeline stage
- Loaded automatically when parsing any config
- Bump the relevant version when code changes affect results
```yaml
policy: 1      # policy training code
data: 1        # data generation code
estimator: 1   # estimator training code
evaluation: 1  # evaluation code
```

**Method Config Files (e.g., monte_carlo.yaml):**
- **Type**: Estimator type (monte_carlo, td, least_squares_mc, least_squares_td)
- **Method-specific parameters**: Learning rate, target_update_rate, etc.

**Legacy Format Support:**
The system also supports legacy configs with all parameters in a single file using `method_configs` list.

### Data Pipeline

**Batch Structure:**

Data generation creates multiple types of batches for different purposes:

1. **Regular batches** (`batch_0.npz`, `batch_1.npz`, ..., `batch_N.npz`):
   - Number controlled by `data_generation.n_batches`
   - Each contains `episodes_per_batch` episodes
   - Used for training value estimators in main experiments

2. **Tuning batch** (`batch_tuning.npz`):
   - Number of episodes controlled by `data_generation.tuning_episodes`
   - Used exclusively for hyperparameter search via W&B sweeps
   - Created only if `tuning_episodes > 0`
   - Has corresponding validation set `batch_tuning_validation.npz` if `validation_episodes_per_batch > 0`

3. **Validation batches** (`batch_0_validation.npz`, `batch_1_validation.npz`, ..., `batch_tuning_validation.npz`):
   - Number of episodes per validation set controlled by `data_generation.validation_episodes_per_batch`
   - Each training batch (including tuning batch) has a corresponding validation set
   - Used for early stopping and model selection during training
   - Created only if `validation_episodes_per_batch > 0`

4. **Evaluation batch** (`batch_eval.npz`):
   - Number of episodes controlled by `evaluation.eval_episodes`
   - For final held-out evaluation
   - Created only if `eval_episodes > 0`

All batches use different random seeds (sequential from base seed) to ensure independence.

**Episode Data Format** (NPZ files in `.../data_NNN/data/`):
```python
batch = np.load('batch_0.npz', allow_pickle=True)
batch['observations']       # List of arrays, one per episode: [(T_i, obs_dim), ...]
batch['actions']           # List[(T_i, act_dim)]
batch['rewards']           # List[(T_i,)]
batch['dones']             # List[(T_i,)]
batch['next_observations'] # List[(T_i, obs_dim)]
batch['episode_lengths']   # (n_episodes,) - metadata
batch['episode_returns']   # (n_episodes,) - metadata
```

Each batch contains `episodes_per_batch` full episodes with variable lengths. Episodes are stored as a list of arrays rather than padded arrays.

### Value Estimator Architecture (src/estimators/)

All estimators inherit from **ValueEstimator** (base.py):

**Common Components:**
- **ValueNetwork**: Simple MLP (obs_dim → hidden layers → 1 value output)
- **train_step(batch)**: Performs one gradient update (MSE loss between predictions and targets)
- **evaluate(batch)**: Computes MSE/MAE metrics without gradient updates
- **compute_targets(batch)**: Abstract method - each estimator implements this differently

**Method-Specific Target Computation:**

1. **MonteCarloEstimator** (monte_carlo.py)
   - Computes full discounted returns: G_t = Σ γ^k * r_{t+k}
   - Simple, unbiased, but high variance

2. **TDEstimator** (td.py)
   - Maintains a Polyak-averaged target network
   - Computes TD(0) targets: r + γ * V_target(s')

**Key Design Pattern**: The `compute_targets()` method returns torch tensors that serve as regression targets for training the value network. This allows different estimation strategies while sharing the same training loop.

### Training Loop (src/train_estimator.py)

**Early Stopping Logic:**
- Tracks evaluation loss every `eval_frequency` epochs
- Stops if relative improvement < `convergence_threshold` for `convergence_patience` epochs
- Example: If loss improves by < 0.0001 for 50 consecutive evaluations, training stops

**Checkpointing:**
- Saves `estimator.pt` at the end of training
- Saves `training_stats.json` with summary statistics

**Fault Tolerance:**
- Each estimator checks if `estimator.pt` exists before training
- Allows resuming experiments by skipping completed jobs

### Experiment Directory Structure

Experiments are organized by parameter identity in a 5-level hierarchy. Each numbered directory contains a `params.json` recording the parameters that produced it. When running a pipeline step, the system scans existing directories for a parameter match; if found it reuses that directory, otherwise it creates the next sequential number.

```
experiments/{env_name}/                                    # e.g. Hopper-v5
├── policy_001/                                            # policy params + policy code version
│   ├── params.json
│   ├── policy/
│   │   ├── policy_final.zip
│   │   ├── policy_metadata.json
│   │   └── config.yaml
│   ├── data_001/                                          # data gen params + data code version
│   │   ├── params.json
│   │   ├── data/
│   │   │   ├── batch_0.npz, batch_1.npz, ...
│   │   │   ├── batch_eval.npz
│   │   │   └── paired_states.npz
│   │   ├── monte_carlo_estimator_001/                     # method config + estimator code version
│   │   │   ├── params.json
│   │   │   ├── 10/batch_0/estimator.pt                    # episode count subfolders
│   │   │   ├── 20/batch_0/estimator.pt
│   │   │   └── eval_001/                                  # eval params + eval code version
│   │   │       ├── params.json
│   │   │       └── results/
│   │   │           ├── ground_truth/ground_truth_returns.parquet
│   │   │           └── monte_carlo/{n_episodes}/predictions.parquet
│   │   └── td_estimator_001/                              # independent from MC
│   │       └── ...
│   └── data_002/                                          # different data gen params
│       └── ...
└── policy_002/                                            # different policy params
    └── ...
```

**Parameter identity per level:**

| Level | What defines identity | What it stores |
|-------|----------------------|----------------|
| **Policy** | algorithm, hyperparams, gamma, network, seed, `policy` code version | trained policy |
| **Data** | n_batches, episodes_per_batch, validation/tuning episodes, `data` code version | episode batches |
| **Estimator** | method config (lr, batch_size, etc.), shared training params, `estimator` code version. **Excludes** `episode_subsets` | trained models (per episode count) |
| **Eval** | eval_episodes, paired state config, `evaluation` code version | predictions + ground truth |

**Key properties:**
- Changing a method's learning rate creates a new estimator dir without affecting other methods
- Adding a new episode count to `episode_subsets` reuses the existing estimator dir (just adds a subfolder)
- Changing data params creates a new data dir, forcing new estimator training
- `code_versions.yaml` versions are included in each level's identity

**Registry module (`src/registry.py`):**
- `resolve_dir(parent, prefix, params)` — find-or-create numbered directory
- `get_policy_params(config)`, `get_data_params(config)`, `get_estimator_params(config, method)`, `get_eval_params(config)` — extract identity params from config

### Reproducibility

**Seed Management:**
- Config `seed` controls policy training and initial RNG state
- Each data batch uses `seed + batch_idx` for diversity
- All NumPy and PyTorch seeds set consistently in each script

**Code Versioning:**
- `code_versions.yaml` tracks version numbers for each pipeline stage
- Included in `params.json` at each directory level
- To reproduce: check out the commit that corresponds to the code version, use the same config

**To reproduce experiments exactly:**
1. Use identical config file and `code_versions.yaml`
2. Same environment versions (Gymnasium, SB3, PyTorch)
3. Same seed value

## Development Notes

### Adding New Value Estimators

1. Create new file in `src/estimators/` (e.g., `my_estimator.py`)
2. Inherit from `ValueEstimator` base class
3. Implement `compute_targets(batch)` method
4. Add config dataclass to `src/config.py` if method-specific params needed
5. Update factory function in `src/train_estimator.py` (`create_estimator()`)
6. Add method name to config's `value_estimators.methods` list

### Working with Batched Data

When implementing `compute_targets()`, remember:
- Input batch contains episodes as lists of arrays (variable length)
- You'll often need to flatten: `np.concatenate(batch['observations'])`
- Must return torch.Tensor of shape `(total_transitions,)` matching flattened observations
- See existing estimators for examples of handling episode boundaries

### Adding New Analysis Metrics

To add a new metric for visualization in the Streamlit dashboard:

1. Implement computation function in `src/analysis/metrics.py`:
```python
@st.cache_data
def compute_my_metric(df, stats):
    """Compute my custom metric."""
    # df: raw predictions DataFrame
    # stats: aggregated statistics DataFrame
    result = stats.copy()
    result['metric_value'] = # your computation here
    return result
```

2. Add metric definition to METRICS dict in `src/analysis/metrics.py`:
```python
METRICS = {
    'my_metric': {
        'name': 'My Metric Name',
        'description': 'Description shown in UI',
        'reference_line': 1.0,  # Optional: value for reference line (or None)
        'reference_label': 'Baseline',  # Optional: label for reference line (or None)
        'compute_fn': compute_my_metric  # Reference to the computation function
    }
}
```

The dashboard will automatically make the new metric available in the metric selector. The `compute_fn` field links the metric to its computation function, eliminating the need to manually update a dispatcher.

### Device Management

- Device auto-detection: uses CUDA if available, else CPU
- Can override with `device='cpu'` or `device='cuda'`
- To force CPU: `export CUDA_VISIBLE_DEVICES=""`
- GPU array scripts distribute jobs across available GPUs automatically

### Wandb Integration

- All runs tagged with `experiment_id` for grouping
- Training metrics logged: `train/loss`, `train/mse`, `train/mae`, `train/mean_value`, `train/mean_target`, `train/best_loss`
- Run name format: `{method} ({env_name}, #{batch_name}, #ep {num_episodes})`

**Offline Mode** (to avoid rate limits):

Set `wandb_mode: "offline"` in your config to store runs locally and sync at the end of each training:

```yaml
logging:
  use_wandb: true
  wandb_mode: "offline"  # Syncs each run after training completes
```

Benefits:
- No rate limits during training
- Faster training (no network overhead)
- Each run syncs automatically at completion
- Works well with cluster jobs (each task syncs independently)

Offline runs are stored in `wandb_offline/` under the relevant estimator directory and automatically synced via `wandb sync` at the end of each training.

## Common Patterns

### Creating a New Experiment

1. Create a new directory: `mkdir configs/my_experiment`
2. Copy and edit main config: `cp configs/hopper/config.yaml configs/my_experiment/config.yaml`
3. Copy method configs: `cp configs/hopper/*.yaml configs/my_experiment/` (excluding config.yaml)
4. Edit `configs/my_experiment/config.yaml`:
   - Adjust environment, policy algorithm, timesteps, etc.
   - Update `value_estimators.methods` list to add/remove methods
   - Set `data_root` if storing outputs on a different filesystem
5. Edit method configs (e.g., `monte_carlo.yaml`) to adjust method-specific parameters
6. Run pipeline: `uv run -m src.train_policy --config configs/my_experiment/config.yaml`

**No need to change `experiment_id` to avoid overwriting** — the parameter-based directory structure automatically separates experiments with different parameters. The `experiment_id` is only used for W&B grouping and log paths.

### Debugging Training Issues

Check training stats (find the relevant estimator directory first):
```bash
cat experiments/{env}/policy_NNN/data_NNN/{method}_estimator_NNN/{n_episodes}/batch_0/training_stats.json | jq '.'
```

### Resuming Interrupted Experiments

Snakemake handles resume automatically: re-running `snakemake` on the same
config picks up where the previous invocation stopped based on which output
files already exist on disk. No flags needed.

If the previous run was killed hard, snakemake may leave a lock file:
```bash
uv run snakemake --unlock --config experiment_config=configs/hopper/config.yaml
uv run snakemake --rerun-incomplete \
    --cores 4 --config experiment_config=configs/hopper/config.yaml
```

To force re-training of one specific cell, delete its `estimator.pt` and
re-run:
```bash
rm experiments/.../monte_carlo_estimator_001/10/batch_3/estimator.pt
uv run snakemake --cores 4 --config experiment_config=configs/hopper/config.yaml
```

To force re-running an entire rule:
```bash
uv run snakemake --forcerun train_estimator --cores 4 \
    --config experiment_config=configs/hopper/config.yaml
```

## Supported Environments

- **MuJoCo**: HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Humanoid-v5 (requires `gymnasium[mujoco]`)

## Key Dependencies

- **PyTorch**: Neural network implementation
- **Stable Baselines 3**: Policy training (PPO, A2C, SAC, TD3)
- **Gymnasium**: Environment API
- **NumPy**: Data storage and manipulation
- **Weights & Biases**: Experiment tracking
- **PyYAML**: Configuration management
