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

### Run Complete Experiment Pipeline
```bash
# Simple shell script that runs all steps sequentially
./run_experiment.sh

# Or run steps individually:
uv run -m src.train_policy --config configs/example_config.yaml
uv run -m src.generate_data --config configs/example_config.yaml
uv run -m src.train_estimator --config configs/example_config.yaml --method monte_carlo --batch-idx 0 --no-overwrite

# Generate specific range of batches:
uv run -m src.generate_data --config configs/example_config.yaml --start-batch-idx 0 --end-batch-idx 10  # Batches 0-9
uv run -m src.generate_data --config configs/example_config.yaml --start-batch-idx 10  # Resume from batch 10
```

### Run All Estimators (Multiple Methods × Multiple Batches)
```bash
# Sequential (simple, good for debugging)
uv run -m src.run_all_estimators --config configs/example_config.yaml --mode sequential --no-overwrite

# Parallel across GPUs (local machine)
uv run -m src.run_all_estimators --config configs/example_config.yaml --mode parallel --no-overwrite

# Cluster mode (SGE array jobs)
uv run -m src.run_all_estimators --config configs/example_config.yaml --mode cluster --no-overwrite

# Cluster mode with custom settings
uv run -m src.run_all_estimators --config configs/example_config.yaml --mode cluster \
    --grid-mem 16g \
    --grid-ncpus 4 \
    --max-concurrent 10 \
    --no-overwrite

# To overwrite existing models, use --overwrite instead of --no-overwrite
uv run -m src.run_all_estimators --config configs/example_config.yaml --mode sequential --overwrite

# Or call the bash script directly for parallel mode
./run_parallel_estimators.sh configs/example_config.yaml monte_carlo,td 10 false
```

**Overwrite Behavior:**
- `--overwrite`: Always train estimators, overwriting existing models
- `--no-overwrite`: Skip training if model file already exists (useful for resuming interrupted experiments)
- **Required**: You must specify either `--overwrite` or `--no-overwrite` (no default)

**Cluster Mode Details:**
- Submits one SGE array job per method
- Each array task (identified by `SGE_TASK_ID`) trains one batch
- `SGE_TASK_ID` is automatically converted to batch index (1-indexed → 0-indexed)
- Monitor jobs with `qstat`
- View logs in `*.o*` and `*.e*` files
- Use `--grid-mem` to specify memory per job (default: 8g)
- Use `--grid-ncpus` to specify CPUs per job (default: 1)
  - Recommended: 4-6 CPUs for large networks (Humanoid, Ant) or RBF features
  - Recommended: 1-2 CPUs for simple networks (CartPole)
- Use `--max-concurrent N` to limit concurrent tasks per method (useful for resource management)

### Evaluate Results
```bash
# Generate predictions from trained models
uv run -m src.evaluate --config configs/example_config.yaml
# Saves to each method's eval dir: .../eval_NNN/results/

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
```bash
# Launch sweep with automatic config injection
uv run launch_episode_sweeps.py --config configs/cartpole/config.yaml --method monte_carlo

# Launch sweep with N parallel agents
uv run launch_episode_sweeps.py --config configs/cartpole/config.yaml --method monte_carlo --launch-agents 4

# Dry run to see generated sweep config
uv run launch_episode_sweeps.py --config configs/cartpole/config.yaml --method monte_carlo --dry-run

# Manual method (after sweep created):
wandb agent <sweep-id>
```

**New Sweep Workflow:**
- **Single sweep per method**: Each hyperparameter set trains on ALL episode counts in one run
- **Episode counts**: Loaded from `config.value_estimators.training.episode_subsets`
- **Seed variation**: Each episode count uses a different seed for diversity
- **Aggregated metrics**: Reports combined statistics across all episode counts
  - `final/mean_val_mc_loss`: Mean of best losses across all episode counts (optimized by sweep)
  - `final/std_val_mc_loss`: Standard deviation across episode counts
  - `final/{N}ep/best_mc_loss`: Best loss for specific episode count


**How it works:**
- Sweep configs in `configs/sweeps/sweep_*.yaml` define hyperparameter search space
- **Monte Carlo / TD**: Tunes learning rate, batch size
- **Least Squares (MC/TD)**: Tunes ridge_lambda
- **Always uses `batch_tuning.npz` for training and `batch_tuning_validation.npz` for validation**
- **Optimizes mean validation MC loss across all episode counts** (`final/mean_val_mc_loss`)
- **W&B mode**: Set `wandb-mode: offline` in sweep config to avoid rate limits
- Results saved under the method's estimator directory: `.../{method}_estimator_NNN/sweeps/<run_id>/<N>ep/`

**Customizing sweeps:**
- Edit `configs/sweeps/sweep_*.yaml` to change search range or method (bayes/grid/random)
- Supported parameters: learning_rate, target_update_rate, batch_size, ridge_lambda
- Episode counts are loaded from the experiment config's `episode_subsets` field
- Set `wandb-mode: offline` to avoid rate limits with many parallel agents

## Architecture

### Project Structure

The codebase is organized into two main parts:

**Training Pipeline** (`src/`):
- `config.py`: Configuration system (dataclasses, YAML parsing)
- `registry.py`: Parameter-based directory resolution (find-or-create numbered dirs)
- `train_policy.py`: Policy training with SB3
- `generate_data.py`: Episode data generation
- `train_estimator.py`: Value estimator training
- `run_all_estimators.py`: Batch training orchestration
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
2. Copy and edit main config: `cp configs/cartpole/config.yaml configs/my_experiment/config.yaml`
3. Copy method configs: `cp configs/cartpole/*.yaml configs/my_experiment/` (excluding config.yaml)
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

**For data generation**, use `--start-batch-idx` and `--end-batch-idx` to control which batches to generate:
```bash
# Resume from batch 10 onwards
uv run -m src.generate_data --config config.yaml --start-batch-idx 10

# Generate only batches 5-15 (exclusive end)
uv run -m src.generate_data --config config.yaml --start-batch-idx 5 --end-batch-idx 15

# Generate first 20 batches only
uv run -m src.generate_data --config config.yaml --end-batch-idx 20
```

**For estimator training**, use `--no-overwrite` to skip already-trained models:
```bash
uv run -m src.run_all_estimators --config config.yaml --mode sequential --no-overwrite
```

The framework checks for existing `estimator.pt` files and skips training if found.

To force retraining all models, use `--overwrite`:
```bash
uv run -m src.run_all_estimators --config config.yaml --mode sequential --overwrite
```

## Supported Environments

- **Classic Control**: CartPole-v1, Acrobot-v1, MountainCar-v0
- **Atari**: ALE/Pong-v5, ALE/Breakout-v5 (requires `gymnasium[atari,accept-rom-license]`)
- **MuJoCo**: HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Humanoid-v5 (requires `gymnasium[mujoco]`)
- Any Gymnasium-compatible environment

## Key Dependencies

- **PyTorch**: Neural network implementation
- **Stable Baselines 3**: Policy training (PPO, A2C, SAC, TD3)
- **Gymnasium**: Environment API
- **NumPy**: Data storage and manipulation
- **Weights & Biases**: Experiment tracking
- **PyYAML**: Configuration management
