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
3. Train multiple value estimators (Monte Carlo, DQN) on the same data
4. Compare estimation performance across methods

All experiments are reproducible via configuration files and logged with Weights & Biases.

## Essential Commands

### Install Dependencies
```bash
uv sync                    # Recommended (uses pyproject.toml)
pip install -e .          # Alternative
```

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
    --max-concurrent 10 \
    --no-overwrite

# To overwrite existing models, use --overwrite instead of --no-overwrite
uv run -m src.run_all_estimators --config configs/example_config.yaml --mode sequential --overwrite

# Or call the bash script directly for parallel mode
./run_parallel_estimators.sh configs/example_config.yaml monte_carlo,dqn 10 false
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
- Use `--max-concurrent N` to limit concurrent tasks per method (useful for resource management)

### Evaluate Results
```bash
# Generate predictions from trained models
uv run -m src.evaluate --config configs/example_config.yaml
# Generates: experiments/<exp_id>/results/predictions.csv

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
# Quick method: Use helper script
./launch_sweep.sh monte_carlo      # Run 1 agent
./launch_sweep.sh dqn 4            # Run 4 parallel agents

# Manual method:
# 1. Create sweep
wandb sweep configs/sweep_monte_carlo.yaml

# 2. Run agent(s) with the sweep ID returned above
wandb agent <sweep-id>

# Launch separate sweeps per episode count (uses episodes from sweep config by default)
uv run launch_episode_sweeps.py --method monte_carlo
uv run launch_episode_sweeps.py --method least_squares_mc

# Or specify custom episode counts
uv run launch_episode_sweeps.py --method monte_carlo --episodes 50 100 200 400

# Auto-launch multiple agents per sweep to parallelize
uv run launch_episode_sweeps.py --method monte_carlo --launch-agents 4  # 4 agents per sweep
```

**Parallel Initialization:**
- By default, multiple initializations within a sweep run are trained sequentially
- To speed up sweeps, enable parallel initialization training in sweep configs:
  ```yaml
  parameters:
    parallel-inits:
      value: true
    n-jobs:
      value: 4  # Number of parallel processes
  ```
- This trains all initializations for a hyperparameter combo in parallel, then aggregates results
- Each hyperparameter combo still appears as a single W&B run with aggregate metrics

**How it works:**
- Sweep configs in `configs/sweep_*.yaml` define hyperparameter search space
- **Monte Carlo / DQN**: Tunes learning rate, batch size, num episodes
- **Least Squares (MC/TD)**: Tunes ridge_lambda, n_components, preprocess_fraction, num episodes
- Uses random search to find optimal values
- **Always uses `batch_tuning.npz` for training and `batch_tuning_validation.npz` for validation**
- **Optimizes validation MC loss** (`final/best_val_mc_loss`) to prevent overfitting
- **W&B mode**: Set `wandb-mode: offline` in sweep config to avoid rate limits (syncs at end) or `online` for real-time monitoring
- Results saved to `experiments/<exp_id>/sweeps/<method>/<run_id>/`
- All runs tracked in W&B for comparison

**Separate sweeps per episode count:**
- `launch_episode_sweeps.py` creates one sweep per episode count
- Uses episode values from sweep config by default (or specify with `--episodes`)
- Each sweep varies hyperparameters (learning rate, ridge_lambda, etc.) for a fixed episode count
- Use `--launch-agents N` to automatically launch N parallel agents per sweep for faster tuning
- Useful for analyzing performance scaling with data size

**Customizing sweeps:**
- Edit `configs/sweep_*.yaml` to change search range or method (bayes/grid/random)
- Supported parameters: learning_rate, target_update_rate, batch_size, num_episodes, ridge_lambda, n_components, preprocess_fraction, wandb_mode
- Set `wandb-mode: offline` to avoid rate limits with many parallel agents (recommended for large sweeps)

## Architecture

### Project Structure

The codebase is organized into two main parts:

**Training Pipeline** (`src/`):
- `config.py`: Configuration system
- `train_policy.py`: Policy training with SB3
- `generate_data.py`: Episode data generation
- `train_estimator.py`: Value estimator training
- `run_all_estimators.py`: Batch training orchestration
- `tune_hyperparameters.py`: W&B hyperparameter sweeps
- `evaluate.py`: Model evaluation and prediction generation
- `estimators/`: Value estimator implementations (Monte Carlo, DQN, etc.)

**Result Analysis** (`src/analysis/`):
- `app.py`: Streamlit dashboard for interactive visualization
- `metrics.py`: Metric definitions and computation functions
- `analyze_variance_ratios.ipynb`: Jupyter notebook for variance analysis

### Configuration System (src/config.py)

All experiments are defined by a YAML config file that gets parsed into nested dataclasses:

- **ExperimentConfig**: Top-level container
  - **EnvironmentConfig**: Gymnasium environment name
  - **PolicyConfig**: SB3 algorithm and hyperparameters
  - **DataGenerationConfig**: Number of batches and episodes per batch
  - **ValueEstimatorsConfig**: Which methods to evaluate + training hyperparameters
    - **MonteCarloConfig**, **DQNConfig**: Method-specific parameters
  - **NetworkConfig**: Neural network architecture (hidden sizes, activation)
  - **LoggingConfig**: Weights & Biases settings

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
   - Number of episodes controlled by `data_generation.eval_episodes`
   - For final held-out evaluation
   - Created only if `eval_episodes > 0`

All batches use different random seeds (sequential from base seed) to ensure independence.

**Episode Data Format** (NPZ files in `experiments/<exp_id>/data/`):
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

2. **DQNEstimator** (dqn.py)
   - Maintains separate target network (frozen, updated periodically)
   - Computes targets: r + γ * V_target(s')
   - Optional Double DQN: averages online and target network

**Key Design Pattern**: The `compute_targets()` method returns torch tensors that serve as regression targets for training the value network. This allows different estimation strategies while sharing the same training loop.

### Training Loop (src/train_estimator.py)

**Early Stopping Logic:**
- Tracks evaluation loss every `eval_frequency` epochs
- Stops if relative improvement < `convergence_threshold` for `convergence_patience` epochs
- Example: If loss improves by < 0.0001 for 50 consecutive evaluations, training stops

**Checkpointing:**
- Saves `estimator_final.pt` at the end of training with the best model found across all initializations
- Saves `training_stats.json` with summary statistics

**Fault Tolerance:**
- Each estimator checks if `estimator_final.pt` exists before training
- Allows resuming experiments by skipping completed jobs

### Experiment Directory Structure

```
experiments/<experiment_id>/
├── config.yaml                    # Copy of configuration
├── policy/
│   ├── policy_final.zip          # Trained SB3 policy
│   └── policy_checkpoint_*.zip   # Intermediate checkpoints
├── data/
│   ├── batch_0.npz               # Episode data batches
│   ├── batch_1.npz
│   └── ...
├── estimators/
│   ├── monte_carlo/
│   │   ├── batch_0/
│   │   │   ├── estimator_final.pt    # Best model across all initializations
│   │   │   └── training_stats.json   # Summary statistics
│   │   ├── batch_1/
│   │   └── ...
│   └── dqn/
│       └── batch_0/
└── results/
    ├── evaluation_results.json
    └── *.png plots
```

### Reproducibility

**Seed Management:**
- Config `seed` controls policy training and initial RNG state
- Each data batch uses `seed + batch_idx` for diversity
- All NumPy and PyTorch seeds set consistently in each script

**To reproduce experiments exactly:**
1. Use identical config file
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
- Run name format: `{method}_{batch_name}_{num_episodes}ep_init{idx}` (one run per initialization)

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

Offline runs are stored in `experiments/<exp_id>/wandb_offline/` and automatically synced via `wandb sync` at the end of each training. If sync fails, you can manually sync later with:
```bash
wandb sync experiments/<exp_id>/wandb_offline/<run-directory>
```

## Common Patterns

### Creating a New Experiment

1. Copy and edit config: `cp configs/example_config.yaml configs/my_experiment.yaml`
2. Change `experiment_id` to avoid overwriting previous results
3. Adjust environment, policy algorithm, timesteps, etc.
4. Run pipeline: `./run_experiment.sh` (update CONFIG variable first)

### Debugging Training Issues

Check training stats:
```bash
cat experiments/<exp_id>/estimators/monte_carlo/batch_0/training_stats.json | jq '.[-1]'
```

Monitor convergence:
```bash
uv run python -c "import json; stats = json.load(open('training_stats.json')); print([s['eval_loss'] for s in stats])"
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

The framework checks for existing `estimator_episodes_*.pt` files and skips training if found.

To force retraining all models, use `--overwrite`:
```bash
uv run -m src.run_all_estimators --config config.yaml --mode sequential --overwrite
```

## Supported Environments

- **Classic Control**: CartPole-v1, Acrobot-v1, MountainCar-v0
- **Atari**: ALE/Pong-v5, ALE/Breakout-v5 (requires `gymnasium[atari,accept-rom-license]`)
- **MuJoCo**: HalfCheetah-v4, Hopper-v4, Walker2d-v4 (requires `gymnasium[mujoco]`)
- Any Gymnasium-compatible environment

## Key Dependencies

- **PyTorch**: Neural network implementation
- **Stable Baselines 3**: Policy training (PPO, A2C, SAC, TD3)
- **Gymnasium**: Environment API
- **NumPy**: Data storage and manipulation
- **Weights & Biases**: Experiment tracking
- **PyYAML**: Configuration management
