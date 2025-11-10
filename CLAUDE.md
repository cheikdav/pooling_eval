# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research framework for evaluating value function estimation methods in reinforcement learning. The core workflow:

1. Train a policy using Stable Baselines 3 (PPO/A2C/SAC/TD3)
2. Generate batches of episodes using the trained policy
3. Train multiple value estimators (Monte Carlo, TD(λ), DQN) on the same data
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
python -m src.train_policy --config configs/example_config.yaml
python -m src.generate_data --config configs/example_config.yaml
python -m src.train_estimator --config configs/example_config.yaml --method monte_carlo --batch-idx 0
```

### Run All Estimators (Multiple Methods × Multiple Batches)
```bash
# Sequential (simple, good for debugging)
python -m src.run_all_estimators --config configs/example_config.yaml --mode sequential

# Parallel across GPUs
python -m src.run_all_estimators --config configs/example_config.yaml --mode parallel

# Or call the bash script directly
./run_parallel_estimators.sh configs/example_config.yaml monte_carlo,td_lambda,dqn 10
```

### Evaluate Results
```bash
python -m src.evaluate --config configs/example_config.yaml
# Generates: experiments/<exp_id>/results/evaluation_results.json + plots
```

### Disable Weights & Biases
```bash
python -m src.train_estimator ... --no-wandb
# Or set use_wandb: false in config
```

### Hyperparameter Tuning with W&B Sweeps
```bash
# Quick method: Use helper script
./launch_sweep.sh monte_carlo      # Run 1 agent
./launch_sweep.sh td_lambda 4      # Run 4 parallel agents

# Manual method:
# 1. Create sweep
wandb sweep configs/sweep_monte_carlo.yaml

# 2. Run agent(s) with the sweep ID returned above
wandb agent <sweep-id>
```

**How it works:**
- Sweep configs in `configs/sweep_*.yaml` define hyperparameter search space (currently: learning rate)
- Uses Bayesian optimization to find optimal values
- **Always uses `batch_tuning.npz`** for hyperparameter search
- Results saved to `experiments/<exp_id>/sweeps/<method>/<run_id>/`
- All runs tracked in W&B with eval/mse metric for comparison

**Customizing sweeps:**
- Edit `configs/sweep_*.yaml` to change search range or method (bayes/grid/random)
- To tune additional hyperparameters, modify `src/tune_hyperparameters.py`

## Architecture

### Configuration System (src/config.py)

All experiments are defined by a YAML config file that gets parsed into nested dataclasses:

- **ExperimentConfig**: Top-level container
  - **EnvironmentConfig**: Gymnasium environment name
  - **PolicyConfig**: SB3 algorithm and hyperparameters
  - **DataGenerationConfig**: Number of batches and episodes per batch
  - **ValueEstimatorsConfig**: Which methods to evaluate + training hyperparameters
    - **MonteCarloConfig**, **TDLambdaConfig**, **DQNConfig**: Method-specific parameters
  - **NetworkConfig**: Neural network architecture (hidden sizes, activation)
  - **LoggingConfig**: Weights & Biases settings

**Important quirk**: TD(λ) uses `lambda_` in Python (avoiding keyword) but `lambda` in YAML. The config system handles conversion automatically.

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

3. **Ground truth batch** (`batch_ground_truth.npz`):
   - Number of episodes controlled by `data_generation.ground_truth_episodes`
   - For training a high-quality reference estimator
   - Created only if `ground_truth_episodes > 0`

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

2. **TDLambdaEstimator** (td_lambda.py)
   - Uses eligibility traces for λ-returns combining n-step returns
   - Bootstraps from current value network predictions
   - Falls back to n-step TD if lambda=0 or lambda=1

3. **DQNEstimator** (dqn.py)
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
- Saves `estimator_best.pt` whenever evaluation loss improves
- Saves `estimator_final.pt` at the end of training
- Saves `training_stats.json` with per-epoch metrics

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
│   │   │   ├── estimator_final.pt
│   │   │   ├── estimator_best.pt
│   │   │   └── training_stats.json
│   │   ├── batch_1/
│   │   └── ...
│   ├── td_lambda/
│   │   └── batch_0/
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

### Device Management

- Device auto-detection: uses CUDA if available, else CPU
- Can override with `device='cpu'` or `device='cuda'`
- To force CPU: `export CUDA_VISIBLE_DEVICES=""`
- GPU array scripts distribute jobs across available GPUs automatically

### Wandb Integration

- All runs tagged with `experiment_id` for grouping
- Training metrics logged: `train/loss`, `train/mean_value`, `train/mean_target`
- Evaluation metrics logged: `eval/mse`, `eval/mae`
- Run name format: `{method}_batch_{idx}`

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
python -c "import json; stats = json.load(open('training_stats.json')); print([s['eval_loss'] for s in stats])"
```

### Resuming Interrupted Experiments

The framework automatically skips completed jobs. Just re-run:
```bash
python -m src.run_all_estimators --config config.yaml --mode sequential
```

It checks for `estimator_final.pt` and skips training if found.

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
