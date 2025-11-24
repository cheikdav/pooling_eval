"""Train a policy using Stable Baselines 3."""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch
import wandb

from src.config import ExperimentConfig
from src.env_utils import create_vec_env


ALGORITHM_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}

# Mapping from string names to PyTorch activation functions
ACTIVATION_FN_MAP = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
    "leaky_relu": torch.nn.LeakyReLU,
    "silu": torch.nn.SiLU,
    "gelu": torch.nn.GELU,
}


class WandbCallback(BaseCallback):
    """Custom callback for logging training metrics to Weights & Biases."""

    def __init__(self, log_frequency: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.log_frequency = log_frequency

    def _on_step(self) -> bool:
        """Called after each step."""
        # Log every N calls (since this is called for each env step)
        if self.n_calls % self.log_frequency == 0:
            log_dict = {'timesteps': self.num_timesteps}

            # Collect info from the training rollout
            if len(self.model.ep_info_buffer) > 0:
                # Get episode statistics (returns, lengths)
                ep_info = self.model.ep_info_buffer
                if len(ep_info) > 0:
                    ep_returns = [info['r'] for info in ep_info]
                    ep_lengths = [info['l'] for info in ep_info]

                    # Check if environment is wrapped with VecNormalize
                    from stable_baselines3.common.vec_env import VecNormalize
                    env = self.training_env
                    is_normalized = isinstance(env, VecNormalize)

                    log_dict.update({
                        'episode/mean_reward': np.mean(ep_returns),
                        'episode/mean_length': np.mean(ep_lengths),
                        'episode/min_reward': np.min(ep_returns),
                        'episode/max_reward': np.max(ep_returns),
                    })

                    # If using VecNormalize, also log unnormalized rewards
                    if is_normalized and env.norm_reward:
                        unnormalized_returns = [env.unnormalize_reward(r) for r in ep_returns]
                        log_dict.update({
                            'episode/mean_reward_unnormalized': np.mean(unnormalized_returns),
                            'episode/min_reward_unnormalized': np.min(unnormalized_returns),
                            'episode/max_reward_unnormalized': np.max(unnormalized_returns),
                        })

            # Log algorithm-specific metrics
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # Try to get the loss values from the logger
                try:
                    # Get all metrics from SB3's logger and use them directly
                    # (they already have proper prefixes like 'train/')
                    metric_keys = ['train/value_loss', 'train/explained_variance',
                                   'train/policy_loss', 'train/entropy_loss', 'train/loss',
                                   'train/policy_gradient_loss', 'train/clip_fraction',
                                   'train/approx_kl', 'train/clip_range', 'train/learning_rate']

                    for key in metric_keys:
                        if key in self.model.logger.name_to_value:
                            # Use the key as-is (already has train/ prefix)
                            log_dict[key] = self.model.logger.name_to_value[key]

                except Exception:
                    pass

            # Only log if we have data
            if len(log_dict) > 1:  # More than just timesteps
                wandb.log(log_dict)

        return True


def train_policy(config: ExperimentConfig, output_dir: Path, use_wandb: bool = True):
    """Train a policy using Stable Baselines 3.

    Args:
        config: Experiment configuration
        output_dir: Directory to save the trained policy
        use_wandb: Whether to use Weights & Biases logging
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Weights & Biases
    if use_wandb and config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=f"policy_{config.policy.algorithm}",
            group=config.experiment_id,
            config={
                'algorithm': config.policy.algorithm,
                'environment': config.environment.name,
                'total_timesteps': config.policy.total_timesteps,
                'learning_rate': config.policy.learning_rate,
                'gamma': config.policy.gamma,
                'seed': config.seed,
                'experiment_id': config.experiment_id,
                **config.policy.__dict__,
            },
            tags=['policy_training', config.policy.algorithm, config.environment.name],
        )

    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create environment
    env, _ = create_vec_env(config, n_envs=config.policy.n_envs, use_monitor=True)

    # Get algorithm class
    if config.policy.algorithm not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {config.policy.algorithm}. "
                        f"Available: {list(ALGORITHM_MAP.keys())}")

    AlgorithmClass = ALGORITHM_MAP[config.policy.algorithm]

    # Prepare algorithm kwargs
    algo_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": config.policy.learning_rate,
        "gamma": config.policy.gamma,
        "verbose": 2,
        "seed": config.seed,
    }

    # Add algorithm-specific parameters
    if config.policy.algorithm in ["PPO", "A2C"]:
        algo_kwargs.update({
            "n_steps": config.policy.n_steps,
            "batch_size": config.policy.batch_size,
            "n_epochs": config.policy.n_epochs,
            "gae_lambda": config.policy.gae_lambda,
        })

    # Add policy_kwargs (network architecture, etc.)
    if config.policy.policy_kwargs:
        policy_kwargs = config.policy.policy_kwargs.copy()

        # Convert activation_fn string to actual PyTorch class
        if "activation_fn" in policy_kwargs:
            activation_fn = policy_kwargs["activation_fn"]
            if isinstance(activation_fn, str):
                activation_fn_lower = activation_fn.lower()
                if activation_fn_lower in ACTIVATION_FN_MAP:
                    policy_kwargs["activation_fn"] = ACTIVATION_FN_MAP[activation_fn_lower]
                else:
                    raise ValueError(
                        f"Unknown activation function: {activation_fn}. "
                        f"Available: {list(ACTIVATION_FN_MAP.keys())}"
                    )

        algo_kwargs["policy_kwargs"] = policy_kwargs

    # Add any additional kwargs from config
    if config.policy.kwargs:
        algo_kwargs.update(config.policy.kwargs)

    # Create model
    model = AlgorithmClass(**algo_kwargs)

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.policy.total_timesteps // 10, 1000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="policy"
    )

    # Setup callbacks list
    callbacks = [checkpoint_callback]

    # Add wandb callback if enabled
    if use_wandb and config.logging.use_wandb:
        # Log every N steps (adjust based on total timesteps for reasonable frequency)
        wandb_log_freq = max(100, config.policy.total_timesteps // 1000)
        wandb_callback = WandbCallback(log_frequency=wandb_log_freq)
        callbacks.append(wandb_callback)

    print(f"\nTraining {config.policy.algorithm} on {config.environment.name}...")
    print(f"Total timesteps: {config.policy.total_timesteps}")
    print(f"Output directory: {output_dir}")
    if use_wandb and config.logging.use_wandb:
        print(f"W&B tracking enabled: {config.logging.wandb_project}\n")
    else:
        print("W&B tracking disabled\n")

    # Calculate log_interval based on expected iterations
    # For on-policy algorithms (PPO/A2C), iterations = timesteps / n_steps
    # We want to log ~10-15 times during training
    if config.policy.algorithm in ["PPO", "A2C"]:
        total_iterations = config.policy.total_timesteps / config.policy.n_steps
        log_interval = max(1, int(total_iterations / 15))
    else:
        # For off-policy algorithms, estimate based on timesteps
        # Assume ~1000 timesteps per iteration (conservative estimate)
        total_iterations = config.policy.total_timesteps / 1000
        log_interval = max(1, int(total_iterations / 15))

    # Train the model
    model.learn(
        total_timesteps=config.policy.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=log_interval
    )

    # Save final model
    model_path = output_dir / "policy_final.zip"
    model.save(model_path)
    print(f"\nPolicy saved to {model_path}")

    # Save VecNormalize statistics if used
    if config.policy.use_vec_normalize:
        vec_normalize_path = output_dir / "vec_normalize.pkl"
        env.save(vec_normalize_path)
        print(f"VecNormalize stats saved to {vec_normalize_path}")

    # Save the critic network architecture info for later use
    try:
        if hasattr(model.policy, 'value_net'):
            critic_info = {
                'hidden_sizes': [layer.out_features for layer in model.policy.value_net
                            if hasattr(layer, 'out_features')],
            }
        elif hasattr(model.policy, 'q_net'):
            critic_info = {
                'hidden_sizes': [layer.out_features for layer in model.policy.q_net
                            if hasattr(layer, 'out_features')],
            }
        else:
            critic_info = {'hidden_sizes': config.network.hidden_sizes}

        np.savez(
            output_dir / "critic_architecture.npz",
            **critic_info
        )
    except Exception as e:
        print(f"Warning: Could not save critic architecture info: {e}")

    env.close()

    # Finish wandb run
    if use_wandb and config.logging.use_wandb:
        wandb.finish()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train a policy using Stable Baselines 3")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (default: experiments/<experiment_id>/policy)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    args = parser.parse_args()

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)

    # Set output directory
    if args.output_dir is None:
        output_dir = Path("experiments") / config.experiment_id / "policy"
    else:
        output_dir = args.output_dir

    # Save config to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Train policy
    train_policy(config, output_dir, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
