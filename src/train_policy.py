"""Train a policy using Stable Baselines 3."""

import argparse
import json
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch
import wandb

from src.config import ExperimentConfig
from src.env_utils import ALGORITHM_MAP, create_vec_env

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
        if self.n_calls % self.log_frequency == 0:
            log_dict = {'timesteps': self.num_timesteps}

            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer
                if len(ep_info) > 0:
                    # Note: Monitor tracks raw (unnormalized) returns before VecNormalize,
                    # so ep_returns are already the true environment returns
                    ep_returns = [info['r'] for info in ep_info]
                    ep_lengths = [info['l'] for info in ep_info]

                    log_dict.update({
                        'episode/mean_reward': np.mean(ep_returns),
                        'episode/mean_length': np.mean(ep_lengths),
                        'episode/min_reward': np.min(ep_returns),
                        'episode/max_reward': np.max(ep_returns),
                    })

            if hasattr(self.model, 'logger') and self.model.logger is not None:
                try:
                    metric_keys = ['train/value_loss', 'train/explained_variance',
                                   'train/policy_loss', 'train/entropy_loss', 'train/loss',
                                   'train/policy_gradient_loss', 'train/clip_fraction',
                                   'train/approx_kl', 'train/clip_range', 'train/learning_rate']

                    for key in metric_keys:
                        if key in self.model.logger.name_to_value:
                            log_dict[key] = self.model.logger.name_to_value[key]

                except Exception:
                    pass

            if len(log_dict) > 1:
                wandb.log(log_dict)

        return True


def verify_saved_policy(model, model_path, AlgorithmClass, n_samples=5):
    """Load saved policy and verify weights and outputs match the original."""
    print("Verifying saved policy...")
    loaded = AlgorithmClass.load(model_path)

    orig_state = model.policy.state_dict()
    loaded_state = loaded.policy.state_dict()
    weights_match = set(orig_state.keys()) == set(loaded_state.keys()) and all(
        torch.allclose(orig_state[k], loaded_state[k]) for k in orig_state
    )

    obs = np.array([model.observation_space.sample() for _ in range(n_samples)])
    with torch.no_grad():
        orig_actions, _ = model.predict(obs, deterministic=True)
        loaded_actions, _ = loaded.predict(obs, deterministic=True)
    outputs_match = np.allclose(orig_actions, loaded_actions)

    if weights_match and outputs_match:
        print("Policy verification passed: weights and outputs match")
    else:
        if not weights_match:
            print("Policy verification FAILED: weights differ!")
        if not outputs_match:
            print(f"Policy verification FAILED: outputs differ!\n  orig={orig_actions}\n  loaded={loaded_actions}")


def train_policy(config: ExperimentConfig, output_dir: Path, use_wandb: bool = True):
    """Train a policy using Stable Baselines 3.

    Args:
        config: Experiment configuration
        output_dir: Directory to save the trained policy
        use_wandb: Whether to use Weights & Biases logging
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb and config.logging.use_wandb:
        wandb.init(
            project=config.logging.get_project_name(config.environment.name),
            entity=config.logging.wandb_entity,
            name=f"policy_{config.policy.algorithm} ({config.environment.name})",
            group=config.experiment_id,
            mode=config.logging.wandb_mode,
            dir=str(config.get_policy_dir() / "wandb_offline") if config.logging.wandb_mode == "offline" else None,
            config={
                'algorithm': config.policy.algorithm,
                'environment': config.environment.name,
                'total_timesteps': config.policy.total_timesteps,
                'learning_rate': config.policy.learning_rate,
                'gamma': config.policy.gamma,
                'seed': config.policy.seed,
                'experiment_id': config.experiment_id,
                **config.policy.__dict__,
            },
            tags=['policy_training', config.policy.algorithm, config.environment.name],
        )

    np.random.seed(config.policy.seed)
    torch.manual_seed(config.policy.seed)

    env, _ = create_vec_env(config, n_envs=config.policy.n_envs, use_monitor=True, seed=config.policy.seed)

    if config.policy.algorithm not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {config.policy.algorithm}. "
                        f"Available: {list(ALGORITHM_MAP.keys())}")

    AlgorithmClass = ALGORITHM_MAP[config.policy.algorithm]

    if len(env.observation_space.shape) == 3:
        policy_type = "CnnPolicy"
    else:
        policy_type = "MlpPolicy"

    algo_kwargs = {
        "policy": policy_type,
        "env": env,
        "learning_rate": config.policy.learning_rate,
        "gamma": config.policy.gamma,
        "verbose": 2,
        "seed": config.policy.seed,
    }

    if config.policy.algorithm in ["PPO", "A2C"]:
        algo_kwargs.update({
            "n_steps": config.policy.n_steps,
            "batch_size": config.policy.batch_size,
            "n_epochs": config.policy.n_epochs,
            "gae_lambda": config.policy.gae_lambda,
            "ent_coef": config.policy.ent_coef,
            "vf_coef": config.policy.vf_coef,
            "max_grad_norm": config.policy.max_grad_norm,
        })

    if config.policy.algorithm == "PPO":
        algo_kwargs.update({
            "clip_range": config.policy.clip_range,
        })

    if config.policy.policy_kwargs:
        policy_kwargs = config.policy.policy_kwargs.copy()

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

    if config.policy.kwargs:
        algo_kwargs.update(config.policy.kwargs)

    model = AlgorithmClass(**algo_kwargs)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.policy.total_timesteps // 10, 1000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="policy"
    )

    callbacks = [checkpoint_callback]

    if use_wandb and config.logging.use_wandb:
        wandb_log_freq = max(100, config.policy.total_timesteps // 1000)
        wandb_callback = WandbCallback(log_frequency=wandb_log_freq)
        callbacks.append(wandb_callback)

    print(f"\nTraining {config.policy.algorithm} on {config.environment.name}...")
    print(f"Total timesteps: {config.policy.total_timesteps}")
    print(f"Output directory: {output_dir}")
    if use_wandb and config.logging.use_wandb:
        print(f"W&B tracking enabled: {config.logging.get_project_name(config.environment.name)}\n")
    else:
        print("W&B tracking disabled\n")

    if config.policy.algorithm in ["PPO", "A2C"]:
        total_iterations = config.policy.total_timesteps / config.policy.n_steps
        log_interval = max(1, int(total_iterations / 15))
    else:
        total_iterations = config.policy.total_timesteps / 1000
        log_interval = max(1, int(total_iterations / 15))

    model.learn(
        total_timesteps=config.policy.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=log_interval
    )

    model_path = output_dir / "policy_final.zip"
    model.save(model_path)
    print(f"\nPolicy saved to {model_path}")
    verify_saved_policy(model, model_path, AlgorithmClass)

    if config.policy.use_vec_normalize:
        vec_normalize_path = output_dir / "vec_normalize.pkl"
        env.save(vec_normalize_path)
        print(f"VecNormalize stats saved to {vec_normalize_path}")

    avg_reward = None
    if len(model.ep_info_buffer) > 0:
        ep_returns = [info['r'] for info in model.ep_info_buffer]
        avg_reward = float(np.mean(ep_returns))

    policy_metadata = {
        'algorithm': config.policy.algorithm,
        'environment': config.environment.name,
        'total_timesteps': config.policy.total_timesteps,
        'learning_rate': config.policy.learning_rate,
        'gamma': config.policy.gamma,
        'seed': config.policy.seed,
        'average_reward': avg_reward,
        'n_envs': config.policy.n_envs,
        'use_vec_normalize': config.policy.use_vec_normalize,
    }

    if config.policy.algorithm in ["PPO", "A2C"]:
        policy_metadata.update({
            'n_steps': config.policy.n_steps,
            'batch_size': config.policy.batch_size,
            'n_epochs': config.policy.n_epochs,
            'gae_lambda': config.policy.gae_lambda,
        })

    if config.policy.policy_kwargs:
        policy_metadata['policy_kwargs'] = {
            k: (v if not isinstance(v, type) else v.__name__)
            for k, v in config.policy.policy_kwargs.items()
        }

    metadata_path = output_dir / "policy_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(policy_metadata, f, indent=2)
    print(f"Policy metadata saved to {metadata_path}")

    try:
        # Extract hidden sizes from the critic network
        # For PPO/A2C: hidden layers are in mlp_extractor.value_net, not value_net (which is just the output head)
        hidden_sizes = None
        if hasattr(model.policy, 'mlp_extractor') and hasattr(model.policy.mlp_extractor, 'value_net'):
            hidden_sizes = [layer.out_features for layer in model.policy.mlp_extractor.value_net
                            if isinstance(layer, torch.nn.Linear)]
        elif hasattr(model.policy, 'q_net'):
            hidden_sizes = [layer.out_features for layer in model.policy.q_net
                            if isinstance(layer, torch.nn.Linear)]

        if not hidden_sizes:
            hidden_sizes = config.network.hidden_sizes

        np.savez(
            output_dir / "critic_architecture.npz",
            hidden_sizes=hidden_sizes
        )
    except Exception as e:
        print(f"Warning: Could not save critic architecture info: {e}")

    env.close()

    if use_wandb and config.logging.use_wandb:
        # Sync offline run if in offline mode
        if config.logging.wandb_mode == "offline":
            print(f"\nSyncing offline run to W&B...")
            import subprocess
            try:
                result = subprocess.run(["wandb", "sync", wandb.run.dir], check=True, capture_output=True, text=True)
                print(f"✓ Successfully synced to W&B")
            except subprocess.CalledProcessError as e:
                print(f"✗ Warning: Failed to sync offline run")
                print(f"Error: {e}")
                if e.stdout:
                    print(f"stdout: {e.stdout}")
                if e.stderr:
                    print(f"stderr: {e.stderr}")
                print(f"You can manually sync later with: wandb sync {wandb.run.dir}")

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
        output_dir = config.get_policy_dir()
    else:
        output_dir = args.output_dir

    # Save config to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    # Train policy
    train_policy(config, output_dir, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
