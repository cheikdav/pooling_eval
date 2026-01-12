"""Configuration management for experiments."""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class EnvironmentConfig:
    name: str


@dataclass
class PolicyConfig:
    algorithm: str
    total_timesteps: int
    learning_rate: float
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_envs: int = 1
    ent_coef: float = 0.0  # Entropy coefficient for exploration
    clip_range: float = 0.2  # Clipping parameter for PPO
    max_grad_norm: float = 0.5  # Max gradient norm for gradient clipping
    vf_coef: float = 0.5  # Value function coefficient
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    use_vec_normalize: bool = False
    normalize_obs: bool = True
    normalize_reward: bool = True
    vec_normalize_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataGenerationConfig:
    n_batches: int
    episodes_per_batch: int
    deterministic_policy: bool = False
    n_envs: int = 1
    tuning_episodes: int = 0
    ground_truth_episodes: int = 0
    eval_episodes: int = 0


@dataclass
class ValueEstimatorTrainingConfig:
    max_epochs: int = 1000
    batch_size: int = 256
    convergence_patience: int = 50
    convergence_threshold: float = 1e-4
    eval_frequency: int = 10
    gamma: float = 0.99
    episode_subsets: Optional[List[int]] = None
    shuffle_frequency: int = 100  # Re-shuffle DataLoader every N epochs (0 = never, 1 = every epoch)
    test_episodes: int = 0  # Number of episodes to sample from eval batch for test set (0 = no test set)


class EstimatorType(str, Enum):
    """Enum for estimator types."""
    MONTE_CARLO = "monte_carlo"
    DQN = "dqn"
    LEAST_SQUARES_MC = "least_squares_mc"
    LEAST_SQUARES_TD = "least_squares_td"


@dataclass
class BaseEstimatorConfig:
    """Base configuration with parameters common to all estimators."""
    name: str  # Name for this method config (used in output paths, logs, etc.)
    type: EstimatorType  # Method type
    learning_rate: float = 0.001
    n_initializations: int = 1  # Number of random initializations to try
    max_epochs: Optional[int] = None  # Override global max_epochs if set


@dataclass
class MonteCarloConfig(BaseEstimatorConfig):
    """Monte Carlo estimator configuration (only uses base parameters)."""
    pass


@dataclass
class DQNConfig(BaseEstimatorConfig):
    """DQN estimator configuration."""
    target_update_rate: float = 1.0e-5


@dataclass
class LeastSquaresMCConfig(BaseEstimatorConfig):
    """Least Squares Monte Carlo estimator configuration."""
    policy_path: str = None  # Path to trained policy (.zip file), auto-set if None
    algorithm: str = "PPO"  # Policy algorithm (PPO, A2C, SAC, TD3)
    ridge_lambda: float = 1e-6  # Ridge regularization parameter


@dataclass
class LeastSquaresTDConfig(BaseEstimatorConfig):
    """Least Squares Temporal Difference estimator configuration."""
    policy_path: str = None  # Path to trained policy (.zip file), auto-set if None
    algorithm: str = "PPO"  # Policy algorithm (PPO, A2C, SAC, TD3)
    ridge_lambda: float = 1e-6  # Ridge regularization parameter


# Registry mapping EstimatorType to config class
ESTIMATOR_CONFIG_REGISTRY: Dict[EstimatorType, Type[BaseEstimatorConfig]] = {
    EstimatorType.MONTE_CARLO: MonteCarloConfig,
    EstimatorType.DQN: DQNConfig,
    EstimatorType.LEAST_SQUARES_MC: LeastSquaresMCConfig,
    EstimatorType.LEAST_SQUARES_TD: LeastSquaresTDConfig,
}


@dataclass
class ValueEstimatorsConfig:
    training: ValueEstimatorTrainingConfig
    method_configs: List[BaseEstimatorConfig] = field(default_factory=list)


@dataclass
class NetworkConfig:
    hidden_sizes: List[int]
    activation: str = "relu"
    device: str = "auto"  # "cpu" or "auto" (GPU if available, else CPU)


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    wandb_project: str = "pooling-eval"
    wandb_entity: Optional[str] = None
    log_frequency: int = 10
    wandb_mode: str = "online"  # "online" (sync immediately) or "offline" (sync at end)


@dataclass
class ExperimentConfig:
    experiment_id: str
    seed: int
    environment: EnvironmentConfig
    policy: PolicyConfig
    data_generation: DataGenerationConfig
    value_estimators: ValueEstimatorsConfig
    network: NetworkConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        # Parse nested configs
        env_config = EnvironmentConfig(**config_dict['environment'])
        policy_config = PolicyConfig(**config_dict['policy'])
        data_gen_config = DataGenerationConfig(**config_dict['data_generation'])
        network_config = NetworkConfig(**config_dict['network'])
        logging_config = LoggingConfig(**config_dict['logging'])

        # Parse value estimators config
        ve_dict = config_dict['value_estimators']
        training_config = ValueEstimatorTrainingConfig(**ve_dict['training'])

        # Parse method configs using registry
        method_configs = []
        if 'method_configs' in ve_dict:
            for method_dict in ve_dict['method_configs']:
                # Get the type and look up the config class
                estimator_type = EstimatorType(method_dict['type'])
                config_class = ESTIMATOR_CONFIG_REGISTRY[estimator_type]

                # Create the config instance
                method_config = config_class(**method_dict)
                method_configs.append(method_config)

        value_estimators_config = ValueEstimatorsConfig(
            training=training_config,
            method_configs=method_configs
        )

        return cls(
            experiment_id=config_dict['experiment_id'],
            seed=config_dict['seed'],
            environment=env_config,
            policy=policy_config,
            data_generation=data_gen_config,
            value_estimators=value_estimators_config,
            network=network_config,
            logging=logging_config
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                # Filter out None values
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items() if v is not None}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, Enum):
                # Convert enum to its value
                return obj.value
            else:
                return obj

        result = dataclass_to_dict(self)
        return result

    def save(self, path: Path):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
