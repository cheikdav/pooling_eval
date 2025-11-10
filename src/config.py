"""Configuration management for experiments."""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


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


@dataclass
class DataGenerationConfig:
    n_batches: int
    episodes_per_batch: int
    deterministic_policy: bool = False
    # Special batches for different purposes
    tuning_episodes: int = 0  # For hyperparameter tuning
    ground_truth_episodes: int = 0  # For training ground truth estimator
    eval_episodes: int = 0  # For final evaluation


@dataclass
class ValueEstimatorTrainingConfig:
    max_epochs: int = 1000
    batch_size: int = 256
    convergence_patience: int = 50
    convergence_threshold: float = 1e-4
    eval_frequency: int = 10


@dataclass
class BaseEstimatorConfig:
    """Base configuration with parameters common to all estimators."""
    discount_factor: float = 0.99
    learning_rate: float = 0.001
    n_initializations: int = 1  # Number of random initializations to try


@dataclass
class MonteCarloConfig(BaseEstimatorConfig):
    """Monte Carlo estimator configuration (only uses base parameters)."""
    pass


@dataclass
class TDLambdaConfig(BaseEstimatorConfig):
    """TD(λ) estimator configuration."""
    lambda_: float = 0.95  # using lambda_ to avoid Python keyword
    n_step: int = 1


@dataclass
class DQNConfig(BaseEstimatorConfig):
    """DQN estimator configuration."""
    target_update_rate: float = 1.0e-5
    double_dqn: bool = True


@dataclass
class ValueEstimatorsConfig:
    methods: List[str]
    training: ValueEstimatorTrainingConfig
    monte_carlo: Optional[MonteCarloConfig] = None
    td_lambda: Optional[TDLambdaConfig] = None
    dqn: Optional[DQNConfig] = None


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
        print(yaml.__version__)
        print(config_dict['value_estimators']['training'])
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

        # Only create configs for methods that are specified in the YAML
        monte_carlo_config = None
        if 'monte_carlo' in ve_dict:
            monte_carlo_config = MonteCarloConfig(**ve_dict['monte_carlo'])

        td_lambda_config = None
        if 'td_lambda' in ve_dict:
            # Handle lambda -> lambda_ conversion for TD
            td_params = ve_dict['td_lambda'].copy()
            if 'lambda' in td_params:
                td_params['lambda_'] = td_params.pop('lambda')
            td_lambda_config = TDLambdaConfig(**td_params)

        dqn_config = None
        if 'dqn' in ve_dict:
            dqn_config = DQNConfig(**ve_dict['dqn'])

        value_estimators_config = ValueEstimatorsConfig(
            methods=ve_dict['methods'],
            training=training_config,
            monte_carlo=monte_carlo_config,
            td_lambda=td_lambda_config,
            dqn=dqn_config
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
            else:
                return obj

        result = dataclass_to_dict(self)
        # Convert lambda_ back to lambda for saving
        if 'value_estimators' in result and 'td_lambda' in result['value_estimators']:
            if 'lambda_' in result['value_estimators']['td_lambda']:
                result['value_estimators']['td_lambda']['lambda'] = \
                    result['value_estimators']['td_lambda'].pop('lambda_')
        return result

    def save(self, path: Path):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
