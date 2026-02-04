"""Feature extractors for value estimators."""

from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, A2C, SAC, TD3


class RunningNormalizer:
    """Tracks running mean and std for online normalization."""

    def __init__(self, dim: int, epsilon: float = 1e-8):
        self.dim = dim
        self.epsilon = epsilon
        self.mean = torch.zeros(dim)
        self.var = torch.ones(dim)
        self.count = 0

    def update(self, data: torch.Tensor) -> None:
        batch_mean = data.mean(dim=0)
        batch_var = data.var(dim=0)
        batch_count = data.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) / torch.sqrt(self.var + self.epsilon)

    def state_dict(self):
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count,
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']


class FeatureExtractor(nn.Module, ABC):
    """Abstract base class for feature extraction.

    Template method pattern: forward() handles normalization, _forward() extracts features.
    Normalization statistics only update during training mode.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

        if self.normalize:
            self.normalizer = RunningNormalizer(self.get_feature_dim())
        else:
            self.normalizer = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self._forward(observations)

        if self.normalize:
            if self.training:
                self.normalizer.update(features)
            features = self.normalizer.normalize(features)

        return features

    @abstractmethod
    def _forward(self, observations: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        pass


class IdentityExtractor(FeatureExtractor):
    """Returns observations as features."""

    def __init__(self, obs_dim: int, normalize: bool = True):
        self.obs_dim = obs_dim
        super().__init__(normalize=normalize)

    def _forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations

    def get_feature_dim(self) -> int:
        return self.obs_dim


class PolicyRepresentationExtractor(FeatureExtractor):
    """Extracts frozen intermediate representations from a trained policy network."""

    def __init__(self, policy_path: str, algorithm: str, device: str = "auto", normalize: bool = True):
        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        self.policy_path = policy_path
        self.algorithm = algorithm.lower()
        self.device = device_obj

        policy_path_obj = Path(policy_path)
        if not policy_path_obj.exists():
            raise FileNotFoundError(f"Policy not found at {policy_path}")

        policy_classes = {
            'ppo': PPO,
            'a2c': A2C,
            'sac': SAC,
            'td3': TD3,
        }

        if self.algorithm not in policy_classes:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        policy = policy_classes[self.algorithm].load(policy_path, device=device_obj)
        self.policy = policy

        with torch.no_grad():
            dummy_obs = torch.zeros(1, policy.observation_space.shape[0], device=device_obj)
            if self.algorithm in ['ppo', 'a2c']:
                dummy_features = policy.policy.mlp_extractor.forward_actor(dummy_obs)
            elif self.algorithm in ['sac', 'td3']:
                dummy_features = policy.critic.features_extractor(dummy_obs)
            else:
                raise ValueError(f"Unsupported algorithm for representation extraction: {self.algorithm}")
            self.repr_dim = dummy_features.shape[-1]

        super().__init__(normalize=normalize)

        if self.algorithm in ['ppo', 'a2c']:
            self.repr_net = policy.policy.mlp_extractor.policy_net
        elif self.algorithm in ['sac', 'td3']:
            self.repr_net = policy.critic.features_extractor

        for param in self.repr_net.parameters():
            param.requires_grad = False
        self.repr_net.eval()

    def _forward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.algorithm in ['ppo', 'a2c']:
                features = self.policy.policy.mlp_extractor.forward_actor(observations)
            elif self.algorithm in ['sac', 'td3']:
                features = self.repr_net(observations)
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        return features

    def get_feature_dim(self) -> int:
        return self.repr_dim
