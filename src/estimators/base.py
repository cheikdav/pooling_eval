"""Base class for value estimators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from src.estimators.feature_extractors import FeatureExtractor


class ValueNetwork(nn.Module):
    """MLP for value function approximation."""

    def __init__(self, feature_dim: int, hidden_sizes: list, activation: str = "relu"):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_sizes = hidden_sizes

        layers = []
        prev_size = feature_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ValueEstimator(ABC):
    """Base class for all value estimators.

    Template method pattern: public methods extract features, abstract methods work with features.
    """

    def __init__(
        self,
        obs_dim: int,
        discount_factor: float,
        feature_extractor: FeatureExtractor,
        device: str = "auto"
    ):
        self.obs_dim = obs_dim
        self.discount_factor = discount_factor
        self.feature_extractor = feature_extractor

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.feature_extractor.to(self.device)
        self.training_step = 0

    @classmethod
    @abstractmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        pass

    @classmethod
    def from_config(cls, method_config, network_config, obs_dim: int, gamma: float):
        raise NotImplementedError("Subclasses must implement from_config")

    def train(self):
        """Set estimator to training mode.

        Note: Does NOT set feature_extractor to training mode.
        Feature extractor mode is managed separately for normalizer control.
        """
        pass

    def eval(self):
        """Set estimator to eval mode.

        Note: Does NOT set feature_extractor to eval mode.
        Feature extractor mode is managed separately for normalizer control.
        """
        pass

    def pre_training_pass(self, mini_batch: Dict[str, torch.Tensor]):
        """Pre-training pass over a batch of data for initialization.

        This should be called once before training on the full training dataset.
        It performs initialization tasks like updating normalizer statistics.

        Sets feature_extractor to training mode to enable normalizer updates.
        After all pre_training_pass calls, feature_extractor should be set to eval mode.

        Args:
            mini_batch: Dictionary containing observations
        """
        self.eval()  # Set estimator to eval mode
        self.feature_extractor.train()  # Set feature extractor to training mode for normalizer updates

        with torch.no_grad():
            obs = mini_batch['observations'].to(self.device)
            # Extract features to update normalizer (only from obs, not next_obs)
            self.feature_extractor(obs)
        self.feature_extractor.eval() 

    def train_step(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Extract features from observations then call _train_step()."""
        obs = mini_batch['observations'].to(self.device)
        next_obs = mini_batch['next_observations'].to(self.device)
        features = self.feature_extractor(obs)
        next_features = self.feature_extractor(next_obs)

        feature_batch = {
            'features': features,
            'next_features': next_features,
            'rewards': mini_batch['rewards'].to(self.device),
            'dones': mini_batch['dones'].to(self.device),
            'mc_returns': mini_batch['mc_returns'].to(self.device),
        }

        return self._train_step(feature_batch)

    @abstractmethod
    def _train_step(self, feature_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        pass

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Extract features from observations then call _predict()."""
        self.eval()

        with torch.no_grad():
            obs = torch.FloatTensor(observations).to(self.device)
            features = self.feature_extractor(obs)
            return self._predict(features)

    @abstractmethod
    def _predict(self, features: torch.Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def compute_targets(self, feature_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute target values for training.

        Args:
            feature_batch: Dictionary containing feature batch data

        Returns:
            Target values
        """
        pass

    def _build_checkpoint(self) -> Dict[str, Any]:
        """Build checkpoint dictionary. Subclasses override to add extra fields."""
        return {
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'feature_extractor_info': self.feature_extractor.get_save_info(),
            'training_step': self.training_step,
            'obs_dim': self.obs_dim,
            'discount_factor': self.discount_factor,
        }

    def save(self, path: Path):
        """Save estimator to disk."""
        checkpoint = self._build_checkpoint()
        torch.save(checkpoint, path)

    def _load_from_checkpoint_dict(self, checkpoint: Dict[str, Any]):
        """Load state from checkpoint dictionary. Subclasses override to load extra fields."""
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.training_step = checkpoint['training_step']

    def load(self, path: Path):
        """Load estimator from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self._load_from_checkpoint_dict(checkpoint)

    @classmethod
    @abstractmethod
    def load_from_checkpoint(cls, path: Path, device: str = "auto"):
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass


