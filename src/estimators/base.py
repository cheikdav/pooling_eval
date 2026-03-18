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
        self.mean_reward = 0.0
        self._reward_sum = 0.0
        self._reward_count = 0
        self._n_terminated = 0

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.feature_extractor.to(self.device)
        self.training_step = 0

    @property
    def reward_offset(self) -> float:
        """Offset to add to centered predictions to recover true values: mean_reward / (1 - gamma)."""
        return self.mean_reward / (1 - self.discount_factor) if self.mean_reward != 0.0 else 0.0

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
        It performs initialization tasks like updating normalizer statistics
        and accumulating reward statistics for reward centering.

        Sets feature_extractor to training mode to enable normalizer updates.
        After all pre_training_pass calls, feature_extractor should be set to eval mode.

        Args:
            mini_batch: Dictionary containing observations and rewards
        """
        self.eval()  # Set estimator to eval mode
        self.feature_extractor.train()  # Set feature extractor to training mode for normalizer updates

        with torch.no_grad():
            obs = mini_batch['observations'].to(self.device)
            # Extract features to update normalizer (only from obs, not next_obs)
            self.feature_extractor(obs)
        self.feature_extractor.eval()

        # Accumulate reward statistics for reward centering
        rewards = mini_batch['rewards']
        self._reward_sum += rewards.sum().item()
        self._reward_count += len(rewards)
        self._n_terminated += mini_batch['dones'].sum().item()

    def finalize_pre_training(self, reward_centering: bool = False):
        """Finalize pre-training: compute mean_reward from accumulated stats.

        Corrects for episode terminations: each terminated episode implies an infinite
        tail of zero rewards, so we add n_terminated/(1-gamma) phantom steps to the
        denominator to avoid overestimating r_bar.

        Args:
            reward_centering: If True, set mean_reward from accumulated reward statistics.
        """
        if reward_centering and self._reward_count > 0:
            effective_count = self._reward_count + self._n_terminated / (1 - self.discount_factor)
            self.mean_reward = self._reward_sum / effective_count

    def cache_features_in_dataset(self, dataset, batch_size: int = 1024):
        """Extract and cache features for entire dataset to avoid recomputation.

        Should be called after pre_training_pass to cache features with finalized normalizer.

        Args:
            dataset: TransitionDataset instance to cache features in
            batch_size: Batch size for feature extraction (default 1024)
        """
        from torch.utils.data import DataLoader

        self.eval()
        self.feature_extractor.eval()

        n_samples = len(dataset)
        feature_dim = self.feature_extractor.get_feature_dim()

        # Pre-allocate tensors for all features
        all_features = torch.zeros(n_samples, feature_dim)
        all_next_features = torch.zeros(n_samples, feature_dim)

        # Create dataloader without shuffling
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            start_idx = 0
            for mini_batch in dataloader:
                batch_len = len(mini_batch['observations'])
                end_idx = start_idx + batch_len

                obs = mini_batch['observations'].to(self.device)
                next_obs = mini_batch['next_observations'].to(self.device)

                features = self.feature_extractor(obs).cpu()
                next_features = self.feature_extractor(next_obs).cpu()

                all_features[start_idx:end_idx] = features
                all_next_features[start_idx:end_idx] = next_features

                start_idx = end_idx

        dataset.set_features(all_features, all_next_features) 

    def train_step(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Extract features from observations then call _train_step()."""
        # Use cached features if available, otherwise extract on-the-fly
        if 'features' in mini_batch and 'next_features' in mini_batch:
            features = mini_batch['features'].to(self.device)
            next_features = mini_batch['next_features'].to(self.device)
        else:
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

    def predict(self, observations: np.ndarray, features: torch.Tensor = None) -> np.ndarray:
        """Extract features from observations then call _predict().

        Adds reward_offset to convert centered predictions back to true values.

        Args:
            observations: Observation array
            features: Optional pre-computed features. If provided, skips feature extraction.

        Returns:
            Predicted values (uncentered, i.e. true value estimates)
        """
        self.eval()

        with torch.no_grad():
            if features is None:
                # Handle object dtype arrays (can occur with certain env data)
                if observations.dtype == np.object_:
                    observations = observations.astype(np.float32)
                obs = torch.FloatTensor(observations).to(self.device)
                features = self.feature_extractor(obs)
            else:
                features = features.to(self.device)
            return self._predict(features) + self.reward_offset

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
            'mean_reward': self.mean_reward,
        }

    def save(self, path: Path):
        """Save estimator to disk."""
        checkpoint = self._build_checkpoint()
        torch.save(checkpoint, path)

    def _load_from_checkpoint_dict(self, checkpoint: Dict[str, Any]):
        """Load state from checkpoint dictionary. Subclasses override to load extra fields."""
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.training_step = checkpoint['training_step']
        self.mean_reward = checkpoint.get('mean_reward', 0.0)

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


