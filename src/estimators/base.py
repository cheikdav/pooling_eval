"""Lightning-based value estimator base."""

from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import copy

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.estimators.feature_extractors import (
    FeatureExtractor,
    create_feature_extractor,
    create_feature_extractor_from_saved_info,
)


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


class ValueEstimator(pl.LightningModule):
    """Lightning base class for all value estimators.

    Subclasses override `compute_targets(batch)` — the algorithm-specific hook.
    Everything else (training loop, checkpointing, logging) is handled by Lightning.
    """

    def __init__(
        self,
        obs_dim: int,
        discount_factor: float,
        feature_extractor_save_info: dict,
        device_str: str = "auto",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.discount_factor = discount_factor
        self.device_str = device_str

        self.feature_extractor = create_feature_extractor_from_saved_info(
            feature_extractor_save_info, device=device_str
        )

        self.mean_reward = 0.0
        self._reward_sum = 0.0
        self._reward_count = 0
        self._n_terminated = 0
        self._reward_centering = False
        self._features_cached = False

    @property
    def reward_offset(self) -> float:
        return self.mean_reward / (1 - self.discount_factor) if self.mean_reward != 0.0 else 0.0

    @abstractmethod
    def compute_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ...

    def _get_features(self, batch):
        if 'features' in batch and 'next_features' in batch:
            return batch['features'], batch['next_features']
        obs = batch['observations']
        next_obs = batch['next_observations']
        return self.feature_extractor(obs), self.feature_extractor(next_obs)

    def predict(self, observations: np.ndarray, features: Optional[torch.Tensor] = None) -> np.ndarray:
        """Predict values for an array of observations (returns uncentered values)."""
        self.eval()
        with torch.no_grad():
            if features is None:
                if observations.dtype == np.object_:
                    observations = observations.astype(np.float32)
                obs = torch.as_tensor(observations, dtype=torch.float32).to(self.device)
                features = self.feature_extractor(obs)
            else:
                features = features.to(self.device)
            values = self._predict_from_features(features)
        return values + self.reward_offset

    @abstractmethod
    def _predict_from_features(self, features: torch.Tensor) -> np.ndarray:
        ...

    # --- pre-training pass (normalizer init + reward stats) + feature caching ---
    def configure_pretraining(self, train_dataset, val_dataset, batch_size: int,
                              reward_centering: bool):
        """Attach preprocessing context. Called before trainer.fit()."""
        self._pre_train_dataset = train_dataset
        self._pre_val_dataset = val_dataset
        self._pre_batch_size = batch_size
        self._reward_centering = reward_centering

    def on_fit_start(self):
        if self._features_cached:
            return
        # Pre-training pass on the training dataloader (no shuffle) to update
        # the feature extractor's running normalizer and accumulate reward stats.
        self.feature_extractor.train()
        loader = DataLoader(self._pre_train_dataset, batch_size=self._pre_batch_size,
                            shuffle=False)
        with torch.no_grad():
            for mb in loader:
                obs = mb['observations'].to(self.device)
                self.feature_extractor(obs)
                self._reward_sum += mb['rewards'].sum().item()
                self._reward_count += len(mb['rewards'])
                self._n_terminated += mb['dones'].sum().item()
        self.feature_extractor.eval()

        if self._reward_centering and self._reward_count > 0:
            eff = self._reward_count + self._n_terminated / (1 - self.discount_factor)
            self.mean_reward = self._reward_sum / eff

        # Cache features into both datasets (with the now-frozen normalizer)
        self._cache_features(self._pre_train_dataset)
        if self._pre_val_dataset is not None:
            self._cache_features(self._pre_val_dataset)
        self._features_cached = True

    def _cache_features(self, dataset):
        loader = DataLoader(dataset, batch_size=self._pre_batch_size, shuffle=False)
        feat_dim = self.feature_extractor.get_feature_dim()
        n = len(dataset)
        all_feat = torch.zeros(n, feat_dim)
        all_next = torch.zeros(n, feat_dim)
        with torch.no_grad():
            i = 0
            for mb in loader:
                j = i + len(mb['observations'])
                f = self.feature_extractor(mb['observations'].to(self.device)).cpu()
                nf = self.feature_extractor(mb['next_observations'].to(self.device)).cpu()
                all_feat[i:j] = f
                all_next[i:j] = nf
                i = j
        dataset.set_features(all_feat, all_next)

    # --- checkpoint hooks for non-hyperparam state ---
    def on_save_checkpoint(self, ckpt):
        ckpt['mean_reward'] = self.mean_reward
        ckpt['feature_extractor_info'] = self.feature_extractor.get_save_info()
        ckpt['feature_extractor_state_dict'] = self.feature_extractor.state_dict()

    def on_load_checkpoint(self, ckpt):
        self.mean_reward = ckpt.get('mean_reward', 0.0)
        if 'feature_extractor_state_dict' in ckpt:
            self.feature_extractor.load_state_dict(ckpt['feature_extractor_state_dict'])

    # --- inference-side loader used by evaluate.py (keeps old API shape) ---
    @classmethod
    def load_estimator(cls, path: Path, device: str = "auto"):
        """Load an estimator from a checkpoint saved by pl.Trainer.

        Named `load_estimator` (not `load_from_checkpoint`) to avoid clashing with
        Lightning's built-in classmethod of that name.
        """
        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        ckpt = torch.load(path, map_location=device_obj, weights_only=False)
        hparams = ckpt['hyper_parameters']
        hparams = dict(hparams)
        hparams['device_str'] = device
        estimator = cls(**hparams)
        estimator.load_state_dict(ckpt['state_dict'], strict=False)
        estimator.on_load_checkpoint(ckpt)
        estimator.to(device_obj)
        estimator.eval()
        return estimator
