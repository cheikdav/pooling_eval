"""Base class for neural network-based value estimators."""

from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
import numpy as np

from ..base import ValueEstimator, ValueNetwork
from ..feature_extractors import FeatureExtractor, IdentityExtractor

class NeuralNetEstimator(ValueEstimator):
    """Base class for neural network estimators."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float,
        feature_extractor: FeatureExtractor,
        activation: str = "relu",
        learning_rate: float = 0.001,
        device: str = "auto"
    ):
        super().__init__(obs_dim, discount_factor, feature_extractor, device)

        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate

        feature_dim = self.feature_extractor.get_feature_dim()
        self.value_net = ValueNetwork(feature_dim, hidden_sizes, activation).to(self.device)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)

    @classmethod
    def from_config(cls, method_config, network_config, obs_dim: int, gamma: float):
        feature_extractor = IdentityExtractor(obs_dim, normalize=True)

        common_params = {
            'obs_dim': obs_dim,
            'hidden_sizes': network_config.hidden_sizes,
            'discount_factor': gamma,
            'feature_extractor': feature_extractor,
            'activation': network_config.activation,
            'learning_rate': method_config.learning_rate,
            'device': network_config.device,
        }

        specific_params = cls._get_method_specific_params(method_config)
        return cls(**common_params, **specific_params)

    def train(self):
        super().train()
        self.value_net.train()

    def eval(self):
        super().eval()
        self.value_net.eval()

    def _train_step(self, feature_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.train()

        features = feature_batch['features']
        mc_returns = feature_batch['mc_returns']

        targets = self.compute_targets(feature_batch)
        values = self.value_net(features).squeeze(-1)

        loss = nn.functional.mse_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            mae = torch.abs(values - targets).mean().item()
            mc_loss = nn.functional.mse_loss(values, mc_returns).item()

        self.training_step += 1

        return {
            'loss': loss.item(),
            'mae': mae,
            'mean_value': values.mean().item(),
            'mean_target': targets.mean().item(),
            'mc_loss': mc_loss,
        }

    def _predict(self, features: torch.Tensor) -> np.ndarray:
        values = self.value_net(features).squeeze(-1)
        return values.cpu().numpy()

    def save(self, path: Path):
        checkpoint = {
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'training_step': self.training_step,
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
        }

        torch.save(checkpoint, path)

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.training_step = checkpoint['training_step']

    @classmethod
    def load_from_checkpoint(cls, path: Path, device: str = "auto"):
        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        checkpoint = torch.load(path, map_location=device_obj)
        feature_extractor = IdentityExtractor(checkpoint['obs_dim'], normalize=True)

        estimator = cls(
            obs_dim=checkpoint['obs_dim'],
            hidden_sizes=checkpoint['hidden_sizes'],
            discount_factor=checkpoint.get('discount_factor', 0.99),
            activation=checkpoint['activation'],
            learning_rate=checkpoint['learning_rate'],
            feature_extractor=feature_extractor,
            device=device
        )

        estimator.load(path)
        return estimator

    def get_config(self) -> Dict[str, Any]:
        return {
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'training_step': self.training_step,
        }


