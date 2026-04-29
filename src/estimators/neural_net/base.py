"""Lightning-based NN value estimator."""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from ..base import ValueEstimator, ValueNetwork


class NeuralNetEstimator(ValueEstimator):
    """Base class for neural-network value estimators.

    Subclasses override `compute_targets(batch)` only.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float,
        feature_extractor_save_info: dict,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        device_str: str = "auto",
    ):
        super().__init__(obs_dim, discount_factor, feature_extractor_save_info, device_str)
        self.save_hyperparameters()

        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate

        feat_dim = self.feature_extractor.get_feature_dim()
        self.value_net = ValueNetwork(feat_dim, hidden_sizes, activation)

    @classmethod
    def from_config(cls, method_config, network_config, obs_dim: int, gamma: float):
        from ..feature_extractors import create_feature_extractor
        fx = create_feature_extractor(method_config.feature_extractor, obs_dim,
                                      device=network_config.device)
        if method_config.network is not None:
            hidden_sizes = method_config.network.hidden_sizes
            activation = method_config.network.activation
        else:
            hidden_sizes = network_config.hidden_sizes
            activation = network_config.activation

        common = dict(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            discount_factor=gamma,
            feature_extractor_save_info=fx.get_save_info(),
            activation=activation,
            learning_rate=method_config.learning_rate,
            device_str=network_config.device,
        )
        specific = cls._get_method_specific_params(method_config)
        return cls(**common, **specific)

    @classmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        return {}

    # --- Lightning hooks ---
    def configure_optimizers(self):
        return torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        features, next_features = self._get_features(batch)
        feature_batch = {
            'idx': batch['idx'],
            'features': features,
            'next_features': next_features,
            'rewards': batch['rewards'],
            'dones': batch['dones'],
            'mc_returns': batch['mc_returns'],
        }
        targets = self.compute_targets(feature_batch)
        values = self.value_net(features).squeeze(-1)
        loss = nn.functional.mse_loss(values, targets)

        with torch.no_grad():
            mc_loss = nn.functional.mse_loss(values, feature_batch['mc_returns'] - self.reward_offset)

        self.log_dict({
            'train/loss': loss.detach(),
            'train/mc_loss': mc_loss,
            'train/mean_value': values.mean().detach(),
            'train/mean_target': targets.mean().detach(),
        }, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        features, _ = self._get_features(batch)
        with torch.no_grad():
            values = self.value_net(features).squeeze(-1)
            mc_loss = nn.functional.mse_loss(values, batch['mc_returns'] - self.reward_offset)
        self.log('val/mc_loss', mc_loss, on_step=False, on_epoch=True, prog_bar=False)
        return mc_loss

    def _predict_from_features(self, features: torch.Tensor) -> np.ndarray:
        return self.value_net(features).squeeze(-1).cpu().numpy()

    def get_config(self) -> Dict[str, Any]:
        return {
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
        }
