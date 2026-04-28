"""TD(0) value estimator with Polyak-averaged target network."""

import copy
from typing import Dict, Any
import torch

from .base import NeuralNetEstimator


class TDEstimator(NeuralNetEstimator):
    """TD(0) with a Polyak-averaged target network."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float,
        feature_extractor_save_info: dict,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        device_str: str = "auto",
        target_update_rate: float = 1e-5,
    ):
        super().__init__(obs_dim, hidden_sizes, discount_factor, feature_extractor_save_info,
                         activation, learning_rate, device_str)
        self.save_hyperparameters()
        self.target_update_rate = target_update_rate
        self.target_net = copy.deepcopy(self.value_net)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.target_net.eval()

    @classmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        return {'target_update_rate': method_config.target_update_rate}

    def compute_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            next_values = self.target_net(batch['next_features']).squeeze(-1)
            rewards = batch['rewards'] - self.mean_reward
            dones = batch['dones']
            terminal = -self.reward_offset
            return rewards + self.discount_factor * (
                next_values * (1 - dones) + dones * terminal
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        tau = self.target_update_rate
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.value_net.parameters()):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

    def on_save_checkpoint(self, ckpt):
        super().on_save_checkpoint(ckpt)
        ckpt['target_net_state_dict'] = self.target_net.state_dict()

    def on_load_checkpoint(self, ckpt):
        super().on_load_checkpoint(ckpt)
        if 'target_net_state_dict' in ckpt:
            self.target_net.load_state_dict(ckpt['target_net_state_dict'])

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg['target_update_rate'] = self.target_update_rate
        cfg['estimator_type'] = 'td'
        return cfg
