"""DQN-style value estimator with target network."""

import torch
from typing import Dict, Any
import copy

from .base import NeuralNetEstimator
from ..feature_extractors import FeatureExtractor


class DQNEstimator(NeuralNetEstimator):
    """DQN estimator with target network and Polyak averaging."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float,
        feature_extractor: FeatureExtractor,
        activation: str = "relu",
        learning_rate: float = 0.001,
        device: str = "auto",
        target_update_rate: float = 1e-5
    ):
        super().__init__(obs_dim, hidden_sizes, discount_factor, feature_extractor, activation, learning_rate, device)
        self.target_update_rate = target_update_rate

        self.target_net = copy.deepcopy(self.value_net).to(self.device)

        # Compile target network for faster execution (PyTorch 2.0+)
        try:
            self.target_net = torch.compile(self.target_net, mode='default')
        except (AttributeError, RuntimeError) as e:
            # torch.compile not available (PyTorch < 2.0) or compilation failed
            pass

        self.target_net.eval()
        self.steps_since_target_update = 0

    @classmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        return {
            'target_update_rate': method_config.target_update_rate,
        }

    def update_target_network(self):
        # In-place Polyak averaging (more efficient than state_dict approach)
        for target_param, param in zip(self.target_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                (1 - self.target_update_rate) * target_param.data +
                self.target_update_rate * param.data
            )
        self.steps_since_target_update = 0

    def compute_targets(self, feature_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        next_features = feature_batch['next_features']
        rewards = feature_batch['rewards']
        dones = feature_batch['dones']

        self.target_net.eval()

        with torch.no_grad():
            next_values = self.target_net(next_features).squeeze(-1)
            targets = rewards + self.discount_factor * next_values * (1 - dones)

        return targets

    def _train_step(self, feature_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        metrics = super()._train_step(feature_batch)
        self.update_target_network()
        return metrics

    def _build_checkpoint(self) -> Dict[str, Any]:
        """Build checkpoint with DQN specific fields."""
        checkpoint = super()._build_checkpoint()
        checkpoint.update({
            'target_net_state_dict': self.target_net.state_dict(),
            'steps_since_target_update': self.steps_since_target_update,
            'target_update_rate': self.target_update_rate,
        })
        return checkpoint

    def _load_from_checkpoint_dict(self, checkpoint: Dict[str, Any]):
        """Load DQN specific fields."""
        super()._load_from_checkpoint_dict(checkpoint)
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.steps_since_target_update = checkpoint['steps_since_target_update']

    @classmethod
    def load_from_checkpoint(cls, path, device: str = "auto"):
        from ..feature_extractors import create_feature_extractor_from_saved_info

        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        checkpoint = torch.load(path, map_location=device_obj)

        feature_extractor = create_feature_extractor_from_saved_info(
            checkpoint['feature_extractor_info'],
            device=device
        )

        estimator = cls(
            obs_dim=checkpoint['obs_dim'],
            hidden_sizes=checkpoint['hidden_sizes'],
            discount_factor=checkpoint.get('discount_factor', 0.99),
            feature_extractor=feature_extractor,
            activation=checkpoint['activation'],
            learning_rate=checkpoint['learning_rate'],
            device=device,
            target_update_rate=checkpoint.get('target_update_rate', 1e-5)
        )

        estimator.load(path)
        return estimator

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            'discount_factor': self.discount_factor,
            'target_update_rate': self.target_update_rate,
            'estimator_type': 'dqn',
        })
        return config
