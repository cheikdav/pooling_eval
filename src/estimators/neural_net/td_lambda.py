"""TD(lambda) value estimator (offline, periodic target refresh)."""

import copy
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

from .base import NeuralNetEstimator


def compute_lambda_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    V: torch.Tensor,
    V_next: torch.Tensor,
    episode_starts: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Backward scan for lambda-returns over flat trajectories.

    Caller is responsible for setting V_next at terminal transitions (done=1)
    to the appropriate bootstrap value: 0 with no reward centering, or
    -reward_offset with centering. V_next is used directly in the delta —
    the (1-done) mask is not applied here.

    Pads trajectories into (n_traj, T_max) and runs one tensor op per time
    step, with all trajectories updated in parallel.
    """
    device = rewards.device
    dtype = rewards.dtype
    deltas = rewards + gamma * V_next - V

    starts = episode_starts[:-1].long().cpu()
    ends = episode_starts[1:].long().cpu()
    n_traj = starts.numel()
    T_max = int((ends - starts).max().item())

    deltas_pad = torch.zeros(n_traj, T_max, device=device, dtype=dtype)
    cf_pad = torch.zeros(n_traj, T_max, device=device, dtype=dtype)
    coef = gamma * lam * (1.0 - dones)
    for i in range(n_traj):
        s, e = int(starts[i]), int(ends[i])
        L = e - s
        deltas_pad[i, :L] = deltas[s:e]
        cf_pad[i, :L] = coef[s:e]

    advantages_pad = torch.zeros_like(deltas_pad)
    a = torch.zeros(n_traj, device=device, dtype=dtype)
    for t in range(T_max - 1, -1, -1):
        a = deltas_pad[:, t] + cf_pad[:, t] * a
        advantages_pad[:, t] = a

    advantages = torch.zeros_like(deltas)
    for i in range(n_traj):
        s, e = int(starts[i]), int(ends[i])
        L = e - s
        advantages[s:e] = advantages_pad[i, :L]

    return V + advantages


class TDLambdaEstimator(NeuralNetEstimator):
    """TD(lambda) with hard-snapshot target network and periodic target refresh."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float,
        feature_extractor_save_info: dict,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        device_str: str = "auto",
        lam: float = 0.95,
        recompute_every: int = 1,
        recompute_unit: str = "epoch",
    ):
        super().__init__(obs_dim, hidden_sizes, discount_factor, feature_extractor_save_info,
                         activation, learning_rate, device_str)
        self.save_hyperparameters()
        self.lam = lam
        self.recompute_every = recompute_every
        self.recompute_unit = recompute_unit
        self.target_net = copy.deepcopy(self.value_net)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.target_net.eval()

    @classmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        return {
            'lam': method_config.lam,
            'recompute_every': method_config.recompute_every,
            'recompute_unit': method_config.recompute_unit,
        }

    @torch.no_grad()
    def recompute_targets(self, dataset) -> None:
        # Hard-snapshot the value net into the target net.
        self.target_net.load_state_dict(self.value_net.state_dict())

        # Forward V_target over the full dataset, in order, using cached features.
        loader = DataLoader(dataset, batch_size=self._pre_batch_size, shuffle=False)
        n = len(dataset)
        V = torch.zeros(n, device=self.device)
        V_next = torch.zeros(n, device=self.device)
        i = 0
        for mb in loader:
            features, next_features = self._get_features(mb)
            features = features.to(self.device)
            next_features = next_features.to(self.device)
            j = i + features.shape[0]
            V[i:j] = self.target_net(features).squeeze(-1)
            V_next[i:j] = self.target_net(next_features).squeeze(-1)
            i = j

        rewards = dataset.rewards.to(self.device) - self.mean_reward
        dones = dataset.dones.to(self.device)
        terminal_offset = -self.reward_offset
        # Replace V_next at terminal transitions with terminal_offset to match TD's
        # treatment of done states (so the lambda=0 case matches TD-online's targets).
        V_next = V_next * (1.0 - dones) + dones * terminal_offset
        episode_starts = dataset.episode_starts.to(self.device)

        targets = compute_lambda_returns(
            rewards=rewards,
            dones=dones,
            V=V,
            V_next=V_next,
            episode_starts=episode_starts,
            gamma=self.discount_factor,
            lam=self.lam,
        )
        self.cached_targets = targets.detach()

    def on_save_checkpoint(self, ckpt):
        super().on_save_checkpoint(ckpt)
        ckpt['target_net_state_dict'] = self.target_net.state_dict()

    def on_load_checkpoint(self, ckpt):
        super().on_load_checkpoint(ckpt)
        if 'target_net_state_dict' in ckpt:
            self.target_net.load_state_dict(ckpt['target_net_state_dict'])

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg['lam'] = self.lam
        cfg['recompute_every'] = self.recompute_every
        cfg['recompute_unit'] = self.recompute_unit
        cfg['estimator_type'] = 'td_lambda'
        return cfg
