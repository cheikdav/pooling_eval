"""Helpers to compute the PPO surrogate gradient ∇θ E[A · log π_θ(a|s)]."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import A2C, PPO, SAC, TD3


def policy_log_prob(model, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """log π_θ(a|s) for arbitrary (obs, actions) batches under the policy.

    Differentiable w.r.t. the policy parameters.
    """
    if isinstance(model, (PPO, A2C)):
        _, log_prob, _ = model.policy.evaluate_actions(obs, actions)
        return log_prob
    if isinstance(model, SAC):
        actor = model.policy.actor
        mean, log_std, _ = actor.get_action_dist_params(obs)
        actor.action_dist.proba_distribution(mean, log_std)
        return actor.action_dist.log_prob(actions)
    if isinstance(model, TD3):
        raise NotImplementedError("TD3 has a deterministic policy; PPO surrogate gradient is undefined.")
    raise NotImplementedError(f"Unsupported algorithm: {type(model).__name__}")


def trainable_policy_params(model) -> List[torch.nn.Parameter]:
    """Parameters that the policy gradient flows through (actor side).

    Critic parameters are also returned for SAC/PPO (since `model.policy.parameters()`
    yields them), but they receive zero gradient from the PPO surrogate; harmless
    for cosine similarity since direction-only.
    """
    return [p for p in model.policy.parameters() if p.requires_grad]


def compute_mc_returns(rewards_list: List[np.ndarray], gamma: float) -> List[np.ndarray]:
    """Per-trajectory MC returns: G_t = Σ γ^k r_{t+k}."""
    out = []
    for r in rewards_list:
        T = len(r)
        g = np.zeros(T, dtype=np.float64)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = float(r[t]) + gamma * running
            g[t] = running
        out.append(g)
    return out


def compute_surrogate_gradient(model, obs_flat: np.ndarray, act_flat: np.ndarray,
                               adv_flat: np.ndarray) -> np.ndarray:
    """Flat ∇θ of (1/T) Σ A_t · log π_θ(a_t|s_t).

    obs_flat: (T, obs_dim), act_flat: (T, act_dim), adv_flat: (T,).
    Returns flat numpy gradient over all trainable policy params.
    """
    obs_t = torch.as_tensor(obs_flat, dtype=torch.float32)
    act_t = torch.as_tensor(act_flat, dtype=torch.float32)
    adv_t = torch.as_tensor(adv_flat, dtype=torch.float32)

    log_prob = policy_log_prob(model, obs_t, act_t)
    surrogate = (adv_t * log_prob).mean()

    params = trainable_policy_params(model)
    grads = torch.autograd.grad(surrogate, params, allow_unused=True)
    flat = torch.cat([
        g.detach().flatten() if g is not None else torch.zeros_like(p).flatten()
        for g, p in zip(grads, params)
    ])
    return flat.cpu().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two flat vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
