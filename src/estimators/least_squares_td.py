"""Least Squares Temporal Difference (LSTD) estimator using policy network representations."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from src.estimators.base import ValueEstimator
from src.policy_representations import load_policy_representation_extractor


class LeastSquaresTDEstimator(ValueEstimator):
    """Least Squares Temporal Difference estimator using frozen policy representations.

    Solves the Bellman equation directly using closed-form solution:
    w = [Φ^T (Φ - γΦ')]^(-1) Φ^T r

    Where:
    - Φ: (n, d) representations of current states
    - Φ': (n, d) representations of next states
    - γ: discount factor
    - r: (n, 1) rewards

    Incremental formulation:
    - A = Φ^T (Φ - γΦ') + λI  (d, d)
    - b = Φ^T r  (d, 1)
    - w = A^(-1) b  (d, 1)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float = 0.99,
        activation: str = "relu",
        learning_rate: float = 0.001,
        device: str = "auto",
        policy_path: Optional[str] = None,
        algorithm: str = "PPO",
        normalize_observations: bool = False,
        ridge_lambda: float = 1e-6
    ):
        """Initialize Least Squares TD estimator.

        Args:
            obs_dim: Observation dimension
            hidden_sizes: Not used (kept for compatibility)
            discount_factor: Discount factor (gamma)
            activation: Not used (kept for compatibility)
            learning_rate: Not used (closed-form solution)
            device: Device to use
            policy_path: Path to trained policy (.zip file)
            algorithm: Policy algorithm (PPO, A2C, SAC, TD3)
            normalize_observations: Not used
            ridge_lambda: Ridge regularization parameter
        """
        self.obs_dim = obs_dim
        self.hidden_sizes = hidden_sizes
        self.discount_factor = discount_factor
        self.activation = activation
        self.learning_rate = learning_rate
        self.normalize_observations = normalize_observations
        self.ridge_lambda = ridge_lambda
        self.algorithm = algorithm

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load policy and create representation extractor
        if policy_path is None:
            raise ValueError("policy_path is required for LeastSquaresTDEstimator")

        policy_path = Path(policy_path)
        self.repr_extractor = load_policy_representation_extractor(policy_path, algorithm, device)
        self.repr_dim = self.repr_extractor.output_dim

        # Incremental LSTD matrices
        # A = Φ^T (Φ - γΦ') + λI
        # b = Φ^T r
        # solution: w = A^(-1) b
        d = self.repr_dim + 1  # +1 for bias
        self.A_inv = (1.0 / self.ridge_lambda) * torch.eye(d, device=self.device)
        self.b = torch.zeros(d, 1, device=self.device)
        self.w = torch.zeros(d, 1, device=self.device)

        self.optimizer = None
        self.training_step = 0

    @classmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        """Get method-specific parameters from config."""
        return {
            'policy_path': method_config.policy_path,
            'algorithm': method_config.algorithm,
            'ridge_lambda': method_config.ridge_lambda,
        }

    def _update_woodbury(self, phi: torch.Tensor, phi_next: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor):
        """Update A_inv and b using Woodbury matrix identity for LSTD.

        Update equations:
        A_new = A_old + Φ^T (Φ - γΦ')
        b_new = b_old + Φ^T r

        For A update, we have: A_new = A_old + U V^T where:
        U = Φ^T = phi.T  (d, n)
        V = (Φ - γΦ') = phi_diff  (n, d)

        Woodbury formula: (A + UV^T)^(-1) = A^(-1) - A^(-1)U(I + V^T A^(-1)U)^(-1)V^T A^(-1)

        Args:
            phi: (n, d) representations of current states with bias
            phi_next: (n, d) representations of next states with bias
            rewards: (n, 1) rewards
            dones: (n, 1) done flags
        """
        n = phi.shape[0]

        # Compute TD difference features: (Φ - γΦ')
        # For terminal states, Φ' = 0 (no next state value)
        gamma_mask = self.discount_factor * (1.0 - dones)  # (n, 1)
        phi_diff = phi - gamma_mask * phi_next  # (n, d)

        # Update b: b_new = b_old + Φ^T r
        self.b = self.b + phi.T @ rewards  # (d, n) @ (n, 1) = (d, 1)

        # Woodbury update for A_inv
        # U = Φ^T (transposed to (d, n))
        # V = phi_diff (n, d)

        # Step 1: A_inv @ U
        U = phi.T  # (d, n)
        A_inv_U = self.A_inv @ U  # (d, d) @ (d, n) = (d, n)

        # Step 2: V^T @ A_inv @ U
        V = phi_diff  # (n, d)
        middle = torch.eye(n, device=self.device) + V @ A_inv_U  # (n, n)

        # Step 3: Invert middle
        middle_inv = torch.linalg.inv(middle)  # (n, n)

        # Step 4: V^T @ A_inv
        V_T_A_inv = V.T @ self.A_inv  # (d, n) @ (d, d) = (d, d)

        # Step 5: Final Woodbury update
        # A_inv_new = A_inv - A_inv @ U @ middle_inv @ V^T @ A_inv
        self.A_inv = self.A_inv - A_inv_U @ middle_inv @ V_T_A_inv  # (d, d)

        # Update weights: w = A_inv @ b
        self.w = self.A_inv @ self.b  # (d, d) @ (d, 1) = (d, 1)

    def compute_targets(self, mini_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute TD targets for evaluation."""
        rewards = mini_batch['rewards'].to(self.device)
        next_obs = mini_batch['next_observations'].to(self.device)
        dones = mini_batch['dones'].to(self.device)

        with torch.no_grad():
            # Get next state values
            next_representations = self.repr_extractor(next_obs)
            ones = torch.ones(next_representations.shape[0], 1, device=self.device)
            phi_next = torch.cat([next_representations, ones], dim=1)
            next_values = (phi_next @ self.w).squeeze(-1)

            # TD target: r + γ * V(s') * (1 - done)
            targets = rewards + self.discount_factor * next_values * (1.0 - dones)

        return targets

    def train(self):
        pass
    
    def eval(self):
        pass

    def train_step(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Incremental LSTD update using Woodbury formula."""
        self.repr_extractor.eval()

        with torch.no_grad():
            # Extract representations for current and next states
            obs = mini_batch['observations'].to(self.device)
            next_obs = mini_batch['next_observations'].to(self.device)
            rewards = mini_batch['rewards'].to(self.device).unsqueeze(1)  # (n, 1)
            dones = mini_batch['dones'].to(self.device).unsqueeze(1)  # (n, 1)

            representations = self.repr_extractor(obs)  # (n, repr_dim)
            next_representations = self.repr_extractor(next_obs)  # (n, repr_dim)

            # Add bias term
            ones = torch.ones(representations.shape[0], 1, device=self.device)
            phi = torch.cat([representations, ones], dim=1)  # (n, d)
            phi_next = torch.cat([next_representations, ones], dim=1)  # (n, d)

            # Incremental LSTD update
            self._update_woodbury(phi, phi_next, rewards, dones)

            # Compute metrics with updated weights
            values = (phi @ self.w).squeeze(-1)  # (n,)

            # Compute TD targets for evaluation
            next_values = (phi_next @ self.w).squeeze(-1)
            targets = rewards.squeeze(-1) + self.discount_factor * next_values * (1.0 - dones.squeeze(-1))

            # Also compute MC loss if available
            mc_returns = mini_batch.get('mc_returns')
            if mc_returns is not None:
                mc_returns = mc_returns.to(self.device)
                mc_loss = torch.nn.functional.mse_loss(values, mc_returns)
            else:
                mc_loss = torch.tensor(0.0)

            loss = torch.nn.functional.mse_loss(values, targets)
            mae = torch.abs(values - targets).mean()

            self.training_step += 1

            return {
                'loss': loss.item(),
                'mae': mae.item(),
                'mean_value': values.mean().item(),
                'mean_target': targets.mean().item(),
                'mc_loss': mc_loss.item(),
            }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict values for given observations."""
        self.repr_extractor.eval()

        with torch.no_grad():
            obs = torch.FloatTensor(observations).to(self.device)
            representations = self.repr_extractor(obs)  # (n, repr_dim)

            # Add bias and compute: phi @ w
            ones = torch.ones(representations.shape[0], 1, device=self.device)
            phi = torch.cat([representations, ones], dim=1)  # (n, d)
            values = (phi @ self.w).squeeze(-1)

            return values.cpu().numpy()

    def save(self, path: Path):
        """Save estimator to disk."""
        torch.save({
            'repr_extractor_state_dict': self.repr_extractor.state_dict(),
            'A_inv': self.A_inv,
            'b': self.b,
            'w': self.w,
            'training_step': self.training_step,
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'normalize_observations': self.normalize_observations,
            'ridge_lambda': self.ridge_lambda,
            'algorithm': self.algorithm,
            'repr_dim': self.repr_dim,
            'discount_factor': self.discount_factor,
        }, path)

    def load(self, path: Path):
        """Load estimator from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.repr_extractor.load_state_dict(checkpoint['repr_extractor_state_dict'])
        self.A_inv = checkpoint['A_inv']
        self.b = checkpoint['b']
        self.w = checkpoint['w']
        self.training_step = checkpoint['training_step']
        self.repr_dim = checkpoint['repr_dim']

    def get_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        return {
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'training_step': self.training_step,
            'ridge_lambda': self.ridge_lambda,
            'algorithm': self.algorithm,
            'discount_factor': self.discount_factor,
            'estimator_type': 'least_squares_td',
        }
