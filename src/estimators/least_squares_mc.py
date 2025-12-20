"""Least Squares Monte Carlo estimator using policy network representations."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from src.estimators.base import ValueEstimator
from src.policy_representations import load_policy_representation_extractor


class LeastSquaresMCEstimator(ValueEstimator):
    """Least Squares Monte Carlo estimator using frozen policy representations.

    Uses incremental updates via Woodbury formula to handle mini-batches efficiently.
    Maintains: A_inv = (Φ^T Φ + λI)^(-1) and b = Φ^T y
    Solution: w = A_inv @ b
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
        """Initialize Least Squares MC estimator.

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
            raise ValueError("policy_path is required for LeastSquaresMCEstimator")

        policy_path = Path(policy_path)
        self.repr_extractor = load_policy_representation_extractor(policy_path, algorithm, device)
        self.repr_dim = self.repr_extractor.output_dim

        # Incremental least squares matrices
        # A = Φ^T Φ + λI, b = Φ^T y, solution: w = A^(-1) b
        d = self.repr_dim + 1  # +1 for bias
        self.A_inv = (1.0 / self.ridge_lambda) * torch.eye(d, device=self.device)
        self.b = torch.zeros(d, 1, device=self.device)
        self.w = torch.zeros(d, 1, device=self.device)  # Current weights

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

    def train(self):
        pass
    
    def eval(self):
        pass

    def _update_woodbury(self, phi: torch.Tensor, targets: torch.Tensor):
        """Update A_inv and b using Woodbury matrix identity.

        Woodbury formula for rank-k update:
        (A + U V^T)^(-1) = A^(-1) - A^(-1) U (I + V^T A^(-1) U)^(-1) V^T A^(-1)

        Args:
            phi: (batch_size, repr_dim+1) representations with bias
            targets: (batch_size, 1) target values
        """
        # Update b: b_new = b_old + Φ^T y
        self.b = self.b + phi.T @ targets

        # Woodbury update for A_inv
        # A_new = A_old + phi.T @ phi
        A_inv_phiT = self.A_inv @ phi.T  # (d, n)
        middle = torch.eye(phi.shape[0], device=self.device) + phi @ A_inv_phiT  # (n, n)
        middle_inv = torch.linalg.inv(middle)  # (n, n)

        # A_inv_new = A_inv - A_inv @ phi.T @ middle_inv @ phi @ A_inv
        self.A_inv = self.A_inv - A_inv_phiT @ middle_inv @ A_inv_phiT.T

        # Update weights: w = A_inv @ b
        self.w = self.A_inv @ self.b

    def compute_targets(self, mini_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Monte Carlo targets."""
        return mini_batch['mc_returns'].to(self.device)

    def train_step(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Incremental update using Woodbury formula."""
        self.repr_extractor.eval()

        with torch.no_grad():
            # Extract representations
            obs = mini_batch['observations'].to(self.device)
            representations = self.repr_extractor(obs)  # (batch_size, repr_dim)

            # Add bias term
            ones = torch.ones(representations.shape[0], 1, device=self.device)
            phi = torch.cat([representations, ones], dim=1)  # (batch_size, repr_dim+1)

            # Target values
            targets = mini_batch['mc_returns'].to(self.device).unsqueeze(1)  # (batch_size, 1)

            # Incremental update
            self._update_woodbury(phi, targets)

            # Compute predictions: phi @ w
            values = (phi @ self.w).squeeze(-1)
            targets_1d = targets.squeeze(-1)

            loss = torch.nn.functional.mse_loss(values, targets_1d)
            mae = torch.abs(values - targets_1d).mean()

            self.training_step += 1

            return {
                'loss': loss.item(),
                'mae': mae.item(),
                'mean_value': values.mean().item(),
                'mean_target': targets_1d.mean().item(),
                'mc_loss': loss.item(),
            }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict values for given observations."""
        self.repr_extractor.eval()

        with torch.no_grad():
            obs = torch.FloatTensor(observations).to(self.device)
            representations = self.repr_extractor(obs)  # (n, repr_dim)

            # Add bias and compute: phi @ w
            ones = torch.ones(representations.shape[0], 1, device=self.device)
            phi = torch.cat([representations, ones], dim=1)  # (n, repr_dim+1)
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
            'estimator_type': 'least_squares_mc',
        }
