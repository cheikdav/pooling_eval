"""Base class for least squares value estimators."""

from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np

from ..base import ValueEstimator
from ..feature_extractors import FeatureExtractor, create_feature_extractor, create_feature_extractor_from_saved_info

class LeastSquaresEstimator(ValueEstimator):
    """Base class for least squares value estimators using policy representations."""

    def __init__(
        self,
        obs_dim: int,
        discount_factor: float,
        feature_extractor: FeatureExtractor,
        ridge_lambda: float = 1e-6,
        n_components: Optional[int] = None,
        device: str = "auto",
        hidden_sizes: Optional[list] = None,
        activation: str = "relu",
        learning_rate: float = 0.001
    ):
        """Initialize least squares estimator.

        Args:
            obs_dim: Observation dimension
            discount_factor: Discount factor (gamma)
            feature_extractor: Feature extractor (typically PolicyRepresentationExtractor)
            ridge_lambda: Ridge regularization parameter
            n_components: Number of SVD components for dimensionality reduction (None = no reduction)
            device: Device to use
            hidden_sizes: Not used (kept for compatibility)
            activation: Not used (kept for compatibility)
            learning_rate: Not used (closed-form solution)
        """
        super().__init__(obs_dim, discount_factor, feature_extractor, device)

        self.hidden_sizes = hidden_sizes or []
        self.activation = activation
        self.learning_rate = learning_rate
        self.ridge_lambda = ridge_lambda
        self.n_components = n_components

        self.repr_dim = self.feature_extractor.get_feature_dim()
        self.working_dim = self.repr_dim

        # Enable bias in feature extractor for least squares
        self.feature_extractor.add_bias = True

        self._initialize_least_squares()

        self.w_is_stale = True  # Flag to track if w needs recomputation
        self.optimizer = None

    def _initialize_least_squares(self):
        """Initialize least squares matrices using working_dim."""
        if self.working_dim is None:
            raise ValueError("working_dim must be set before initializing least squares")

        #print(f"Initializing least squares with working dimension: {self.working_dim}")

        self.d = self.working_dim + 1
        self.A = self.ridge_lambda * torch.eye(self.d, device=self.device)
        self.b = torch.zeros(self.d, 1, device=self.device)
        self.w = torch.zeros(self.d, 1, device=self.device)

    def _update_w(self):
        """Solve the linear system A @ w = b to update w.

        Only called when w_is_stale=True, indicating that A and b have been
        updated but w has not been recomputed yet.

        If n_components is set, uses SVD-based dimensionality reduction:
        - Computes SVD of A: A = U @ S @ V^T
        - Keeps top k components: U_k, S_k, V_k
        - Computes pseudo-inverse: A_k_pinv = V_k @ S_k^-1 @ U_k^T
        - Solves: w = A_k_pinv @ b
        """
        if not self.w_is_stale:
            return

        if self.n_components is None:
            # No dimensionality reduction - solve directly
            try:
                self.w = torch.linalg.solve(self.A, self.b)
            except torch._C._LinAlgError:
                self.w = torch.linalg.lstsq(self.A, self.b, rcond=None).solution
        else:
            # SVD-based dimensionality reduction
            # A = U @ S @ V^T
            U, S, Vt = torch.linalg.svd(self.A, full_matrices=False)

            # Keep top k components
            k = min(self.n_components, S.shape[0])
            U_k = U[:, :k]
            S_k = S[:k]
            V_k = Vt[:k, :].T  # Transpose to get V from V^T

            # Compute pseudo-inverse: A_k_pinv = V_k @ diag(1/S_k) @ U_k^T
            S_k_inv = 1.0 / S_k
            A_k_pinv = V_k @ torch.diag(S_k_inv) @ U_k.T

            # Solve: w = A_k_pinv @ b
            self.w = A_k_pinv @ self.b

        self.w_is_stale = False

    @classmethod
    def from_config(cls, method_config, network_config, obs_dim: int, gamma: float):
        """Create estimator from configuration.

        Args:
            method_config: Method-specific configuration (LeastSquaresConfig subclass)
            network_config: Network configuration
            obs_dim: Observation dimension
            gamma: Discount factor (from training config, shared by all methods)

        Returns:
            Estimator instance
        """
        feature_extractor = create_feature_extractor(
            method_config.feature_extractor,
            obs_dim,
            device=network_config.device
        )

        common_params = {
            'obs_dim': obs_dim,
            'discount_factor': gamma,
            'feature_extractor': feature_extractor,
            'device': network_config.device,
            'hidden_sizes': network_config.hidden_sizes,
            'activation': network_config.activation,
            'learning_rate': method_config.learning_rate,
        }

        specific_params = cls._get_method_specific_params(method_config)

        return cls(**common_params, **specific_params)

    @abstractmethod
    def _update_A_and_b(self, feature_batch: Dict[str, torch.Tensor]) -> None:
        """Update A and b matrices based on feature batch data.

        Subclasses implement specific update logic (MC vs TD).
        Features in feature_batch already include bias term.

        Args:
            feature_batch: Dictionary containing feature batch data (features include bias)
        """
        pass


    def _train_step(self, feature_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Incremental update by accumulating A and b.

        Note: This method updates A and b but does NOT recompute w immediately.
        Instead, it sets w_is_stale=True. The w vector is only recomputed when
        needed (e.g., during prediction or evaluation).
        """
        with torch.no_grad():
            self._update_A_and_b(feature_batch)
            self.w_is_stale = True

            self.training_step += 1

            return {}

    def _predict(self, features: torch.Tensor) -> np.ndarray:
        """Predict values from features.

        Args:
            features: Tensor of shape (n, repr_dim+1) - already includes bias

        Returns:
            Predicted values of shape (n,)
        """
        # Ensure w is up to date before prediction
        self._update_w()

        values = (features @ self.w).squeeze(-1)
        return values.cpu().numpy()

    def _build_checkpoint(self) -> Dict[str, Any]:
        """Build checkpoint with least squares specific fields."""
        checkpoint = super()._build_checkpoint()
        checkpoint.update({
            'A': self.A,
            'b': self.b,
            'w': self.w,
            'd': self.d,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'ridge_lambda': self.ridge_lambda,
            'n_components': self.n_components,
            'repr_dim': self.repr_dim,
            'working_dim': self.working_dim,
        })
        return checkpoint

    def _load_from_checkpoint_dict(self, checkpoint: Dict[str, Any]):
        """Load least squares specific fields."""
        super()._load_from_checkpoint_dict(checkpoint)

        self.working_dim = checkpoint.get('working_dim', self.repr_dim)

        self.A = checkpoint['A']
        self.b = checkpoint['b']
        self.w = checkpoint['w']
        self.d = checkpoint.get('d', self.working_dim + 1)
        self.repr_dim = checkpoint['repr_dim']

    @classmethod
    def load_from_checkpoint(cls, path: Path, device: str = "auto"):
        """Load estimator from checkpoint file.

        Args:
            path: Path to checkpoint file
            device: Device to load model on ('auto', 'cpu', or 'cuda')

        Returns:
            Estimator instance with loaded weights
        """
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
            discount_factor=checkpoint.get('discount_factor', 0.99),
            feature_extractor=feature_extractor,
            ridge_lambda=checkpoint.get('ridge_lambda', 1e-6),
            n_components=checkpoint.get('n_components', None),
            device=device,
            hidden_sizes=checkpoint.get('hidden_sizes', []),
            activation=checkpoint.get('activation', 'relu'),
            learning_rate=checkpoint.get('learning_rate', 0.001)
        )

        estimator.load(path)

        return estimator

    def get_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        return {
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'training_step': self.training_step,
            'ridge_lambda': self.ridge_lambda,
            'n_components': self.n_components,
        }
