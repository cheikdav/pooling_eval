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
        device: str = "auto",
        hidden_sizes: Optional[list] = None,
        activation: str = "relu",
        learning_rate: float = 0.001,
        use_pca_projection: bool = False,
        n_components: Optional[int] = None
    ):
        """Initialize least squares estimator.

        Args:
            obs_dim: Observation dimension
            discount_factor: Discount factor (gamma)
            feature_extractor: Feature extractor (typically PolicyRepresentationExtractor)
            ridge_lambda: Ridge regularization parameter
            device: Device to use
            hidden_sizes: Not used (kept for compatibility)
            activation: Not used (kept for compatibility)
            learning_rate: Not used (closed-form solution)
            use_pca_projection: Whether to project features onto top-k eigenvectors
            n_components: Number of eigenvectors to keep (if use_pca_projection=True)
        """
        super().__init__(obs_dim, discount_factor, feature_extractor, device)

        self.hidden_sizes = hidden_sizes or []
        self.activation = activation
        self.learning_rate = learning_rate
        self.ridge_lambda = ridge_lambda
        self.use_pca_projection = use_pca_projection
        self.n_components = n_components

        self.repr_dim = self.feature_extractor.get_feature_dim()

        self.pca_mean = None
        self.pca_components = None

        if use_pca_projection:
            self.working_dim = None
        else:
            self.working_dim = self.repr_dim
            self._initialize_least_squares()

        self.optimizer = None

    def _initialize_least_squares(self):
        """Initialize least squares matrices using working_dim.

        Should be called after working_dim is set (either in __init__ or after PCA fitting).
        """
        if self.working_dim is None:
            raise ValueError("working_dim must be set before initializing least squares")

        print(f"Initializing least squares with working dimension: {self.working_dim}")

        self.d = self.working_dim + 1
        self.A = self.ridge_lambda * torch.eye(self.d, device=self.device)
        self.b = torch.zeros(self.d, 1, device=self.device)
        self.w = torch.zeros(self.d, 1, device=self.device)

    def _get_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply PCA projection (if enabled) and add bias column.

        This method handles PCA projection and always adds bias as the last feature.

        Args:
            features: (batch_size, repr_dim) tensor from feature extractor

        Returns:
            features: (batch_size, working_dim+1) tensor with bias as last column
        """
        if self.use_pca_projection and self.pca_components is not None:
            centered = features - self.pca_mean
            features = centered @ self.pca_components

        bias = torch.ones(features.shape[0], 1, device=self.device)
        return torch.cat([features, bias], dim=1)

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

    def fit_pca_projection(self, preprocess_batch: Dict[str, np.ndarray]):
        """Fit PCA projection using covariance matrix from preprocessing data.

        Computes covariance matrix, extracts top-k eigenvectors as orthonormal basis.
        All subsequent feature projections will use this basis.

        Args:
            preprocess_batch: Preprocessed batch for computing covariance
                - observations: (n_samples, obs_dim) array
        """
        if not self.use_pca_projection:
            return

        if self.n_components is None:
            raise ValueError("n_components must be set when use_pca_projection=True")

        self.feature_extractor.eval()

        with torch.no_grad():
            obs = torch.FloatTensor(preprocess_batch['observations']).to(self.device)
            representations = self.feature_extractor(obs)

            self.pca_mean = representations.mean(dim=0, keepdim=True)
            centered = representations - self.pca_mean

            n_samples = centered.shape[0]
            cov_matrix = (centered.T @ centered) / n_samples

            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            k = min(self.n_components, self.repr_dim)
            self.pca_components = eigenvectors[:, :k]

            self.working_dim = k
            self._initialize_least_squares()

            print(f"PCA projection fitted: {self.repr_dim} -> {k} dimensions")
            print(f"Explained variance ratio (top {k}): {eigenvalues[:k].sum() / eigenvalues.sum():.4f}")

    @abstractmethod
    def _update_A_and_b(self, feature_batch: Dict[str, torch.Tensor], phi: torch.Tensor) -> None:
        """Update A and b matrices based on feature batch data.

        Subclasses implement specific update logic (MC vs TD).

        Args:
            feature_batch: Dictionary containing feature batch data (features, next_features, rewards, dones, mc_returns)
            phi: (batch_size, working_dim+1) features with bias
        """
        pass

    @abstractmethod
    def _compute_targets_for_metrics(self, feature_batch: Dict[str, torch.Tensor], phi: torch.Tensor) -> torch.Tensor:
        """Compute targets for metric evaluation (not used in update).

        Args:
            feature_batch: Dictionary containing feature batch data
            phi: (batch_size, working_dim+1) features with bias

        Returns:
            Target values for computing loss/MAE metrics
        """
        pass

    def _train_step(self, feature_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Incremental update by accumulating A and b."""
        with torch.no_grad():
            features = feature_batch['features']
            phi = self._get_features(features)

            self._update_A_and_b(feature_batch, phi)

            try:
                self.w = torch.linalg.solve(self.A, self.b)
            except torch._C._LinAlgError:
                self.w = torch.linalg.lstsq(self.A, self.b, rcond=None).solution

            values = (phi @ self.w).squeeze(-1)
            targets = self._compute_targets_for_metrics(feature_batch, phi)

            loss = torch.nn.functional.mse_loss(values, targets)
            mae = torch.abs(values - targets).mean()

            mc_returns = feature_batch.get('mc_returns')
            if mc_returns is not None:
                mc_loss = torch.nn.functional.mse_loss(values, mc_returns)
            else:
                mc_loss = torch.tensor(0.0)

            self.training_step += 1

            return {
                'loss': loss.item(),
                'mae': mae.item(),
                'mean_value': values.mean().item(),
                'mean_target': targets.mean().item(),
                'mc_loss': mc_loss.item(),
            }

    def get_matrix_diagnostics(self) -> Dict[str, float]:
        """Compute eigenvalue diagnostics for the A matrix.

        Returns:
            Dictionary with min/max eigenvalues and condition number
        """
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvalsh(self.A)
            min_eig = eigenvalues.min().item()
            max_eig = eigenvalues.max().item()
            condition_num = max_eig / (abs(min_eig) + 1e-10)

            abs_eigenvalues = torch.abs(eigenvalues)
            sorted_indices = torch.argsort(abs_eigenvalues, descending=True)
            sorted_eigs = eigenvalues[sorted_indices]

            top_5 = sorted_eigs[:5].cpu().numpy()
            bottom_5 = sorted_eigs[-5:].cpu().numpy()

            return {
                'min_eigenvalue': min_eig,
                'max_eigenvalue': max_eig,
                'condition_number': condition_num,
                'top_5_abs': top_5,
                'bottom_5_abs': bottom_5,
            }

    def _predict(self, features: torch.Tensor) -> np.ndarray:
        """Predict values from features.

        Args:
            features: Tensor of shape (n, repr_dim)

        Returns:
            Predicted values of shape (n,)
        """
        features_with_bias = self._get_features(features)
        values = (features_with_bias @ self.w).squeeze(-1)
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
            'repr_dim': self.repr_dim,
            'use_pca_projection': self.use_pca_projection,
            'n_components': self.n_components,
            'pca_mean': self.pca_mean,
            'pca_components': self.pca_components,
            'working_dim': self.working_dim,
        })
        return checkpoint

    def _load_from_checkpoint_dict(self, checkpoint: Dict[str, Any]):
        """Load least squares specific fields."""
        super()._load_from_checkpoint_dict(checkpoint)

        self.use_pca_projection = checkpoint.get('use_pca_projection', False)
        self.n_components = checkpoint.get('n_components', None)
        self.pca_mean = checkpoint.get('pca_mean', None)
        self.pca_components = checkpoint.get('pca_components', None)
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
            device=device,
            hidden_sizes=checkpoint.get('hidden_sizes', []),
            activation=checkpoint.get('activation', 'relu'),
            learning_rate=checkpoint.get('learning_rate', 0.001),
            use_pca_projection=checkpoint.get('use_pca_projection', False),
            n_components=checkpoint.get('n_components', None)
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
        }
