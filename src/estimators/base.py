"""Base class for value estimators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class RunningNormalizer:
    """Tracks running mean and std for online normalization."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, values: torch.Tensor):
        """Update statistics with new batch of values."""
        batch_mean = values.mean().item()
        batch_var = values.var().item()
        batch_count = values.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Welford's online algorithm for running statistics
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize values using running statistics."""
        std = np.sqrt(self.var + self.epsilon)
        return (values - self.mean) / std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize values back to original scale."""
        std = np.sqrt(self.var + self.epsilon)
        return values * std + self.mean

    def state_dict(self) -> Dict[str, Any]:
        """Get state for saving."""
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count,
            'epsilon': self.epsilon,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']
        self.epsilon = state['epsilon']


class ValueNetwork(nn.Module):
    """Simple MLP for value function approximation."""

    def __init__(self, input_dim: int, hidden_sizes: list, activation: str = "relu", normalize_observations: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.normalize_observations = normalize_observations

        if self.normalize_observations:
            self.obs_normalizer = RunningNormalizer()
        else:
            self.obs_normalizer = None

        # Build network layers
        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_size = hidden_size

        # Output layer (single value)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Value estimates of shape (batch_size, 1)
        """
        if self.normalize_observations:
            if self.training:
                self.obs_normalizer.update(x)
            x = self.obs_normalizer.normalize(x)

        return self.network(x)


class ValueEstimator(ABC):
    """Base class for all value function estimators."""

    def __init__(
        self,
        obs_dim: int,
        discount_factor: float,
        device: str = "auto"
    ):
        """Initialize value estimator.

        Args:
            obs_dim: Observation dimension
            discount_factor: Discount factor (gamma) - common to all estimators
            device: Device to use ('auto', 'cpu', or 'cuda')
        """
        self.obs_dim = obs_dim
        self.discount_factor = discount_factor

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.training_step = 0

    @classmethod
    @abstractmethod
    def _get_method_specific_params(cls, method_config) -> Dict[str, Any]:
        """Get method-specific parameters from config.

        Args:
            method_config: Method-specific configuration

        Returns:
            Dictionary of method-specific parameters to pass to __init__
        """
        pass

    @classmethod
    def from_config(cls, method_config, network_config, obs_dim: int, gamma: float):
        """Create estimator from configuration.

        Args:
            method_config: Method-specific configuration (BaseEstimatorConfig subclass)
            network_config: Network configuration
            obs_dim: Observation dimension
            gamma: Discount factor (from training config, shared by all methods)

        Returns:
            Estimator instance
        """
        raise NotImplementedError("Subclasses must implement from_config")

    @abstractmethod
    def compute_targets(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute target values for the batch.

        Args:
            batch: Dictionary containing preprocessed transition data with fields:
                - observations: (n_transitions, obs_dim) array
                - next_observations: (n_transitions, obs_dim) array
                - rewards: (n_transitions,) array
                - dones: (n_transitions,) array
                - mc_returns: (n_transitions,) array - precomputed Monte Carlo returns

        Returns:
            Target values as torch tensor of shape (n_transitions,)
        """
        pass

    @abstractmethod
    def train(self):
        """Set estimator to training mode."""
        pass

    @abstractmethod
    def eval(self):
        """Set estimator to evaluation mode."""
        pass

    @abstractmethod
    def train_step(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step on a mini-batch.

        Args:
            mini_batch: Dictionary containing mini-batch data (torch tensors):
                - observations: (batch_size, obs_dim)
                - next_observations: (batch_size, obs_dim)
                - rewards: (batch_size,)
                - dones: (batch_size,)
                - mc_returns: (batch_size,)

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict values for given observations.

        Args:
            observations: Array of shape (n, obs_dim)

        Returns:
            Predicted values of shape (n,)
        """
        pass

    @abstractmethod
    def save(self, path: Path):
        """Save estimator to disk.

        Args:
            path: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load estimator from disk.

        Args:
            path: Path to checkpoint
        """
        pass

    @classmethod
    @abstractmethod
    def load_from_checkpoint(cls, path: Path, device: str = "auto"):
        """Load estimator from checkpoint file.

        Args:
            path: Path to checkpoint file
            device: Device to load model on ('auto', 'cpu', or 'cuda')

        Returns:
            Estimator instance with loaded weights
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        pass


class NeuralNetEstimator(ValueEstimator):
    """Base class for neural network-based value estimators."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list,
        discount_factor: float,
        activation: str = "relu",
        learning_rate: float = 0.001,
        device: str = "auto",
        normalize_observations: bool = True
    ):
        """Initialize neural network estimator.

        Args:
            obs_dim: Observation dimension
            hidden_sizes: List of hidden layer sizes
            discount_factor: Discount factor (gamma)
            activation: Activation function ('relu' or 'tanh')
            learning_rate: Learning rate for optimizer
            device: Device to use ('auto', 'cpu', or 'cuda')
            normalize_observations: Whether to normalize observations using running statistics
        """
        super().__init__(obs_dim, discount_factor, device)

        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.normalize_observations = normalize_observations

        self.value_net = ValueNetwork(obs_dim, hidden_sizes, activation, normalize_observations).to(self.device)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)

    @classmethod
    def from_config(cls, method_config, network_config, obs_dim: int, gamma: float):
        """Create estimator from configuration.

        Args:
            method_config: Method-specific configuration (BaseEstimatorConfig subclass)
            network_config: Network configuration
            obs_dim: Observation dimension
            gamma: Discount factor (from training config, shared by all methods)

        Returns:
            Estimator instance
        """
        # Common parameters
        common_params = {
            'obs_dim': obs_dim,
            'hidden_sizes': network_config.hidden_sizes,
            'discount_factor': gamma,
            'activation': network_config.activation,
            'learning_rate': method_config.learning_rate,
            'device': network_config.device,
        }

        # Get method-specific parameters
        specific_params = cls._get_method_specific_params(method_config)

        # Instantiate with all parameters
        return cls(**common_params, **specific_params)

    def train(self):
        """Set estimator to training mode."""
        self.value_net.train()

    def eval(self):
        """Set estimator to evaluation mode."""
        self.value_net.eval()

    def train_step(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step on a mini-batch.

        Args:
            mini_batch: Dictionary containing mini-batch data (torch tensors):
                - observations: (batch_size, obs_dim)
                - next_observations: (batch_size, obs_dim)
                - rewards: (batch_size,)
                - dones: (batch_size,)
                - mc_returns: (batch_size,)

        Returns:
            Dictionary of training metrics
        """
        self.train()

        # Move tensors to device
        obs = mini_batch['observations'].to(self.device)
        mc_returns = mini_batch['mc_returns'].to(self.device)

        # Forward pass
        targets = self.compute_targets(mini_batch)
        values = self.value_net(obs).squeeze(-1)

        # Compute loss and backprop
        loss = nn.functional.mse_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute metrics
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

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict values for given observations.

        Args:
            observations: Array of shape (n, obs_dim)

        Returns:
            Predicted values of shape (n,)
        """
        self.eval()

        with torch.no_grad():
            obs = torch.FloatTensor(observations).to(self.device)
            values = self.value_net(obs).squeeze(-1)
            return values.cpu().numpy()

    def save(self, path: Path):
        """Save estimator to disk.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'obs_dim': self.obs_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'normalize_observations': self.normalize_observations,
            'discount_factor': self.discount_factor,
        }, path)

    def load(self, path: Path):
        """Load estimator from disk.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

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

        # Create estimator instance with saved parameters
        estimator = cls(
            obs_dim=checkpoint['obs_dim'],
            hidden_sizes=checkpoint['hidden_sizes'],
            discount_factor=checkpoint.get('discount_factor', 0.99),  # Default if not saved
            activation=checkpoint['activation'],
            learning_rate=checkpoint['learning_rate'],
            device=device,
            normalize_observations=checkpoint.get('normalize_observations', True)
        )

        # Load state
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
        }


class LeastSquaresEstimator(ValueEstimator):
    """Base class for least squares value estimators using policy representations."""

    def __init__(
        self,
        obs_dim: int,
        discount_factor: float,
        policy_path: str,
        algorithm: str,
        ridge_lambda: float = 1e-6,
        device: str = "auto",
        hidden_sizes: Optional[list] = None,
        activation: str = "relu",
        learning_rate: float = 0.001,
        normalize_observations: bool = False,
        use_pca_projection: bool = False,
        n_components: Optional[int] = None
    ):
        """Initialize least squares estimator.

        Args:
            obs_dim: Observation dimension
            discount_factor: Discount factor (gamma)
            policy_path: Path to trained policy (.zip file)
            algorithm: Policy algorithm (PPO, A2C, SAC, TD3)
            ridge_lambda: Ridge regularization parameter
            device: Device to use
            hidden_sizes: Not used (kept for compatibility)
            activation: Not used (kept for compatibility)
            learning_rate: Not used (closed-form solution)
            normalize_observations: Not used
            use_pca_projection: Whether to project features onto top-k eigenvectors
            n_components: Number of eigenvectors to keep (if use_pca_projection=True)
        """
        super().__init__(obs_dim, discount_factor, device)

        self.hidden_sizes = hidden_sizes or []
        self.activation = activation
        self.learning_rate = learning_rate
        self.normalize_observations = normalize_observations
        self.ridge_lambda = ridge_lambda
        self.algorithm = algorithm
        self.policy_path = policy_path
        self.use_pca_projection = use_pca_projection
        self.n_components = n_components

        # Load policy and create representation extractor
        if policy_path is None:
            raise ValueError(f"policy_path is required for {self.__class__.__name__}")

        from src.policy_representations import load_policy_representation_extractor
        policy_path = Path(policy_path)
        self.repr_extractor = load_policy_representation_extractor(policy_path, algorithm, device)
        self.repr_dim = self.repr_extractor.output_dim

        # PCA projection matrices (set during fit_pca_projection or load)
        self.pca_mean = None
        self.pca_components = None  # Orthonormal basis of top-k eigenvectors

        # Working dimension: repr_dim if no PCA, or projected dim if PCA enabled
        # If PCA enabled, this will be set during fit_pca_projection
        if use_pca_projection:
            self.working_dim = None  # Will be set when PCA is fitted
            self.value_net = None    # Will be created after PCA is fitted
        else:
            self.working_dim = self.repr_dim
            self._initialize_value_net()  # Create immediately

        self.optimizer = None

    def _initialize_value_net(self):
        """Initialize value network and least squares matrices using working_dim.

        Should be called after working_dim is set (either in __init__ or after PCA fitting).
        """
        if self.working_dim is None:
            raise ValueError("working_dim must be set before initializing value network")

        print(f"Initializing value network with working dimension: {self.working_dim}")

        # Create ValueNetwork with no hidden layers (just input -> output linear layer)
        self.value_net = ValueNetwork(
            self.working_dim,
            hidden_sizes=[],
            activation='relu',
            normalize_observations=False
        ).to(self.device)

        # Initialize weights and bias to zero (single linear layer)
        linear_layer = self.value_net.network[0]
        torch.nn.init.zeros_(linear_layer.weight)
        torch.nn.init.zeros_(linear_layer.bias)

        # Initialize least squares matrices: A = Φ^T Φ + λI, b = Φ^T y
        self.d = self.working_dim + 1  # +1 for bias
        self.A = self.ridge_lambda * torch.eye(self.d, device=self.device)
        self.b = torch.zeros(self.d, 1, device=self.device)
        self.w = torch.zeros(self.d, 1, device=self.device)

    def _get_features(self, observations: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        """Extract features from observations: repr -> PCA projection (if enabled) -> bias (if requested).

        This method handles all PCA logic in one place. Child classes don't need to know about PCA.

        Args:
            observations: (batch_size, obs_dim) tensor
            add_bias: Whether to add bias column (default: True for least squares updates)

        Returns:
            features: (batch_size, working_dim) or (batch_size, working_dim+1) with bias
        """
        # Extract representations from policy network
        representations = self.repr_extractor(observations)

        # Apply PCA projection if enabled
        if self.use_pca_projection and self.pca_components is not None:
            centered = representations - self.pca_mean
            representations = centered @ self.pca_components

        # Add bias column if requested
        if add_bias:
            bias = torch.ones(representations.shape[0], 1, device=self.device)
            return torch.cat([representations, bias], dim=1)

        return representations

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
        # Common parameters
        common_params = {
            'obs_dim': obs_dim,
            'discount_factor': gamma,
            'device': network_config.device,
            'hidden_sizes': network_config.hidden_sizes,
            'activation': network_config.activation,
            'learning_rate': method_config.learning_rate,
        }

        # Get method-specific parameters
        specific_params = cls._get_method_specific_params(method_config)

        # Instantiate with all parameters
        return cls(**common_params, **specific_params)

    def train(self):
        """Least squares estimators don't have train mode."""
        pass

    def eval(self):
        """Least squares estimators don't have eval mode."""
        pass

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

        self.repr_extractor.eval()

        with torch.no_grad():
            obs = torch.FloatTensor(preprocess_batch['observations']).to(self.device)
            representations = self.repr_extractor(obs)  # (n_samples, repr_dim)

            # Center the data
            self.pca_mean = representations.mean(dim=0, keepdim=True)  # (1, repr_dim)
            centered = representations - self.pca_mean  # (n_samples, repr_dim)

            # Compute covariance matrix: C = (1/n) * X^T X
            n_samples = centered.shape[0]
            cov_matrix = (centered.T @ centered) / n_samples  # (repr_dim, repr_dim)

            # Eigendecomposition (symmetric matrix)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

            # Sort by descending eigenvalues
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            # Keep top k components (already orthonormal from eigh)
            k = min(self.n_components, self.repr_dim)
            self.pca_components = eigenvectors[:, :k]  # (repr_dim, k)

            # Set working dimension and initialize value network
            self.working_dim = k
            self._initialize_value_net()

            print(f"PCA projection fitted: {self.repr_dim} -> {k} dimensions")
            print(f"Explained variance ratio (top {k}): {eigenvalues[:k].sum() / eigenvalues.sum():.4f}")

    @abstractmethod
    def _update_A_and_b(self, mini_batch: Dict[str, torch.Tensor], phi: torch.Tensor) -> None:
        """Update A and b matrices based on mini-batch data.

        Subclasses implement specific update logic (MC vs TD).

        Args:
            mini_batch: Dictionary containing mini-batch data
            phi: (batch_size, repr_dim+1) representations with bias
        """
        pass

    @abstractmethod
    def _compute_targets_for_metrics(self, mini_batch: Dict[str, torch.Tensor], phi: torch.Tensor) -> torch.Tensor:
        """Compute targets for metric evaluation (not used in update).

        Args:
            mini_batch: Dictionary containing mini-batch data
            phi: (batch_size, repr_dim+1) representations with bias

        Returns:
            Target values for computing loss/MAE metrics
        """
        pass

    def train_step(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Incremental update by accumulating A and b."""
        self.repr_extractor.eval()

        with torch.no_grad():
            obs = mini_batch['observations'].to(self.device)

            # Extract features with bias (handles PCA projection automatically)
            phi = self._get_features(obs, add_bias=True)  # (batch_size, working_dim+1)

            # Update A and b (method-specific)
            self._update_A_and_b(mini_batch, phi)

            # Solve: A w = b (with fallback for singular matrices)
            try:
                self.w = torch.linalg.solve(self.A, self.b)
            except torch._C._LinAlgError:
                # Matrix is singular, use pseudoinverse as fallback
                self.w = torch.linalg.lstsq(self.A, self.b, rcond=None).solution

            # Update value_net weights with the solved w
            # w is (working_dim+1, 1): first working_dim elements are weights, last is bias
            linear_layer = self.value_net.network[0]
            linear_layer.weight.data = self.w[:-1].T  # (1, working_dim)
            linear_layer.bias.data = self.w[-1]       # (1,)

            # Compute predictions and targets for metrics using value_net
            features = self._get_features(obs, add_bias=False)  # (batch_size, working_dim)
            values = self.value_net(features).squeeze(-1)
            targets = self._compute_targets_for_metrics(mini_batch, phi)

            loss = torch.nn.functional.mse_loss(values, targets)
            mae = torch.abs(values - targets).mean()

            # Also compute MC loss if available
            mc_returns = mini_batch.get('mc_returns')
            if mc_returns is not None:
                mc_returns = mc_returns.to(self.device)
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

            # Sort by absolute value
            abs_eigenvalues = torch.abs(eigenvalues)
            sorted_indices = torch.argsort(abs_eigenvalues, descending=True)
            sorted_eigs = eigenvalues[sorted_indices]

            # Get top 5 and bottom 5 by absolute value
            top_5 = sorted_eigs[:5].cpu().numpy()
            bottom_5 = sorted_eigs[-5:].cpu().numpy()

            return {
                'min_eigenvalue': min_eig,
                'max_eigenvalue': max_eig,
                'condition_number': condition_num,
                'top_5_abs': top_5,
                'bottom_5_abs': bottom_5,
            }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict values for given observations."""
        self.repr_extractor.eval()
        self.value_net.eval()

        with torch.no_grad():
            obs = torch.FloatTensor(observations).to(self.device)

            # Extract features without bias (handles PCA projection automatically)
            features = self._get_features(obs, add_bias=False)

            values = self.value_net(features).squeeze(-1)

            return values.cpu().numpy()

    def save(self, path: Path):
        """Save estimator to disk."""
        torch.save({
            'value_net_state_dict': self.value_net.state_dict(),
            'repr_extractor_state_dict': self.repr_extractor.state_dict(),
            'vec_normalize_stats': self.repr_extractor.vec_normalize_stats,
            'A': self.A,
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
            'policy_path': getattr(self, 'policy_path', None),
            'use_pca_projection': self.use_pca_projection,
            'n_components': self.n_components,
            'pca_mean': self.pca_mean,
            'pca_components': self.pca_components,
            'working_dim': self.working_dim,
        }, path)

    def load(self, path: Path):
        """Load estimator from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load PCA settings first
        self.use_pca_projection = checkpoint.get('use_pca_projection', False)
        self.n_components = checkpoint.get('n_components', None)
        self.pca_mean = checkpoint.get('pca_mean', None)
        self.pca_components = checkpoint.get('pca_components', None)
        self.working_dim = checkpoint.get('working_dim', self.repr_dim)

        # Get the saved feature dimension
        saved_feature_dim = checkpoint['value_net_state_dict']['network.0.weight'].shape[1]

        # Check if we need to recreate value network
        if self.value_net is None:
            # Value net hasn't been created yet (use_pca_projection=True case)
            self.value_net = ValueNetwork(saved_feature_dim, hidden_sizes=[], activation='relu', normalize_observations=False).to(self.device)
            self.d = saved_feature_dim + 1  # +1 for bias in least squares matrices
        else:
            # Value net exists, check if dimensions match
            current_feature_dim = self.value_net.network[0].weight.shape[1]
            if saved_feature_dim != current_feature_dim:
                print(f"Recreating value network: saved dim {saved_feature_dim} != current dim {current_feature_dim}")
                self.value_net = ValueNetwork(saved_feature_dim, hidden_sizes=[], activation='relu', normalize_observations=False).to(self.device)
                self.d = saved_feature_dim + 1  # +1 for bias in least squares matrices

        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.repr_extractor.load_state_dict(checkpoint['repr_extractor_state_dict'])

        # Restore VecNormalize stats if they exist
        vec_normalize_stats = checkpoint.get('vec_normalize_stats', None)
        if vec_normalize_stats is not None:
            self.repr_extractor.vec_normalize_stats = vec_normalize_stats

        self.A = checkpoint['A']
        self.b = checkpoint['b']
        self.w = checkpoint['w']
        self.training_step = checkpoint['training_step']
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

        # Extract policy_path from the checkpoint - need to reconstruct it
        # This is a limitation: we need the policy_path to be saved in the checkpoint
        policy_path = checkpoint.get('policy_path')
        if policy_path is None:
            raise ValueError("Checkpoint does not contain 'policy_path'. Cannot load LeastSquaresEstimator.")

        # Create estimator instance with saved parameters
        estimator = cls(
            obs_dim=checkpoint['obs_dim'],
            discount_factor=checkpoint.get('discount_factor', 0.99),
            policy_path=policy_path,
            algorithm=checkpoint['algorithm'],
            ridge_lambda=checkpoint.get('ridge_lambda', 1e-6),
            device=device,
            hidden_sizes=checkpoint.get('hidden_sizes', []),
            activation=checkpoint.get('activation', 'relu'),
            learning_rate=checkpoint.get('learning_rate', 0.001),
            normalize_observations=checkpoint.get('normalize_observations', False),
            use_pca_projection=checkpoint.get('use_pca_projection', False),
            n_components=checkpoint.get('n_components', None)
        )

        # Load state
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
            'algorithm': self.algorithm,
        }
