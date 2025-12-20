"""Extract frozen representations from trained policy networks."""

import torch
import torch.nn as nn
from pathlib import Path
from stable_baselines3 import PPO, A2C, SAC, TD3


ALGORITHM_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


class PolicyRepresentationExtractor(nn.Module):
    """Extracts frozen representations from policy network's last hidden layer."""

    def __init__(self, policy_model, algorithm: str):
        super().__init__()
        self.algorithm = algorithm

        # Extract the policy network and freeze it
        if algorithm in ["PPO", "A2C"]:
            # For on-policy algorithms, extract the shared features extractor
            self.features_extractor = policy_model.policy.features_extractor
            self.mlp_extractor = policy_model.policy.mlp_extractor

            # Get the last hidden layer dimension (policy latent)
            self.output_dim = self.mlp_extractor.latent_dim_pi

        elif algorithm in ["SAC", "TD3"]:
            # For off-policy algorithms, extract actor network
            self.features_extractor = policy_model.policy.actor.features_extractor
            self.latent_pi = policy_model.policy.actor.latent_pi

            # Get dimension of last hidden layer
            self.output_dim = self.latent_pi[-2].out_features if len(self.latent_pi) > 1 else self.features_extractor.features_dim
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract representation from observations.

        Args:
            obs: (batch_size, obs_dim)

        Returns:
            representations: (batch_size, repr_dim)
        """
        with torch.no_grad():
            if self.algorithm in ["PPO", "A2C"]:
                # Extract features
                features = self.features_extractor(obs)
                # Get policy latent (last hidden layer before action head)
                latent_pi, _ = self.mlp_extractor(features)
                return latent_pi
            else:  # SAC, TD3
                features = self.features_extractor(obs)
                # Pass through all but the last layer
                for layer in self.latent_pi[:-1]:
                    features = layer(features)
                return features


def load_policy_representation_extractor(policy_path: Path, algorithm: str, device: str = "auto") -> PolicyRepresentationExtractor:
    """Load a policy and create a frozen representation extractor.

    Args:
        policy_path: Path to trained policy (.zip file)
        algorithm: Policy algorithm (PPO, A2C, SAC, TD3)
        device: Device to use ('auto', 'cpu', or 'cuda')

    Returns:
        PolicyRepresentationExtractor instance
    """
    if not policy_path.exists():
        raise ValueError(f"Policy not found at {policy_path}")

    if algorithm not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHM_MAP.keys())}")

    # Load policy
    AlgorithmClass = ALGORITHM_MAP[algorithm]
    policy_model = AlgorithmClass.load(policy_path)

    # Create extractor
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    extractor = PolicyRepresentationExtractor(policy_model, algorithm).to(device)
    return extractor
