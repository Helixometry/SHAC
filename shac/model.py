"""GROOT model for synthetic heart sound detection."""

import torch
import torch.nn as nn
from typing import Dict, Optional


class GROOT(nn.Module):
    """
    GROOT: Spectral-SSL fusion for synthetic heart sound detection.

    Combines spectral features (MFCC, LFCC) with self-supervised learning
    features (WavLM, HuBERT) for robust cardiac sound authentication.

    Args:
        spectral_dim (int): Spectral feature dimension (e.g., 13 for MFCC)
        ssl_dim (int): SSL feature dimension (e.g., 768 for WavLM)
        hidden_dim (int): Fusion hidden dimension
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        spectral_dim: int = 13,
        ssl_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.spectral_dim = spectral_dim
        self.ssl_dim = ssl_dim
        self.hidden_dim = hidden_dim

        # Projection layers
        self.spectral_proj = nn.Linear(spectral_dim, hidden_dim)
        self.ssl_proj = nn.Linear(ssl_dim, hidden_dim)

        # Spectral-SSL fusion
        from .fusion import SpectralSSLFusion
        self.fusion = SpectralSSLFusion(hidden_dim, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # Binary: Real/Synthetic
        )

    def forward(
        self,
        spectral_features: torch.Tensor,
        ssl_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GROOT.

        Args:
            spectral_features: Spectral features (batch, seq_len, spectral_dim)
            ssl_features: SSL features (batch, seq_len, ssl_dim)

        Returns:
            Dictionary with logits and probabilities
        """
        # Project to common space
        spectral_proj = self.spectral_proj(spectral_features)
        ssl_proj = self.ssl_proj(ssl_features)

        # Spectral-SSL fusion
        fused = self.fusion(spectral_proj, ssl_proj)

        # Classification
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "real_prob": probs[:, 0],
            "synthetic_prob": probs[:, 1],
            "features": fused,
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "GROOT":
        """Load pretrained GROOT model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def save(self, checkpoint_path: str, config: Optional[Dict] = None) -> None:
        """Save GROOT model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {
                "spectral_dim": self.spectral_dim,
                "ssl_dim": self.ssl_dim,
                "hidden_dim": self.hidden_dim,
            },
        }
        torch.save(checkpoint, checkpoint_path)
