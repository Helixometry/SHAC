"""Spectral-SSL fusion module."""

import torch
import torch.nn as nn


class SpectralSSLFusion(nn.Module):
    """
    Learnable fusion of spectral and SSL features.

    Args:
        hidden_dim (int): Feature dimension
        dropout (float): Dropout probability
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spectral: torch.Tensor,
        ssl: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse spectral and SSL features.

        Args:
            spectral: (batch, seq_len, hidden_dim)
            ssl: (batch, seq_len, hidden_dim)

        Returns:
            Fused representation: (batch, hidden_dim)
        """
        # Average pooling to get utterance-level representations
        spectral_pool = spectral.mean(dim=1)  # (batch, hidden_dim)
        ssl_pool = ssl.mean(dim=1)             # (batch, hidden_dim)

        # Normalize
        spectral_pool = self.norm1(spectral_pool)
        ssl_pool = self.norm2(ssl_pool)

        # Learnable fusion gate
        combined = torch.cat([spectral_pool, ssl_pool], dim=-1)
        gate = self.fusion_gate(combined)

        # Gated combination
        fused = gate * spectral_pool + (1 - gate) * ssl_pool

        return self.dropout(fused)
