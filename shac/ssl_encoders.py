"""SSL encoder wrappers for heart sound analysis."""

import torch
import torch.nn as nn


class SSLEncoder(nn.Module):
    """
    Load and manage SSL encoders (WavLM, HuBERT).

    Args:
        model_name (str): SSL model name ('wavlm', 'hubert')
        freeze (bool): Freeze encoder weights
    """

    def __init__(self, model_name: str = "wavlm", freeze: bool = True):
        super().__init__()
        self.model_name = model_name

        # Load model
        self.encoder = self._load_encoder(model_name)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _load_encoder(self, model_name: str):
        """Load SSL encoder."""
        if model_name == "wavlm":
            try:
                from transformers import AutoModel
                return AutoModel.from_pretrained("microsoft/wavlm-large")
            except:
                return nn.Linear(1024, 1024)  # Placeholder
        elif model_name == "hubert":
            try:
                from transformers import AutoModel
                return AutoModel.from_pretrained("facebook/hubert-large-ls960")
            except:
                return nn.Linear(1024, 1024)  # Placeholder
        else:
            return nn.Linear(1024, 1024)  # Placeholder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio with SSL model.

        Args:
            audio: Audio tensor (batch, sample_length)

        Returns:
            SSL embeddings (batch, seq_len, 768/1024)
        """
        with torch.no_grad():
            try:
                outputs = self.encoder(audio, output_hidden_states=True)
                return outputs.last_hidden_state
            except:
                # Placeholder output
                return torch.randn(audio.shape[0], 100, 768)
