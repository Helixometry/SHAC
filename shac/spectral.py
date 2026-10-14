"""Spectral feature extraction for heart sounds."""

import torch
import torch.nn as nn
import numpy as np
from scipy.fftpack import fft
from librosa.feature import mfcc, melspectrogram


class SpectralFeatureExtractor(nn.Module):
    """
    Extract MFCC and LFCC features from heart sound recordings.

    Args:
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients
        n_fft (int): FFT size
        hop_length (int): Hop length for STFT
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        super().__init__()
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features.

        Args:
            audio: Audio tensor (batch, sample_length)

        Returns:
            MFCC features (batch, n_mfcc, time_steps)
        """
        # Convert to numpy for librosa
        audio_np = audio.cpu().numpy()

        batch_size = audio_np.shape[0]
        features_list = []

        for i in range(batch_size):
            # Extract MFCC
            feat = mfcc(
                y=audio_np[i],
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            features_list.append(feat)

        # Stack and convert to tensor
        features = torch.FloatTensor(np.array(features_list))

        return features.to(audio.device)


class LFCCExtractor(nn.Module):
    """
    Extract Log Filterbank Energy Coefficients (LFCC).

    Args:
        sr (int): Sample rate
        n_mels (int): Number of mel bands
        n_fft (int): FFT size
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 13,
        n_fft: int = 2048,
    ):
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract LFCC features.

        Args:
            audio: Audio tensor (batch, sample_length)

        Returns:
            LFCC features (batch, n_mels, time_steps)
        """
        # Convert to numpy
        audio_np = audio.cpu().numpy()

        batch_size = audio_np.shape[0]
        features_list = []

        for i in range(batch_size):
            # Extract melspectrogram
            mel_spec = melspectrogram(
                y=audio_np[i],
                sr=self.sr,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
            )
            # Log scale
            lfcc = np.log(mel_spec + 1e-9)
            features_list.append(lfcc)

        # Stack and convert to tensor
        features = torch.FloatTensor(np.array(features_list))

        return features.to(audio.device)
