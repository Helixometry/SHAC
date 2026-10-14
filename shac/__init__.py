"""SHAC: Detecting Neural Audio Codec Synthesized Heart Sounds"""

__version__ = "1.0.0"
__author__ = "Girish, Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Bhavinkumar Vinodbhai Kuwar, Swarup Ranjan Behera, Arun Balaji Buduru"
__email__ = "girish281003@gmail.com"

from .model import GROOT
from .fusion import SpectralSSLFusion
from .spectral import SpectralFeatureExtractor
from .ssl_encoders import SSLEncoder

__all__ = [
    "GROOT",
    "SpectralSSLFusion",
    "SpectralFeatureExtractor",
    "SSLEncoder",
]
