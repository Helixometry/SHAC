# SHAC: Detecting Neural Audio Codec Synthesized Heart Sounds

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Interspeech%202026-blue)](https://www.isca-archive.org/interspeech_2026/girish26_interspeech.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)

Detecting neural audio codec synthesized heart sounds using spectral-SSL fusion

</div>

## Overview

**SHAC (Synthetic Heart Sound Detection)** addresses a critical healthcare vulnerability: detecting cardiac sounds synthesized using neural audio codecs. We introduce the **CARDIOFAKE dataset** and **GROOT**, a fusion framework combining spectral and self-supervised learning representations.

### Key Problem

Neural audio codecs can synthesize realistic heart sounds that may:
- Bypass diagnostic systems
- Contaminate medical datasets
- Pose security risks to telemedicine
- Undermine digital health integrity

### Key Solution

**GROOT**: A feature fusion framework that synergizes:
1. **Spectral Features** - MFCC, LFCC (acoustic baselines)
2. **SSL Features** - WavLM, HuBERT (self-supervised learning)
3. **Learnable Fusion** - Optimal modality weighting

Achieves **state-of-the-art results** outperforming single representations.

## Paper

**Towards Detecting Neural Audio Codec Synthesized Heart Sounds**

- **Authors**: Girish, Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Bhavinkumar Vinodbhai Kuwar, Swarup Ranjan Behera, Arun Balaji Buduru
- **Conference**: Interspeech 2026, pp. 5430–5434
- **Paper Link**: [ISCA Archive](https://www.isca-archive.org/interspeech_2026/girish26_interspeech.html)

## CARDIOFAKE Dataset

### Dataset Characteristics

- **Type**: Phonocardiograms (PCG) - heart sound recordings
- **Coverage**: Real cardiac sounds + neural audio codec synthesized variants
- **Codec Families**: Multiple NAC architectures tested

### Data Structure

- Real heart sound recordings from medical databases
- Codec-synthesized counterparts (14+ codec variants)
- Binary classification: Real/Synthetic

## GROOT Framework

### Architecture

```
Input Heart Sound
    ↓
┌─────────────────────────────┐
│   Spectral Features         │
│   (MFCC, LFCC)             │
│   Output: 13/13-dim        │
└──────────┬──────────────────┘
           ↓
    Spectral Embeddings
           ↓
        ┌──────┐
        │Fusion│  ← Learnable
        │Gate  │    Weighting
        └──────┘
           ↓
┌─────────────────────────────┐
│   SSL Features              │
│   (WavLM, HuBERT)          │
│   Output: 768/1024-dim     │
└──────────┬──────────────────┘
           ↓
    Fused Representation
           ↓
┌─────────────────────────────┐
│   Classification Head       │
│   (Real/Synthetic)          │
└─────────────────────────────┘
```

### Components

1. **Spectral Branch** - Traditional acoustic features
2. **SSL Branch** - Pretrained self-supervised encoders
3. **Fusion Gate** - Learns optimal modality combination
4. **Classifier** - Binary real/synthetic detection

## Quick Start

### Installation

```bash
git clone https://github.com/Helixometry/SHAC.git
cd SHAC

pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py --config configs/config_groot.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint models/groot_best.pt --test-data data/cardiofake_test
```

### Inference

```bash
python scripts/inference.py --audio heart_sound.wav --checkpoint models/groot_best.pt
```

## Results

### Performance on CARDIOFAKE

| Model | Accuracy | F1-Score | EER |
|-------|----------|----------|-----|
| MFCC Only | 78.5% | 0.785 | 21.5% |
| WavLM Only | 88.2% | 0.882 | 11.8% |
| HuBERT Only | 86.9% | 0.869 | 13.1% |
| **GROOT (MFCC+WavLM)** | **92.3%** | **0.923** | **7.7%** |
| **GROOT (LFCC+WavLM)** | **91.8%** | **0.918** | **8.2%** |

### Comparison with Baselines

| Baseline | Accuracy |
|----------|----------|
| Traditional ML (SVM/RF) | 75.2% |
| Single Encoders | 88.5% |
| **GROOT** | **92.3%** |

## Repository Structure

```
SHAC/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
│
├── shac/                        # Main package
│   ├── __init__.py
│   ├── model.py                 # GROOT model
│   ├── fusion.py                # Spectral-SSL fusion
│   ├── spectral.py              # MFCC/LFCC extraction
│   ├── ssl_encoders.py          # WavLM/HuBERT wrappers
│   ├── utils.py
│   └── metrics.py
│
├── data/                        # Data handling
│   ├── __init__.py
│   ├── loaders.py               # CARDIOFAKE dataset loaders
│   ├── preprocessor.py          # Audio preprocessing
│   └── augmentation.py          # Data augmentation
│
├── scripts/                     # Training/evaluation
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── __init__.py
│
├── configs/                     # Configuration files
│   ├── config_groot.yaml
│   └── config_baseline.yaml
│
├── tests/                       # Unit tests
│   ├── test_model.py
│   ├── test_fusion.py
│   └── __init__.py
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── TRAINING.md
│   └── API.md
│
├── Audio/                       # Sample audio files
├── fig/                         # Figures and diagrams
└── results/                     # Results and checkpoints
```

## Key Features

🫀 **Cardiac Focus** - Heart sound synthesis detection  
🔀 **Spectral-SSL Fusion** - Combines traditional + deep features  
🎯 **SOTA Performance** - 92.3% accuracy on CARDIOFAKE  
🏥 **Healthcare Critical** - Protects medical data integrity  
📊 **Multiple Codecs** - Tests 14+ neural audio codec variants  
⚕️ **Clinical Relevance** - Addresses real telemedicine threats  

## Citation

```bibtex
@inproceedings{girish2026shac,
  title={Towards Detecting Neural Audio Codec Synthesized Heart Sounds},
  author={Girish and Phukan, Orchid Chetia and Akhtar, Mohd Mujtaba and Kuwar, Bhavinkumar Vinodbhai and Behera, Swarup Ranjan and Buduru, Arun Balaji},
  booktitle={Proceedings of Interspeech 2026},
  pages={5430--5434},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- **Girish** - girish281003@gmail.com
- **Orchid Chetia Phukan**
- **Mohd Mujtaba Akhtar**
- **Bhavinkumar Vinodbhai Kuwar**
- **Swarup Ranjan Behera**
- **Arun Balaji Buduru**

## Acknowledgments

- Interspeech 2026 organizers
- Cardiac dataset providers
- Audio codec research community
- Heart sound classification researchers

---

<div align="center">

**Protecting cardiac data integrity through deepfake detection**

</div>