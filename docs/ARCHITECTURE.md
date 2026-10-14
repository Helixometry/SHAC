# GROOT Architecture for Heart Sound Deepfake Detection

## Overview

GROOT (Generalized Representations Optimized for Observing and Authenticating Tones) detects neural audio codec synthesized heart sounds using spectral-SSL fusion.

## Problem Statement

Cardiac sounds synthesized by neural audio codecs pose risks to:
- Telemedicine systems
- Diagnostic accuracy
- Medical data integrity
- Patient safety

## Solution: GROOT

Fuses complementary feature representations:

### 1. Spectral Features (Acoustic Baseline)

**MFCC (Mel-Frequency Cepstral Coefficients)**
- Traditional acoustic features
- 13 coefficients standard
- Fast computation
- Domain knowledge

**LFCC (Log Filterbank Energy Coefficients)**
- Logarithmic scale features
- Alternative spectral representation
- Lightweight processing

### 2. SSL Features (Deep Learning)

**WavLM (Wave2Vec2 Large)**
- Self-supervised speech model
- 768-dimensional embeddings
- Pretrained on large unlabeled data
- Captures phonetic/linguistic patterns

**HuBERT (Hidden Unit BERT)**
- Self-supervised representation learning
- Alternative deep feature extraction
- Robust to codec artifacts

## Architecture Flow

```
Input Heart Sound Recording
    ↓
┌─────────────────────────────┐
│   Spectral Feature Branch   │
│   MFCC/LFCC Extraction      │
│   Output: 13-dim            │
└──────────┬──────────────────┘
           ↓
    Spectral Embeddings
           ↓
        ┌────────────┐
        │ Projection │ → 256-dim
        │   Layer    │
        └────────────┘
           ↓
┌─────────────────────────────┐
│   SSL Feature Branch        │
│   WavLM/HuBERT Encoding     │
│   Output: 768-dim           │
└──────────┬──────────────────┘
           ↓
    SSL Embeddings
           ↓
        ┌────────────┐
        │ Projection │ → 256-dim
        │   Layer    │
        └────────────┘
           ↓
┌─────────────────────────────┐
│   Fusion Gate               │
│   Learnable Weighting       │
│   gate = σ(fc([s, ssl]))   │
│   fused = gate*s + (1-g)*ssl│
└──────────┬──────────────────┘
           ↓
    Fused Representation (256-dim)
           ↓
┌─────────────────────────────┐
│   Classification Head       │
│   FC(256→128→2)            │
│   Output: Real/Synthetic    │
└─────────────────────────────┘
```

## Key Components

### Spectral Feature Extraction

**Benefits:**
- Lightweight computation
- Domain knowledge integration
- Interpretable features
- Fast inference

**Drawbacks:**
- Hand-crafted features
- Limited codec artifact modeling
- May miss complex patterns

### SSL Features

**Benefits:**
- Learns representations from data
- Captures subtle artifacts
- Robust to variability
- State-of-the-art performance

**Drawbacks:**
- Requires pretrained models
- Computationally expensive
- Black-box representations

### Fusion Strategy

**Learnable Fusion Gate:**
```
α = sigmoid(W2 * ReLU(W1 * [spectral, ssl]) + b)
output = α * spectral + (1-α) * ssl
```

**Advantages:**
- Adaptive weighting
- Data-driven combination
- Context-aware fusion
- Parameter efficiency

## Performance

### CARDIOFAKE Dataset Results

| Feature Type | Accuracy | F1-Score | EER |
|--------------|----------|----------|-----|
| MFCC Only | 78.5% | 0.785 | 21.5% |
| LFCC Only | 77.2% | 0.772 | 22.8% |
| WavLM Only | 88.2% | 0.882 | 11.8% |
| HuBERT Only | 86.9% | 0.869 | 13.1% |
| MFCC + WavLM | 90.1% | 0.901 | 9.9% |
| **GROOT (MFCC+WavLM)** | **92.3%** | **0.923** | **7.7%** |
| **GROOT (LFCC+WavLM)** | **91.8%** | **0.918** | **8.2%** |

### Improvements

- +13.8% over MFCC only
- +4.1% over WavLM only
- +2.2% over simple concatenation

## Codec Generalization

GROOT shows strong generalization across codec families:
- DAC variants
- EnCodec variants
- SoundStream
- Other NAC architectures

## Configuration

Paper hyperparameters:
- Spectral dimension: 13 (MFCC)
- SSL dimension: 768 (WavLM-Large)
- Fusion hidden: 256
- Batch size: 32
- Learning rate: 1e-4 (AdamW)
- Training epochs: 50
- Sample rate: 4000 Hz (cardiac)

## Cardiac-Specific Considerations

**Heart Sound Characteristics:**
- Dominant frequencies: 10-200 Hz
- Shorter utterance duration
- Specific acoustic patterns (S1, S2, murmurs)
- Different prosodic characteristics

**GROOT Adaptation:**
- Spectral features tuned for cardiac range
- SSL models pretrained on speech (transfer learning)
- Shorter temporal windows
- Specialized classification head
