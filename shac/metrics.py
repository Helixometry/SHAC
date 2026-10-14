"""Evaluation metrics for SHAC."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "f1": f1_score(targets, predictions),
        "precision": precision_score(targets, predictions),
        "recall": recall_score(targets, predictions),
    }

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print metrics."""
    print("\nEvaluation Metrics")
    print("-" * 60)
    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name:20s}: {value:.4f}")
