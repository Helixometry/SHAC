"""Inference script for SHAC."""

import argparse
import torch
from pathlib import Path

from shac.model import GROOT
from shac.utils import get_device


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Inference with GROOT")
    parser.add_argument("--audio", type=str, required=True, help="Audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint")

    args = parser.parse_args()

    device = get_device()
    model = GROOT.from_pretrained(args.checkpoint).to(device)

    print(f"\nSHAC Inference Results")
    print("=" * 60)
    print(f"Audio: {Path(args.audio).name}")
    print(f"\n(Placeholder inference)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
