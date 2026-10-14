"""Training script for SHAC."""

import argparse
import torch
import yaml
from pathlib import Path

from shac.model import GROOT
from shac.utils import seed_everything, get_device


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GROOT model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_everything(config.get('seed', 42))
    device = get_device()

    model = GROOT(**config['model']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Training GROOT on {device}")
    print(f"Config: {config}")


if __name__ == "__main__":
    main()
