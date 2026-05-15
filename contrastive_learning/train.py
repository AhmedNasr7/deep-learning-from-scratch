"""
Contrastive learning training script scaffold.

Usage examples:
    python -m contrastive_learning.train --method simclr --epochs 100
    python -m contrastive_learning.train --method mae --epochs 200
"""

import argparse


def train(method: str, dataset: str, batch_size: int, epochs: int):
    raise NotImplementedError("Contrastive training script is not implemented yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train contrastive learning models")
    parser.add_argument("--method", type=str, default="simclr",
                        choices=["simclr", "mae"], help="Which method to train")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    train(args.method, args.dataset, args.batch_size, args.epochs)
