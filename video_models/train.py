"""
Video model training script scaffold.

Usage examples:
    python -m video_models.train --dataset ucf101 --epochs 50
    python -m video_models.train --dataset kinetics --epochs 100
"""

import argparse


def train(dataset: str, batch_size: int, epochs: int):
    raise NotImplementedError("Video model training is not implemented yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train video transformer models")
    parser.add_argument("--dataset", type=str, default="ucf101",
                        help="Video dataset name")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    train(args.dataset, args.batch_size, args.epochs)
