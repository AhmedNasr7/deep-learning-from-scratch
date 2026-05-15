"""
Training script for Vision Transformer.

Usage:
    python -m vit.train --arch vit_tiny --dataset cifar10 --epochs 50
    python -m vit.train --arch vit_small --dataset tiny-imagenet --epochs 100
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import build_vit
from .data import get_cifar10_loaders, get_tiny_imagenet_loaders


def train(arch: str, dataset: str, epochs: int, batch_size: int, lr: float):
    """Main training loop for ViT."""

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    if dataset == "cifar10":
        train_loader, val_loader = get_cifar10_loaders(batch_size)
        num_classes, img_size = 10, 32
    elif dataset == "tiny-imagenet":
        train_loader, val_loader = get_tiny_imagenet_loaders(batch_size)
        num_classes, img_size = 200, 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # --- Model ---
    model = build_vit(arch, num_classes=num_classes, img_size=img_size).to(device)
    print(f"Model: {arch} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Optimizer + Scheduler ---
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # --- Training ---
    best_acc = 0.0
    save_dir = os.path.join("checkpoints", f"{arch}_{dataset}")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        train_acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            print(f"  → Best: {best_acc:.2f}%")

    print(f"Done. Best val accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT")
    parser.add_argument("--arch", type=str, default="vit_tiny", choices=["vit_tiny", "vit_small", "vit_base", "vit_large", "vit_huge"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "tiny-imagenet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train(args.arch, args.dataset, args.epochs, args.batch_size, args.lr)
