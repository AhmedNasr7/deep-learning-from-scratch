"""
Training script for Vision Transformer.

Usage:
    python -m vit.train --arch vit_tiny --dataset cifar10 --epochs 100
    python -m vit.train --arch vit_small --dataset tiny-imagenet --epochs 100
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import build_vit
from .data import get_cifar10_loaders, get_tiny_imagenet_loaders

# ── Dataset normalization stats (for visualization denormalization) ───────────
NORM_STATS = {
    "cifar10":       dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "tiny-imagenet": dict(mean=(0.485,  0.456,  0.406),  std=(0.229,  0.224,  0.225)),
}


def _denorm(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """Reverse normalization and convert to HWC numpy for plotting."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (tensor.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def save_metrics_plot(train_losses, train_accs, val_accs, save_dir: str):
    """Save a dual-panel loss + accuracy chart to disk."""
    epochs_done = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("ViT Training Metrics", fontsize=13, fontweight="bold")

    ax1.plot(epochs_done, train_losses, "#e74c3c", lw=2, label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Cross-Entropy Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs_done, train_accs, "#3498db", lw=2, ls="--", label="Train Acc")
    ax2.plot(epochs_done, val_accs,   "#2ecc71", lw=2, label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    path = os.path.join(save_dir, "metrics.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Metrics plot saved → {path}")


def save_attention_maps(model, val_loader, device, img_size, dataset, save_dir: str):
    """Save CLS-token attention heatmaps from the last encoder block."""
    model.eval()
    stats = NORM_STATS[dataset]

    img_tensor, gt_labels = next(iter(val_loader))
    num_viz = min(4, img_tensor.size(0))
    n_heads = model.blocks[-1].attn.n_heads
    gs = img_size // 4  # grid size (patch_size=4 for tiny, 8 for small, etc.)

    fig, axes = plt.subplots(num_viz, n_heads + 1,
                             figsize=(2.5 * (n_heads + 1), 2.5 * num_viz))
    fig.suptitle("CLS Attention Maps — Last Block", fontsize=12, fontweight="bold")

    for row in range(num_viz):
        single = img_tensor[row:row+1].to(device)
        with torch.no_grad():
            _ = model(single)
            attn_w = model.blocks[-1].attn.attention.attn_weights  # (1, H, S, S)

        cls_attn = attn_w[0, :, 0, 1:].cpu()  # (n_heads, num_patches)
        gs_actual = int(cls_attn.shape[1] ** 0.5)
        orig = _denorm(img_tensor[row], **stats)

        row_axes = axes[row] if num_viz > 1 else axes
        row_axes[0].imshow(orig, interpolation="bilinear")
        row_axes[0].set_title(f"GT: {gt_labels[row].item()}", fontsize=8, fontweight="bold")
        row_axes[0].axis("off")

        for h in range(n_heads):
            attn_map = cls_attn[h].reshape(1, 1, gs_actual, gs_actual).float()
            upscaled = F.interpolate(attn_map, size=(img_size, img_size),
                                     mode="bicubic", align_corners=False).squeeze().numpy()
            row_axes[h+1].imshow(orig, interpolation="bilinear")
            row_axes[h+1].imshow(upscaled, alpha=0.55, cmap="jet",
                                 vmin=upscaled.min(), vmax=upscaled.max())
            row_axes[h+1].set_title(f"Head {h+1}", fontsize=8)
            row_axes[h+1].axis("off")

    path = os.path.join(save_dir, "attention_maps.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Attention maps saved → {path}")


def train(arch: str, dataset: str, epochs: int, batch_size: int, lr: float):
    """Main training loop for ViT."""

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if dataset == "cifar10":
        train_loader, val_loader = get_cifar10_loaders(batch_size)
        num_classes, img_size = 10, 32
    elif dataset == "tiny-imagenet":
        train_loader, val_loader = get_tiny_imagenet_loaders(batch_size)
        num_classes, img_size = 200, 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_vit(arch, num_classes=num_classes, img_size=img_size).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Built {arch} — {total_params:,} trainable parameters")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Output dirs ───────────────────────────────────────────────────────────
    save_dir = os.path.join("checkpoints", f"{arch}_{dataset}")
    viz_dir  = os.path.join("visualizations", f"{arch}_{dataset}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(viz_dir,  exist_ok=True)

    best_acc = 0.0
    train_losses, train_accs, val_accs = [], [], []

    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        batch_bar = tqdm(train_loader, desc=f"  Train E{epoch+1}", leave=False, unit="batch")
        for imgs, lbls in batch_bar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss = criterion(logits, lbls)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, preds = logits.max(1)
            total   += lbls.size(0)
            correct += preds.eq(lbls).sum().item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss  = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"  Val   E{epoch+1}", leave=False):
                imgs, lbls = imgs.to(device), lbls.to(device)
                _, preds = model(imgs).max(1)
                val_total   += lbls.size(0)
                val_correct += preds.eq(lbls).sum().item()

        val_acc = 100.0 * val_correct / val_total
        val_accs.append(val_acc)
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"loss={avg_loss:.4f} | train={train_acc:.1f}% | val={val_acc:.1f}%"
        )
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", val=f"{val_acc:.1f}%")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            logger.success(f"New best: {best_acc:.2f}% — checkpoint saved")

        # ── Save visualizations every 10 epochs ───────────────────────────────
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            save_metrics_plot(train_losses, train_accs, val_accs, viz_dir)
            save_attention_maps(model, val_loader, device, img_size, dataset, viz_dir)

    logger.success(f"Training complete! Best val accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT")
    parser.add_argument("--arch",       type=str,   default="vit_tiny",
                        choices=["vit_tiny", "vit_small", "vit_base", "vit_large", "vit_huge"])
    parser.add_argument("--dataset",    type=str,   default="cifar10",
                        choices=["cifar10", "tiny-imagenet"])
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    args = parser.parse_args()

    train(args.arch, args.dataset, args.epochs, args.batch_size, args.lr)
