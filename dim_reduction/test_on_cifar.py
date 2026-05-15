"""
Dim Reduction Visual Test — CIFAR-10 & Tiny-ImageNet.

Loads images, flattens to pixel vectors, runs PCA → t-SNE → UMAP,
and saves a grid of scatter plots coloured by class.

Usage:
    python -m dim_reduction.test_on_cifar --dataset cifar10 --method pca
    python -m dim_reduction.test_on_cifar --dataset cifar10 --method tsne
    python -m dim_reduction.test_on_cifar --dataset tiny-imagenet --method umap

NOTE: Methods must be implemented before this script will run.
"""

import argparse
import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from dim_reduction.pca import PCA
from dim_reduction.tsne import TSNE
from dim_reduction.umap import UMAP

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def load_cifar10(n_samples: int = 2000) -> tuple:
    """Load a flat subset of CIFAR-10 pixel vectors and labels."""
    transform = T.Compose([T.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                           download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=True)
    images, labels = next(iter(loader))
    # (N, C, H, W) → (N, C*H*W)
    X = images.view(images.size(0), -1)
    return X, labels, CIFAR10_CLASSES


def load_tiny_imagenet(n_samples: int = 2000) -> tuple:
    """Load a flat subset of Tiny-ImageNet via HuggingFace."""
    from datasets import load_dataset
    transform = T.Compose([T.ToTensor()])
    hf = load_dataset("zh-plus/tiny-imagenet")["valid"]

    images, labels = [], []
    for i, item in enumerate(hf):
        if i >= n_samples:
            break
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(transform(img))
        labels.append(item["label"])

    X = torch.stack(images).view(len(images), -1)
    return X, torch.tensor(labels), [str(i) for i in range(200)]


def plot_embedding(Z: np.ndarray, labels: np.ndarray, class_names: list,
                   title: str, save_path: str):
    """Scatter plot of 2D embedding coloured by class label."""
    n_classes = len(set(labels.tolist()))
    cmap = plt.cm.get_cmap("tab10" if n_classes <= 10 else "tab20", n_classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        ax.scatter(Z[mask, 0], Z[mask, 1], s=5, alpha=0.6,
                   color=cmap(cls_idx),
                   label=class_names[cls_idx] if n_classes <= 10 else None)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if n_classes <= 10:
        ax.legend(markerscale=3, loc="best", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved → {save_path}")


def run(dataset: str, method: str, n_samples: int = 2000):
    print(f"Loading {dataset} ({n_samples} samples)...")
    if dataset == "cifar10":
        X, labels, class_names = load_cifar10(n_samples)
    elif dataset == "tiny-imagenet":
        X, labels, class_names = load_tiny_imagenet(n_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Data shape: {X.shape}  →  reducing with {method.upper()}...")

    if method == "pca":
        # First reduce to 50D with PCA (then 2D for visualisation)
        pca_50 = PCA(n_components=50)
        X_50 = pca_50.fit_transform(X)
        reducer = PCA(n_components=2)
        Z = reducer.fit_transform(X_50).numpy()
    elif method == "tsne":
        # Pre-reduce to 50D first (standard practice)
        pca = PCA(n_components=50)
        X_50 = pca.fit_transform(X)
        reducer = TSNE(n_components=2, perplexity=30, n_iter=500)
        Z = reducer.fit_transform(X_50).numpy()
    elif method == "umap":
        pca = PCA(n_components=50)
        X_50 = pca.fit_transform(X)
        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        Z = reducer.fit_transform(X_50).numpy()
    else:
        raise ValueError(f"Unknown method: {method}")

    os.makedirs("visualizations/dim_reduction", exist_ok=True)
    save_path = f"visualizations/dim_reduction/{dataset}_{method}.png"
    plot_embedding(Z, labels.numpy(), class_names,
                   title=f"{method.upper()} on {dataset.upper()}",
                   save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "tiny-imagenet"])
    parser.add_argument("--method", default="pca",
                        choices=["pca", "tsne", "umap"])
    parser.add_argument("--n_samples", type=int, default=2000)
    args = parser.parse_args()
    run(args.dataset, args.method, args.n_samples)
