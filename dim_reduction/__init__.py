"""
Dimensionality Reduction methods from scratch.

Modules:
  pca.py              — PCA via SVD (linear, fast, exact)
  tsne.py             — t-SNE (non-linear, visualization-only, 2D/3D)
  umap.py             — UMAP (non-linear, faster than t-SNE, preserves global structure)
  test_on_cifar.py    — Test script: run all methods on CIFAR-10 / Tiny-ImageNet

Usage:
    from dim_reduction.pca import PCA
    from dim_reduction.tsne import TSNE
    from dim_reduction.umap import UMAP

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)  # (N, 2)
"""

from .pca import PCA
from .tsne import TSNE
from .umap import UMAP

__all__ = ["PCA", "TSNE", "UMAP"]
