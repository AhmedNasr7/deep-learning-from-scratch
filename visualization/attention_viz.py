"""
Attention pattern visualization.

Visualize what the transformer is attending to during inference.
Useful for debugging and understanding model behavior.

Usage:
    from visualization import plot_attention_map, plot_multi_head_attention

    # Single head
    plot_attention_map(attn_weights, src_tokens, tgt_tokens)

    # All heads
    plot_multi_head_attention(attn_weights, src_tokens, tgt_tokens)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


def plot_attention_map(
    attention: torch.Tensor,
    src_tokens: List[str] = None,
    tgt_tokens: List[str] = None,
    title: str = "Attention Weights",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Plot a single attention weight matrix as a heatmap.

    Args:
        attention:  (tgt_len, src_len) attention weights
        src_tokens: source token labels for x-axis
        tgt_tokens: target token labels for y-axis
        title:      plot title
        save_path:  save path (optional)
        show:       display plot
    """
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention, cmap="viridis", aspect="auto", vmin=0, vmax=attention.max())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if src_tokens:
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=9)
    if tgt_tokens:
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=9)

    ax.set_xlabel("Source", fontsize=11)
    ax.set_ylabel("Target", fontsize=11)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_multi_head_attention(
    attention: torch.Tensor,
    src_tokens: List[str] = None,
    tgt_tokens: List[str] = None,
    max_heads: int = 8,
    title: str = "Multi-Head Attention",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Plot attention maps for multiple heads in a grid.

    Args:
        attention:  (n_heads, tgt_len, src_len) attention weights
        src_tokens: source token labels
        tgt_tokens: target token labels
        max_heads:  max number of heads to display
        title:      plot title
        save_path:  save path (optional)
        show:       display plot
    """
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()

    n_heads = min(attention.shape[0], max_heads)
    cols = min(n_heads, 4)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_heads):
        ax = axes[i]
        im = ax.imshow(attention[i], cmap="viridis", aspect="auto", vmin=0)
        ax.set_title(f"Head {i}", fontsize=10)

        if src_tokens and i >= n_heads - cols:
            ax.set_xticks(range(len(src_tokens)))
            ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticks([])

        if tgt_tokens and i % cols == 0:
            ax.set_yticks(range(len(tgt_tokens)))
            ax.set_yticklabels(tgt_tokens, fontsize=7)
        else:
            ax.set_yticks([])

    for i in range(n_heads, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_attention_rollout(
    attention_maps: List[torch.Tensor],
    image: torch.Tensor = None,
    patch_size: int = 4,
    img_size: int = 32,
    title: str = "Attention Rollout",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Attention rollout — aggregate attention across all layers.

    Multiplies attention matrices from all layers to see where
    the final representation is "really" attending.

    Used in ViT to visualize which image patches matter for classification.

    Args:
        attention_maps: list of (n_heads, T, T) attention per layer
        image:          optional (C, H, W) original image to overlay
        patch_size:     ViT patch size
        img_size:       original image size
        title:          plot title
        save_path:      save path (optional)
        show:           display plot
    """
    # Average across heads per layer, then multiply all layers
    result = None
    for attn in attention_maps:
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu().numpy()

        # Average across heads: (n_heads, T, T) → (T, T)
        attn_avg = attn.mean(axis=0)

        # Add identity (residual connections)
        attn_avg = attn_avg + np.eye(attn_avg.shape[0])

        # Normalize rows
        attn_avg = attn_avg / attn_avg.sum(axis=-1, keepdims=True)

        if result is None:
            result = attn_avg
        else:
            result = result @ attn_avg

    # Extract CLS token attention (row 0, skip CLS column)
    cls_attention = result[0, 1:]  # skip CLS→CLS

    # Reshape to patch grid
    grid_size = img_size // patch_size
    attn_map = cls_attention.reshape(grid_size, grid_size)

    # Plot
    if image is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        if isinstance(image, torch.Tensor):
            img_np = image.detach().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        else:
            img_np = image

        ax1.imshow(img_np)
        ax1.set_title("Original", fontsize=11)
        ax1.axis("off")

        ax2.imshow(attn_map, cmap="hot", interpolation="bilinear")
        ax2.set_title("Attention Rollout", fontsize=11)
        ax2.axis("off")

        # Overlay
        import matplotlib
        attn_resized = np.array(
            plt.cm.hot(matplotlib.colors.Normalize()(
                np.kron(attn_map, np.ones((patch_size, patch_size)))
            ))
        )[:, :, :3]
        overlay = 0.5 * img_np + 0.5 * attn_resized
        ax3.imshow(overlay)
        ax3.set_title("Overlay", fontsize=11)
        ax3.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_map, cmap="hot", interpolation="bilinear")
        fig.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
