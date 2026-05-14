"""
VAE latent space visualization utilities.

  - Latent space scatter plots (colored by class)
  - Interpolation between two data points
  - Interactive slider for latent space exploration
  - Reconstruction grids
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import Optional, List


def plot_latent_space(model, data_loader, device, title="Latent Space", save_path=None):
    """
    Scatter plot of encoded data in 2D latent space, colored by label.

    Best with latent_dim=2. For higher dims, uses first 2 components.
    """
    model.eval()
    zs, labels = [], []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            mu, _ = model.encoder(x)
            zs.append(mu.cpu())
            labels.append(y)

    zs = torch.cat(zs).numpy()
    labels = torch.cat(labels).numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(zs[:, 0], zs[:, 1], c=labels, cmap="tab10", s=2, alpha=0.6)
    fig.colorbar(scatter, ax=ax)
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_interpolation(model, x1, x2, device, steps=10, save_path=None):
    """
    Linear interpolation between two data points in latent space.

    Shows: x1 → decode(lerp(encode(x1), encode(x2))) → x2
    """
    model.eval()
    with torch.no_grad():
        mu1, _ = model.encoder(x1.unsqueeze(0).to(device))
        mu2, _ = model.encoder(x2.unsqueeze(0).to(device))

        alphas = torch.linspace(0, 1, steps)
        interpolated = []
        for a in alphas:
            z = (1 - a) * mu1 + a * mu2
            recon = model.decoder(z).cpu()
            interpolated.append(recon.squeeze(0))

    fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 2))
    for i, img in enumerate(interpolated):
        ax = axes[i]
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
        ax.axis("off")
        ax.set_title(f"α={alphas[i]:.1f}", fontsize=8)

    plt.suptitle("Latent Space Interpolation", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_reconstruction_grid(model, data_loader, device, n=8, save_path=None):
    """
    Side-by-side grid: original vs reconstruction.
    """
    model.eval()
    x, _ = next(iter(data_loader))
    x = x[:n].to(device)

    with torch.no_grad():
        recon, _, _ = model(x)

    x = x.cpu()
    recon = recon.cpu()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        for row, img in enumerate([x[i], recon[i]]):
            ax = axes[row, i]
            if img.shape[0] == 1:
                ax.imshow(img.squeeze(0), cmap="gray")
            else:
                ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
            ax.axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)

    plt.suptitle("Reconstruction Quality", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_random_samples(model, device, latent_dim, n=16, save_path=None):
    """
    Generate random samples by sampling z ~ N(0, I) and decoding.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device)
        samples = model.decoder(z).cpu()

    cols = int(np.sqrt(n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        img = samples[i]
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
        ax.axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Random Samples from Prior", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def interactive_latent_slider(model, device, latent_dim=2):
    """
    Interactive ipywidgets slider to explore the latent space in Jupyter.

    Creates one slider per latent dimension (up to 10).
    Moving a slider regenerates the decoded output in real-time.

    Requirements:
        pip install ipywidgets
        In Jupyter: works out of the box with %matplotlib inline
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    model.eval()
    n_sliders = min(latent_dim, 10)

    # Build sliders
    sliders = {}
    for i in range(n_sliders):
        sliders[f"z{i}"] = widgets.FloatSlider(
            value=0.0, min=-4.0, max=4.0, step=0.05,
            description=f"z{i}",
            continuous_update=True,
            style={"description_width": "30px"},
            layout=widgets.Layout(width="500px"),
        )

    output = widgets.Output()

    def render(**kwargs):
        z = torch.zeros(1, latent_dim).to(device)
        for i in range(n_sliders):
            z[0, i] = kwargs[f"z{i}"]

        with torch.no_grad():
            img = model.decoder(z).cpu().squeeze(0)

        with output:
            clear_output(wait=True)
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            if img.shape[0] == 1:
                ax.imshow(img.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            ax.axis("off")
            ax.set_title("Generated from Latent Space")
            plt.tight_layout()
            plt.show()

    ui = widgets.interactive(render, **sliders)
    display(widgets.VBox([ui, output]))

