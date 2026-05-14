"""
Training visualization — loss curves, metrics, and TensorBoard integration.

Usage in notebooks:
    from visualization import plot_losses, plot_metrics

    plot_losses(train_losses, val_losses, save_path="loss_curve.png")
    plot_metrics({"train_acc": [...], "val_acc": [...]})

Usage with TensorBoard:
    from visualization.training_plots import TBLogger

    logger = TBLogger("runs/experiment_1")
    logger.log_scalar("loss/train", loss, step)
    logger.log_scalar("loss/val", val_loss, step)
    logger.close()

    # Then: tensorboard --logdir runs/
"""

import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def plot_losses(
    train_losses: List[float],
    val_losses: List[float] = None,
    title: str = "Training Loss",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Plot training (and optionally validation) loss curves.

    Args:
        train_losses: list of training losses per epoch
        val_losses:   list of validation losses per epoch (optional)
        title:        plot title
        save_path:    if set, saves the figure to this path
        show:         if True, displays the plot (set False in scripts)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Train", linewidth=2)

    if val_losses:
        ax.plot(epochs, val_losses, "r-", label="Val", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Training Metrics",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Plot multiple metrics on the same figure.

    Args:
        metrics:   dict of metric_name → list of values per epoch
                   e.g. {"train_acc": [...], "val_acc": [...], "bleu": [...]}
        title:     plot title
        save_path: if set, saves the figure
        show:      if True, displays the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name, values in metrics.items():
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, label=name, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


class TBLogger:
    """
    Thin TensorBoard wrapper for easy logging.

    Usage:
        logger = TBLogger("runs/my_experiment")
        logger.log_scalar("loss/train", 0.5, step=100)
        logger.log_scalars("loss", {"train": 0.5, "val": 0.7}, step=100)
        logger.close()
    """

    def __init__(self, log_dir: str = "runs"):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int) -> None:
        self.writer.add_scalars(main_tag, values, step)

    def log_image(self, tag: str, image, step: int) -> None:
        self.writer.add_image(tag, image, step)

    def close(self) -> None:
        self.writer.close()
