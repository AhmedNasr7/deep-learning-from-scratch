"""
Pooling Layers — spatial downsampling operations.
"""

import torch
import torch.nn as nn


class MaxPool2d(nn.Module):
    """
    2D Max Pooling — takes the maximum value in each kernel window.

    Used for: spatial downsampling, translation invariance.

    Input:  (B, C, H, W)
    Output: (B, C, H_out, W_out)

    Args:
        kernel_size: window size
        stride:      step size (default: kernel_size)
        padding:     zero-padding before pooling
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use F.unfold to extract windows, take max per window
        raise NotImplementedError


class AvgPool2d(nn.Module):
    """
    2D Average Pooling — takes the mean value in each kernel window.

    Args:
        kernel_size: window size
        stride:      step size (default: kernel_size)
        padding:     zero-padding before pooling
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AdaptiveAvgPool2d(nn.Module):
    """
    Adaptive Average Pooling — pools to a target output size regardless of input size.

    Used in: ResNet final pooling before classifier head.

    Args:
        output_size: (H_out, W_out) or single int for square output
    """
    def __init__(self, output_size: int | tuple):
        super().__init__()
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute dynamic kernel/stride from input size and output_size
        raise NotImplementedError
