"""
Convolutional Layers — Conv1d and Conv2d from scratch.

Implements discrete cross-correlation (what DL calls "convolution"):
    out[n, c_out, h, w] = Σ_{c_in, kh, kw} weight[c_out, c_in, kh, kw]
                           * input[n, c_in, h*stride+kh, w*stride+kw]

Reference:
  - LeCun et al., "Gradient-based learning applied to document recognition", 1998
"""

import torch
import torch.nn as nn


class Conv1d(nn.Module):
    """
    1D Convolution — slides a kernel over a sequence.

    Input:  (B, C_in, L)
    Output: (B, C_out, L_out)
    L_out = floor((L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)

    Args:
        in_channels:  number of input channels
        out_channels: number of filters (output channels)
        kernel_size:  size of the convolving kernel
        stride:       step size (default: 1)
        padding:      zero-padding on both sides (default: 0)
        bias:         whether to add bias term
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement as explicit sliding window or via F.unfold for efficiency.
        """
        raise NotImplementedError


class Conv2d(nn.Module):
    """
    2D Convolution — slides a kernel over an image.

    Input:  (B, C_in, H, W)
    Output: (B, C_out, H_out, W_out)

    Args:
        in_channels:  number of input channels
        out_channels: number of filters
        kernel_size:  (kH, kW) or single int (square kernel)
        stride:       (sH, sW) or single int
        padding:      (pH, pW) or single int
        groups:       grouped convolution (groups=in_channels → depthwise conv)
        bias:         whether to add bias
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple, stride: int | tuple = 1,
                 padding: int | tuple = 0, groups: int = 1, bias: bool = True):
        super().__init__()
        # Normalize int → tuple
        kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kH, kW))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use F.unfold to extract patches, then matrix multiply.

        Steps:
            1. F.unfold(x, kernel_size, padding, stride) → (B, C_in*kH*kW, N_patches)
            2. weight.view(C_out, -1) @ unfolded → (B, C_out, N_patches)
            3. F.fold or reshape to (B, C_out, H_out, W_out)
            4. Add bias
        """
        raise NotImplementedError
