"""
Flow Matching (Lipman et al., 2023).

Simpler alternative to diffusion: learns a velocity field v(x, t)
that transports noise → data along straight-line paths.
No noise schedule needed — just interpolate between noise and data.

Training:
    1. Sample x_0 ~ data, x_1 ~ N(0, I), t ~ U(0, 1)
    2. Interpolate: x_t = (1 - t) · x_0 + t · x_1
    3. Target velocity: v* = x_1 - x_0
    4. Predict: v̂ = model(x_t, t)
    5. Loss = MSE(v̂, v*)

Sampling:
    Start from x_1 ~ N(0, I), integrate ODE dx/dt = v(x, t) from t=1 → t=0.
"""

import torch
import torch.nn as nn


class FlowMatching(nn.Module):
    """
    Flow matching wrapper around a velocity network.

    Args:
        model: velocity network — same architecture as diffusion UNet
               takes (x_t, t) and outputs velocity v(x, t)

    Input:  x_0 (batch, C, H, W) — clean data
    Output: scalar loss
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def training_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Compute flow matching loss.

        1. Sample noise x_1 ~ N(0, I) and t ~ U(0, 1)
        2. Interpolate x_t = (1-t) · x_0 + t · x_1
        3. Target velocity = x_1 - x_0
        4. Loss = MSE(model(x_t, t), target)
        """
        # your code here
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device, num_steps: int = 100) -> torch.Tensor:
        """
        Generate samples by Euler integration of the learned velocity field.

        Start from x ~ N(0, I), integrate from t=1 → t=0.

        Args:
            shape:     (B, C, H, W)
            device:    torch device
            num_steps: number of Euler steps (more = better quality)

        Output: (B, C, H, W) generated images
        """
        # your code here
        # dt = 1.0 / num_steps
        # for t in [1.0, 1-dt, 1-2dt, ..., dt]:
        #     v = model(x, t)
        #     x = x - dt * v
        raise NotImplementedError
