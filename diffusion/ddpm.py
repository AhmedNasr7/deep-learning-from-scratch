"""
DDPM — Denoising Diffusion Probabilistic Models (Ho et al., 2020).

Forward process: gradually add Gaussian noise over T steps.
Reverse process: learn to denoise step-by-step using a neural network (UNet).
"""

import torch
import torch.nn as nn
import math


class NoiseSchedule:
    """
    Linear or cosine noise schedule for diffusion.

    Precomputes: betas, alphas, alpha_bars (cumulative products)
    for T timesteps.

    Args:
        timesteps:     total diffusion steps T
        schedule_type: 'linear' or 'cosine'
        beta_start:    starting beta (linear only)
        beta_end:      ending beta (linear only)
    """

    def __init__(self, timesteps: int = 1000, schedule_type: str = "linear",
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps

        # your code here
        # 1. Compute betas: (T,) noise level per step
        # 2. alphas = 1 - betas
        # 3. alpha_bars = cumulative product of alphas
        # 4. Store as tensors
        raise NotImplementedError


class DDPM(nn.Module):
    """
    DDPM wrapper: ties noise schedule + denoising model (UNet).

    Args:
        model:    denoising network (UNet) — predicts noise ε
        schedule: NoiseSchedule instance

    Training:
        1. Sample x_0 from data, t ~ Uniform(1, T), ε ~ N(0, I)
        2. Compute noisy x_t = √(α̅_t) · x_0 + √(1 - α̅_t) · ε
        3. Predict ε̂ = model(x_t, t)
        4. Loss = MSE(ε, ε̂)

    Sampling:
        Start from x_T ~ N(0, I), iteratively denoise to x_0.
    """

    def __init__(self, model: nn.Module, schedule: NoiseSchedule):
        super().__init__()
        self.model = model
        self.schedule = schedule

    def forward_diffusion(self, x_0: torch.Tensor, t: torch.Tensor,
                          noise: torch.Tensor = None) -> tuple:
        """
        Add noise to x_0 at timestep t.

        Input:  x_0 (batch, C, H, W), t (batch,)
        Output: (x_t, noise) — noisy image and the noise that was added
        """
        # your code here
        raise NotImplementedError

    def training_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Compute DDPM training loss for a batch.

        1. Sample random timesteps
        2. Add noise via forward_diffusion
        3. Predict noise with self.model
        4. Return MSE loss
        """
        # your code here
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Generate samples by iterative denoising from x_T ~ N(0, I).

        Input:  shape (B, C, H, W), device
        Output: (B, C, H, W) generated images
        """
        # your code here
        raise NotImplementedError
