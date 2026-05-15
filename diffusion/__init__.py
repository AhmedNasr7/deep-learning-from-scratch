"""
Diffusion Models — generative modeling via iterative denoising.

Modules:
  ddpm.py — DDPM: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
  unet.py — UNet denoising backbone (U-shaped encoder-decoder with skip connections)
"""

from .ddpm import DDPM, NoiseSchedule

__all__ = ["DDPM", "NoiseSchedule"]
