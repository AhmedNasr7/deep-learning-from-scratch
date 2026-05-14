"""
Denoising Autoencoder (Vincent et al., 2008).

Trains by corrupting input with noise, then learning to reconstruct
the clean version. Forces the encoder to learn robust features
rather than just memorizing the identity function.

Noise types:
  - Gaussian: x_noisy = x + N(0, σ²)
  - Masking:  randomly zero out a fraction of pixels
  - Salt & pepper: randomly set pixels to 0 or 1
"""

import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for images.

    Same architecture as vanilla autoencoder, but during training:
      1. Corrupt input x → x_noisy
      2. Encode x_noisy
      3. Decode → x_recon
      4. Loss = MSE(x_recon, x_clean)  ← compare to CLEAN input

    Args:
        in_channels:  image channels (1 or 3)
        latent_dim:   bottleneck dimension
        base_channels: base conv width
        noise_type:   "gaussian", "masking", or "salt_pepper"
        noise_factor: noise intensity (σ for gaussian, fraction for masking)

    Input:  (batch, in_channels, H, W)
    Output: (batch, in_channels, H, W) — denoised reconstruction
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 64,
                 base_channels: int = 32, noise_type: str = "gaussian",
                 noise_factor: float = 0.3):
        super().__init__()
        self.noise_type = noise_type
        self.noise_factor = noise_factor

        # your code here
        # Encoder: Conv blocks to downsample → flatten → linear to latent_dim
        # Decoder: Linear → reshape → ConvTranspose blocks to upsample → Sigmoid
        raise NotImplementedError

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Corrupt input with noise. Used only during training.

        Args:
            x: clean input (batch, C, H, W) in [0, 1]

        Returns:
            x_noisy: corrupted input, clamped to [0, 1]
        """
        # your code here
        # "gaussian":     x + N(0, noise_factor²), clamp to [0, 1]
        # "masking":      randomly zero out noise_factor fraction of pixels
        # "salt_pepper":  randomly set noise_factor fraction to 0 or 1
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        # your code here
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        # your code here
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with noise injection (if training).

        Input:  x (batch, C, H, W) — clean image
        Output: (x_recon, x_noisy)
                x_recon compared to x_clean for loss
                x_noisy returned for visualization
        """
        # your code here
        # if training: x_noisy = add_noise(x), else: x_noisy = x
        # z = encode(x_noisy)
        # x_recon = decode(z)
        # return (x_recon, x_noisy)
        raise NotImplementedError
