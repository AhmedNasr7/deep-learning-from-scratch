"""
Vanilla Autoencoder.

Learns a compressed latent representation by training encoder and decoder
to minimize reconstruction error. No probabilistic sampling — just
a bottleneck.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Convolutional encoder: image → latent vector.

    Args:
        in_channels: input channels (1 for MNIST, 3 for CIFAR)
        latent_dim:  bottleneck dimension

    Input:  (batch, in_channels, H, W)
    Output: (batch, latent_dim)
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        # your code here
        # Conv layers to downsample, then flatten + linear to latent_dim
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        raise NotImplementedError


class Decoder(nn.Module):
    """
    Convolutional decoder: latent vector → reconstructed image.

    Args:
        latent_dim:   bottleneck dimension
        out_channels: output channels

    Input:  (batch, latent_dim)
    Output: (batch, out_channels, H, W)
    """

    def __init__(self, latent_dim: int = 32, out_channels: int = 1):
        super().__init__()

        # your code here
        # Linear to spatial, then ConvTranspose layers to upsample
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # your code here
        raise NotImplementedError


class Autoencoder(nn.Module):
    """
    Full autoencoder: encode → latent → decode.

    Loss: MSE or BCE reconstruction loss.

    Args:
        in_channels: input image channels
        latent_dim:  bottleneck size

    Input:  (batch, in_channels, H, W)
    Output: (batch, in_channels, H, W) — reconstruction
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # z = encoder(x), x_recon = decoder(z)
        raise NotImplementedError
