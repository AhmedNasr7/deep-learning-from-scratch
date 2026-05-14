"""
Variational Autoencoder (Kingma & Welling, 2014).

Extends the autoencoder with a probabilistic latent space:
  - Encoder outputs μ and log(σ²) instead of a single vector
  - Reparameterization trick: z = μ + σ · ε,  ε ~ N(0, I)
  - Loss = reconstruction + KL divergence (regularizes latent space)
"""

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """
    VAE encoder: image → (μ, log_var).

    Args:
        in_channels: input channels
        latent_dim:  latent space dimension

    Input:  (batch, in_channels, H, W)
    Output: mu (batch, latent_dim), log_var (batch, latent_dim)
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        super().__init__()

        # your code here
        # Conv layers → flatten → two separate linear heads for μ and log(σ²)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple:
        # your code here
        # Returns: (mu, log_var)
        raise NotImplementedError


class VAEDecoder(nn.Module):
    """
    VAE decoder: latent sample → reconstructed image.

    Same architecture as vanilla autoencoder decoder.

    Input:  (batch, latent_dim)
    Output: (batch, out_channels, H, W)
    """

    def __init__(self, latent_dim: int = 32, out_channels: int = 1):
        super().__init__()

        # your code here
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # your code here
        raise NotImplementedError


class VAE(nn.Module):
    """
    Full VAE: encode → reparameterize → decode.

    Loss = reconstruction_loss + β * KL_divergence
      KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

    Args:
        in_channels: input image channels
        latent_dim:  latent space size

    Input:  (batch, in_channels, H, W)
    Output: reconstruction, mu, log_var  (for loss computation)
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, in_channels)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ · ε

        Allows gradients to flow through the sampling step.

        Input:  mu (batch, latent_dim), log_var (batch, latent_dim)
        Output: z (batch, latent_dim)
        """
        # your code here
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple:
        # your code here
        # 1. mu, log_var = encoder(x)
        # 2. z = reparameterize(mu, log_var)
        # 3. x_recon = decoder(z)
        # Returns: (x_recon, mu, log_var)
        raise NotImplementedError

    @staticmethod
    def loss_function(x_recon: torch.Tensor, x: torch.Tensor,
                      mu: torch.Tensor, log_var: torch.Tensor,
                      beta: float = 1.0) -> dict:
        """
        VAE ELBO loss = reconstruction + β · KL divergence.

        Args:
            x_recon: reconstructed output
            x:       original input
            mu, log_var: encoder outputs
            beta:    KL weight (β-VAE, higher = more disentangled)

        Returns:
            dict with "loss" (tensor), "recon_loss" (float), "kl_loss" (float)
        """
        # your code here
        # recon_loss = F.mse_loss(x_recon, x, reduction="sum") / batch_size
        # kl_loss = -0.5 * sum(1 + log_var - mu² - exp(log_var)) / batch_size
        # total = recon_loss + beta * kl_loss
        raise NotImplementedError
