"""
CLIP — Contrastive Language-Image Pretraining.

Trains an image encoder and a text encoder jointly by maximizing agreement
between paired (image, text) embeddings and minimizing agreement between
unpaired examples. Uses a symmetric cross-entropy loss over a cosine
similarity matrix.

Reference:
  - Radford et al., "Learning Transferable Visual Models From Natural Language
    Supervision", ICML 2021
"""

import torch
import torch.nn as nn


class CLIPImageEncoder(nn.Module):
    """
    Vision encoder for CLIP.

    Typically a ViT or ResNet that maps images to a fixed-size embedding.

    Args:
        backbone:    any nn.Module that produces (B, embed_dim) image features
        embed_dim:   output embedding dimension (shared with text encoder)
        projection_dim: final projected embedding size (e.g. 512)
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, projection_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(embed_dim, projection_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W)
        Returns:
            image_embeddings: (B, projection_dim) — L2 normalized
        """
        # your code here
        # 1. Pass through backbone → (B, embed_dim)
        # 2. Project → (B, projection_dim)
        # 3. L2 normalize
        raise NotImplementedError


class CLIPTextEncoder(nn.Module):
    """
    Text encoder for CLIP.

    A transformer encoder that maps tokenized text to a fixed-size embedding.

    Args:
        vocab_size:     vocabulary size
        d_model:        transformer model dimension
        n_heads:        number of attention heads
        n_layers:       number of transformer layers
        max_seq_len:    maximum token sequence length
        projection_dim: final projected embedding size (shared with image encoder)
    """

    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, max_seq_len: int = 77, projection_dim: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # TODO: add transformer encoder blocks
        self.proj = nn.Linear(d_model, projection_dim, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) token indices
        Returns:
            text_embeddings: (B, projection_dim) — L2 normalized
        """
        # your code here
        # 1. Token + positional embeddings
        # 2. Pass through transformer layers
        # 3. Pool (take [EOS] token representation)
        # 4. Project + L2 normalize
        raise NotImplementedError


class CLIP(nn.Module):
    """
    Full CLIP model.

    Combines image and text encoders with a learnable temperature parameter
    and symmetric InfoNCE loss.

    Args:
        image_encoder: CLIPImageEncoder
        text_encoder:  CLIPTextEncoder
        init_temp:     initial log temperature (learnable)
    """

    def __init__(self, image_encoder: CLIPImageEncoder,
                 text_encoder: CLIPTextEncoder, init_temp: float = 0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * init_temp)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> tuple:
        """
        Args:
            images:    (B, C, H, W)
            token_ids: (B, T)
        Returns:
            (image_embeds, text_embeds, logit_scale)
        """
        # your code here
        # 1. Encode images → (B, D)
        # 2. Encode text   → (B, D)
        # 3. Return embeddings and temperature for loss computation
        raise NotImplementedError


def clip_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor,
              logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE (contrastive) loss for CLIP.

    Computes cosine similarity matrix, then cross-entropy in both directions
    (image→text and text→image), averaging them.

    Args:
        image_embeds: (B, D) normalized image embeddings
        text_embeds:  (B, D) normalized text embeddings
        logit_scale:  scalar temperature (learned)

    Returns:
        scalar loss
    """
    # your code here
    # 1. logits = logit_scale * image_embeds @ text_embeds.T   (B x B)
    # 2. labels = torch.arange(B)  (diagonal = positive pairs)
    # 3. loss = 0.5 * (CE(logits, labels) + CE(logits.T, labels))
    raise NotImplementedError
