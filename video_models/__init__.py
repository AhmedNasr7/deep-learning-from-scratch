"""
Video transformer and spatiotemporal model scaffolds.

This package is intended for from-scratch video attention and video ViT research.
"""

from .tubelet_embedding import TubeletEmbedding
from .video_transformers import VideoTransformer
from .train import train

__all__ = ["TubeletEmbedding", "VideoTransformer", "train"]
