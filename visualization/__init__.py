"""
Visualization utilities for training and analysis.

  - Loss/metric plotting (matplotlib, TensorBoard)
  - Attention pattern visualization
"""

from .training_plots import plot_losses, plot_metrics
from .attention_viz import plot_attention_map, plot_multi_head_attention
