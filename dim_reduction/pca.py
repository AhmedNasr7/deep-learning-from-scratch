"""
PCA — Principal Component Analysis.

Linear dimensionality reduction via eigen-decomposition of the covariance matrix.
Finds orthogonal directions of maximum variance in the data.

Methods:
  - Exact SVD: X = U S V^T, take top-k columns of V as principal components
  - Randomized SVD: approximate fast version for large matrices
  - Incremental PCA: streaming version for data too large to fit in memory

Reference:
  - Pearson, "On lines and planes of closest fit to systems of points in space", 1901
  - Halko et al., "Finding Structure with Randomness", 2011
"""

import torch


class PCA:
    """
    PCA from scratch using SVD.

    Workflow:
        1. Center data: X = X - mean(X)
        2. SVD: X = U S V^T
        3. Principal components = columns of V (right singular vectors)
        4. Projection: Z = X @ V[:, :k]

    Args:
        n_components: number of principal components to keep
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None   # (n_components, n_features) — principal axes
        self.mean_ = None         # (n_features,) — data mean for centering
        self.explained_variance_ = None  # eigenvalues

    def fit(self, X: torch.Tensor) -> "PCA":
        """
        Fit PCA on data X.

        Args:
            X: (N, D) data matrix

        Steps:
            1. Compute and store mean
            2. Center: X_c = X - mean
            3. SVD: U, S, Vt = torch.linalg.svd(X_c, full_matrices=False)
            4. Store top-k rows of Vt as principal components
            5. Compute explained variance from singular values
        """
        raise NotImplementedError

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project X onto principal components.

        Args:
            X: (N, D)
        Returns:
            Z: (N, n_components)
        """
        raise NotImplementedError

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and project in one step."""
        raise NotImplementedError

    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct approximate X from low-dimensional Z.
        Useful for visualizing reconstruction quality.

        Args:
            Z: (N, n_components)
        Returns:
            X_approx: (N, D)
        """
        raise NotImplementedError
