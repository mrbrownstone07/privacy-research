"""
Graph Generation Toolkit for Privacy-Preserving Machine Learning

This module provides a unified interface for constructing graph structures from
vector data, specifically optimized for spectral utility and Differential Privacy.

Key Research References:
------------------------
[1] von Luxburg, U. (2007). A tutorial on spectral clustering.
    Statistics and Computing, 17(4), 395-416.
[2] Zelnik-Manor, L., & Perona, P. (2004). Self-tuning spectral clustering.
    Advances in Neural Information Processing Systems (NIPS).
[3] Jebara, T., Wang, J., & Chang, S. F. (2009). Graph construction and b-matching.
    International Conference on Machine Learning (ICML).

Author: Mahdi Mohammad Shibli
Date: 2026
"""

import numpy as np
from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import StandardScaler


class GraphMethod(Enum):
    """Enumeration of graph construction strategies."""
    KNN_BINARY = "knn_binary"               # Standard k-NN [von Luxburg 2007]
    MUTUAL_KNN = "mutual_knn"               # Mutual k-NN [Wang & Zhang 2008]
    EPSILON_RADIUS = "epsilon_radius"       # Fixed radius neighborhood
    GAUSSIAN_KNN = "gaussian_knn"           # RBF kernel on k-NN [Ng et al. 2002]
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian" # Local scaling [Zelnik-Manor 2004]
    B_MATCHING = "b_matching"               # Balanced degree graph [Jebara 2009]


@dataclass
class GraphResult:
    """Container for generated graph data and structural metadata."""
    adjacency: np.ndarray
    laplacian: np.ndarray
    method: GraphMethod
    n_edges: int
    density: float
    metadata: Dict


class GraphGenerator:
    """
    Unified framework for generating graphs from feature matrices.

    This class implements various construction methods that serve as the
    foundation for Laplacian-based privacy perturbation.
    """

    def __init__(self, X: np.ndarray, standardize: bool = True):
        """
        Initialize the generator with source data.

        Args:
            X: Data matrix (n_samples × n_features)
            standardize: Whether to apply Z-score normalization
        """
        self.X_original = X
        self.n, self.d = X.shape

        if standardize:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)
        else:
            self.X = X.copy()
            self.scaler = None

    def generate(self,
                 method: GraphMethod = GraphMethod.KNN_BINARY,
                 **kwargs) -> GraphResult:
        """
        Main interface to route graph construction.

        Args:
            method: Strategy to use for edge creation
            **kwargs: Method-specific parameters (k, epsilon, sigma, etc.)

        Returns:
            GraphResult containing adjacency, Laplacian, and stats.
        """
        method_map = {
            GraphMethod.KNN_BINARY: self._knn_binary,
            GraphMethod.MUTUAL_KNN: self._mutual_knn,
            GraphMethod.EPSILON_RADIUS: self._epsilon_radius,
            GraphMethod.GAUSSIAN_KNN: self._gaussian_knn,
            GraphMethod.ADAPTIVE_GAUSSIAN: self._adaptive_gaussian,
            GraphMethod.B_MATCHING: self._b_matching_approx,
        }

        if method not in method_map:
            raise ValueError(f"Method {method} not implemented.")

        A = method_map[method](**kwargs)
        L = self.compute_laplacian(A, normalized=kwargs.get('normalized', False))

        n_edges = int(np.count_nonzero(A) / 2)
        density = n_edges / (self.n * (self.n - 1) / 2)

        return GraphResult(
            adjacency=A,
            laplacian=L,
            method=method,
            n_edges=n_edges,
            density=density,
            metadata=kwargs
        )

    # --- Internal Construction Methods ---

    def _knn_binary(self, k: int = 10, **kwargs) -> np.ndarray:
        """Standard k-Nearest Neighbors graph."""
        A = kneighbors_graph(self.X, n_neighbors=k, mode='connectivity', include_self=False).toarray()
        return np.maximum(A, A.T)

    def _mutual_knn(self, k: int = 10, **kwargs) -> np.ndarray:
        """Mutual k-NN: Edge exists only if j is in k-NN(i) AND i is in k-NN(j)."""
        A_dir = kneighbors_graph(self.X, n_neighbors=k, mode='connectivity', include_self=False).toarray()
        return np.minimum(A_dir, A_dir.T)

    def _epsilon_radius(self, epsilon: float = 0.5, **kwargs) -> np.ndarray:
        """Connects all nodes within Euclidean distance epsilon."""
        D = squareform(pdist(self.X, 'euclidean'))
        A = (D <= epsilon).astype(float)
        np.fill_diagonal(A, 0)
        return A

    def _gaussian_knn(self, k: int = 10, sigma: Optional[float] = None, **kwargs) -> np.ndarray:
        """Weighted graph using Gaussian RBF kernel on k-NN structure."""
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(self.X)
        distances, indices = nbrs.kneighbors(self.X)

        # Median heuristic for sigma if not provided
        if sigma is None:
            sigma = np.median(pdist(self.X))

        # Vectorized weight calculation
        weights = np.exp(-(distances[:, 1:]**2) / (2 * sigma**2))

        A = np.zeros((self.n, self.n))
        rows = np.repeat(np.arange(self.n), k)
        cols = indices[:, 1:].flatten()
        A[rows, cols] = weights.flatten()

        return (A + A.T) / 2

    def _adaptive_gaussian(self, k: int = 7, **kwargs) -> np.ndarray:
        """Self-tuning graph with local scaling sigma_i based on k-th neighbor."""
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)

        sigmas = distances[:, k] # Local scale for each node
        D_sq = squareform(pdist(self.X, 'sqeuclidean'))

        # Scale denominator: σ_i * σ_j
        scale_matrix = np.outer(sigmas, sigmas)
        A = np.exp(-D_sq / (scale_matrix + 1e-10))
        np.fill_diagonal(A, 0)

        # Sparsify based on k-NN structure to keep it efficient
        A_mask = self._knn_binary(k=k)
        return A * A_mask

    def _b_matching_approx(self, b: int = 5, **kwargs) -> np.ndarray:
        """Greedy approximation of b-matching (regular degree graphs)."""
        D = squareform(pdist(self.X, 'euclidean'))
        # Get indices of flattened upper triangle sorted by distance
        triu_idx = np.triu_indices(self.n, k=1)
        distances = D[triu_idx]
        sorted_indices = np.argsort(distances)

        A = np.zeros((self.n, self.n))
        degrees = np.zeros(self.n)

        for idx in sorted_indices:
            u, v = triu_idx[0][idx], triu_idx[1][idx]
            if degrees[u] < b and degrees[v] < b:
                A[u, v] = A[v, u] = 1
                degrees[u] += 1
                degrees[v] += 1
        return A

    # --- Static Utilities ---

    @staticmethod
    def compute_laplacian(A: np.ndarray, normalized: bool = False) -> np.ndarray:
        """Computes the Laplacian matrix L = D - A."""
        degrees = A.sum(axis=1)
        if normalized:
            d_inv_sqrt = np.power(degrees + 1e-10, -0.5)
            D_inv_sqrt = np.diag(d_inv_sqrt)
            return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
        return np.diag(degrees) - A

    def get_summary(self, result: GraphResult):
        """Prints a research-formatted summary of the graph."""
        print(f"--- Graph Summary: {result.method.value} ---")
        print(f"Nodes: {self.n} | Edges: {result.n_edges} | Density: {result.density:.4f}")
        print(f"Standardized: {self.scaler is not None}")
        print("-" * 36)
