import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Optional

class GeneralizedSpectralAnalyst:
    """
    Analyzes the 'Normalized' structural properties of a Graph Laplacian.

    Solves the generalized eigenproblem: base_laplacian @ x = lambda @ base_degree_matrix @ x.
    This approach accounts for local density variation (degree) within the graph.
    """

    def __init__(self, base_laplacian: np.ndarray, base_degree_matrix: np.ndarray, tolerance: float = 1e-10):
        self.base_laplacian = base_laplacian
        self.base_degree_matrix = base_degree_matrix
        self.num_nodes = base_laplacian.shape[0]
        self.tolerance = tolerance
        self.identity_matrix = np.eye(self.num_nodes)

        # Immediate spectral characterization of the graph's structural DNA
        self._analyze_base_structure()

    def _analyze_base_structure(self):
        """
        Solves the initial generalized eigenproblem.
        Eigenvectors are normalized such that: vector.T @ base_degree_matrix @ vector = 1
        """
        # 1. Solve the generalized problem using the standard symmetric solver
        self.eigenvalues, self.eigenvectors = eigh(self.base_laplacian, self.base_degree_matrix)

        # 2. Identify Connectivity (Counting 'islands' or zero-eigenvalue modes)
        self.num_connected_components = np.sum(self.eigenvalues < self.tolerance)

        # 3. Locate Generalized Fiedler Mode (The first non-zero mode)
        # This mode represents the most significant structural split in the data.
        self.fiedler_index = self.num_connected_components
        self.base_fiedler_value = self.eigenvalues[self.fiedler_index]
        self.base_fiedler_vector = self.eigenvectors[:, self.fiedler_index]

        print("--- Generalized Structural Analysis Initialized ---")
        print(f"Connected Components (Islands): {self.num_connected_components}")
        print(f"Fiedler Mode located at Index: {self.fiedler_index}")
        print(f"Base Algebraic Connectivity (λ₀): {self.base_fiedler_value:.6f}")

    def get_fiedler_mode(self) -> Tuple[float, np.ndarray]:
        """
        Retrieves the baseline Fiedler value and vector before any changes are applied.
        """
        return self.base_fiedler_value, self.base_fiedler_vector

    def theoretical_sensitivity(
        self,
        laplacian_perturbation: np.ndarray,
        degree_perturbation: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Predicts the 1st-order shift in the Fiedler mode using perturbation theory.

        Args:
            laplacian_perturbation: The hypothetical change to the connectivity (L1).
            degree_perturbation: The hypothetical change to node densities (D1).
        """
        if degree_perturbation is None:
            degree_perturbation = np.zeros_like(self.base_degree_matrix)

        # 1. Calculate the predicted eigenvalue shift (λ₁)
        # Formula: λ₁ = x₀ᵀ (L₁ - λ₀D₁) x₀
        # (Assuming D₀-normalization of the eigenvectors)
        operator_perturbation = laplacian_perturbation - (self.base_fiedler_value * degree_perturbation)
        predicted_value_shift = self.base_fiedler_vector.T @ operator_perturbation @ self.base_fiedler_vector

        # 2. Calculate the predicted eigenvector shift (x₁)
        # We solve the generalized system on the orthogonal subspace:
        # (L₀ - λ₀D₀) x₁ = -(L₁ - λ₀D₁ - λ₁D₀) x₀
        right_hand_side = -(operator_perturbation - (predicted_value_shift * self.base_degree_matrix)) @ self.base_fiedler_vector

        # Construct the singular operator and find the response via Moore-Penrose pseudoinverse
        singular_operator = self.base_laplacian - (self.base_fiedler_value * self.base_degree_matrix)
        pseudo_inverse_operator = np.linalg.pinv(singular_operator)

        predicted_vector_shift = pseudo_inverse_operator @ right_hand_side

        return float(predicted_value_shift), predicted_vector_shift
