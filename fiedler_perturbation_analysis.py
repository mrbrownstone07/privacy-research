"""
Fiedler Value/Vector Perturbation Analysis for Graph-Based Differential Privacy
================================================================================

Based on: Greenbaum, Li, Overton (2020) "First-Order Perturbation Theory
for Eigenvalues and Eigenvectors"

This implementation addresses:
1. Laplacian-distributed noise (not Gaussian)
2. Frobenius norm scaling
3. First non-zero eigenvalue detection (proper Fiedler value)
4. First-order perturbation theory predictions
5. Privacy-utility tradeoff analysis
"""

import numpy as np
from scipy.linalg import eigh
from scipy.stats import laplace
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import warnings


class FiedlerPerturbationAnalyzer:
    """
    Analyzes perturbations to the Fiedler value and vector of a graph Laplacian
    using first-order perturbation theory.

    Literature Reference: Greenbaum et al. (2020), Section 2-3
    - Eigenvalue perturbation: λ₁ = (y₀ᴴ A₁ x₀) / (y₀ᴴ x₀)  [Eq 2.1]
    - Eigenvector perturbation: P₁(A₀ - λ₀I)⁻¹ P₁(λ₁I - A₁)x₀  [Eq 3.3-3.4]
    """

    def __init__(self, laplacian: np.ndarray, tol: float = 1e-10):
        """
        Initialize with a graph Laplacian matrix.

        Args:
            laplacian: Symmetric graph Laplacian matrix (n x n)
            tol: Tolerance for zero eigenvalue detection
        """
        self.L0 = laplacian
        self.n = laplacian.shape[0]
        self.tol = tol

        # Verify Laplacian properties
        self._verify_laplacian()

        # Compute original eigenstructure
        self.eigvals, self.eigvecs = eigh(self.L0)

        # Find Fiedler value (first non-zero eigenvalue)
        self.fiedler_idx = self._find_fiedler_index()
        self.lambda0 = self.eigvals[self.fiedler_idx]
        self.x0 = self.eigvecs[:, self.fiedler_idx]  # Fiedler vector (right)
        self.y0 = self.x0.copy()  # For symmetric matrices, left = right

        # Normalize: y0ᴴ x0 = 1 (eigenprojector normalization)
        # Literature: Section 4, page 473-476
        norm_factor = np.dot(self.y0.conj(), self.x0)
        self.x0 = self.x0 / np.sqrt(np.abs(norm_factor))
        self.y0 = self.y0 / np.sqrt(np.abs(norm_factor))

        print(f"✓ Laplacian initialized ({self.n}×{self.n})")
        print(f"✓ Fiedler value λ₀ = {self.lambda0:.6f} (index {self.fiedler_idx})")
        print(f"✓ Spectral gap: {self.eigvals[self.fiedler_idx+1] - self.lambda0:.6f}")

    def _verify_laplacian(self):
        """Verify matrix is a valid graph Laplacian."""
        # Check symmetry
        if not np.allclose(self.L0, self.L0.T):
            warnings.warn("Laplacian is not symmetric! Symmetrizing...")
            self.L0 = (self.L0 + self.L0.T) / 2

        # Check row sums (should be zero for proper Laplacian)
        row_sums = np.sum(self.L0, axis=1)
        if not np.allclose(row_sums, 0, atol=1e-8):
            warnings.warn(f"Row sums not zero! Max deviation: {np.max(np.abs(row_sums))}")

    def _find_fiedler_index(self) -> int:
        """
        Find index of Fiedler value (first non-zero eigenvalue).

        Important: Don't blindly pick index 1!
        Literature: Professor feedback - "make sure second eigen value is not zero"
        """
        # Count zero eigenvalues (connected components)
        zero_count = np.sum(np.abs(self.eigvals) < self.tol)

        if zero_count == 0:
            warnings.warn("No zero eigenvalue found! Graph may not be properly formed.")
            return 0
        elif zero_count > 1:
            warnings.warn(f"Found {zero_count} zero eigenvalues! Graph is disconnected.")

        # Return first non-zero eigenvalue
        fiedler_idx = zero_count  # This is the first non-zero

        print(f"  Zero eigenvalues: {zero_count}")
        print(f"  First 5 eigenvalues: {self.eigvals[:5]}")

        return fiedler_idx

    def generate_laplacian_noise(self, epsilon: float, sensitivity: float = 1.0) -> np.ndarray:
        """
        Generate Laplacian-distributed noise scaled by Frobenius norm.

        Literature: Professor feedback points 1-3
        - Use Laplacian distribution (not Gaussian)
        - Scale by Frobenius norm of original Laplacian
        - Sensitivity parameter controls privacy

        Args:
            epsilon: Privacy parameter (smaller = more privacy, more noise)
            sensitivity: Sensitivity of the query (default 1.0)

        Returns:
            Noise matrix (symmetric)
        """
        # Frobenius norm of original Laplacian
        # ||L||_F = sqrt(sum of all squared elements)
        frobenius_norm = np.linalg.norm(self.L0, 'fro')

        # Laplacian scale parameter for differential privacy
        # scale = sensitivity / epsilon
        scale = (sensitivity * frobenius_norm) / epsilon

        # Generate Laplacian noise for each entry
        noise = laplace.rvs(loc=0, scale=scale, size=(self.n, self.n))

        # Make symmetric (Laplacian must be symmetric)
        noise = (noise + noise.T) / 2

        # Zero diagonal (Laplacian has zero row sums)
        # Adjust diagonal to maintain zero row sums
        row_sums = np.sum(noise, axis=1)
        np.fill_diagonal(noise, noise.diagonal() - row_sums)

        print(f"✓ Laplacian noise generated (ε={epsilon:.3f}, scale={scale:.6f})")
        print(f"  Frobenius norm of noise: {np.linalg.norm(noise, 'fro'):.6f}")

        return noise

    def first_order_eigenvalue_perturbation(self, noise: np.ndarray) -> float:
        """
        Compute first-order perturbation to Fiedler value.

        Literature: Greenbaum et al. (2020), Theorem 2.1, Equation 2.1, page 465-466

        For L(κ) = L₀ + κ*L₁, the eigenvalue perturbation is:
        λ₁ = (y₀ᴴ L₁ x₀) / (y₀ᴴ x₀)

        Args:
            noise: Perturbation matrix L₁

        Returns:
            First-order correction λ₁
        """
        numerator = np.dot(self.y0.conj(), np.dot(noise, self.x0))
        denominator = np.dot(self.y0.conj(), self.x0)

        lambda1 = numerator / denominator

        return np.real(lambda1)  # Should be real for symmetric matrices

    def first_order_eigenvector_perturbation(self, noise: np.ndarray,
                                            lambda1: float) -> np.ndarray:
        """
        Compute first-order perturbation to Fiedler vector.

        Literature: Greenbaum et al. (2020), Theorem 3.1, Equations 3.3-3.4, page 468-470

        The eigenvector correction x₁ satisfies:
        (L₀ - λ₀I)x₁ = (λ₁I - L₁)x₀ + αx₀

        Solution using complementary projector P₁:
        x₁ = P₁(L₀ - λ₀I)⁻¹ P₁(λ₁I - L₁)x₀

        where P₁ = (I - x₀y₀ᴴ) / (y₀ᴴ x₀)

        Args:
            noise: Perturbation matrix L₁
            lambda1: First-order eigenvalue correction

        Returns:
            First-order eigenvector correction x₁
        """
        # Complementary projector P₁ (projects onto complement of eigenspace)
        # Literature: Section 3, page 469
        y0_x0 = np.dot(self.y0.conj(), self.x0)
        P1 = np.eye(self.n) - np.outer(self.x0, self.y0.conj()) / y0_x0

        # Right-hand side: (λ₁I - L₁)x₀
        rhs = lambda1 * self.x0 - np.dot(noise, self.x0)

        # Apply P₁ to RHS
        rhs_projected = np.dot(P1, rhs)

        # Solve (L₀ - λ₀I)z = rhs_projected
        # Note: (L₀ - λ₀I) is singular, but P₁ projects onto its range

        # Use pseudoinverse for numerical stability
        L_shifted = self.L0 - self.lambda0 * np.eye(self.n)

        # Apply P₁ to L_shifted to get full-rank system on subspace
        L_projected = np.dot(P1, np.dot(L_shifted, P1))

        # Add small regularization for numerical stability
        reg = 1e-10
        L_reg = L_projected + reg * np.eye(self.n)

        # Solve system
        z = np.linalg.solve(L_reg, rhs_projected)

        # Project result back
        x1 = np.dot(P1, z)

        return x1

    def analyze_perturbation(self, epsilon: float, kappa: float = 1.0,
                           num_samples: int = 100) -> Dict:
        """
        Complete perturbation analysis with Monte Carlo validation.

        Args:
            epsilon: Privacy parameter
            kappa: Perturbation scale parameter
            num_samples: Number of Monte Carlo samples for validation

        Returns:
            Dictionary with analysis results
        """
        results = {
            'epsilon': epsilon,
            'kappa': kappa,
            'lambda0': self.lambda0,
            'theoretical': {},
            'numerical': {},
            'monte_carlo': {}
        }

        # Generate noise
        L1 = self.generate_laplacian_noise(epsilon)

        # Theoretical first-order predictions
        lambda1_theory = self.first_order_eigenvalue_perturbation(L1)
        x1_theory = self.first_order_eigenvector_perturbation(L1, lambda1_theory)

        results['theoretical']['lambda1'] = lambda1_theory
        results['theoretical']['lambda_predicted'] = self.lambda0 + kappa * lambda1_theory
        results['theoretical']['x1_norm'] = np.linalg.norm(x1_theory)

        # Numerical validation (compute actual perturbed eigenvalue)
        L_perturbed = self.L0 + kappa * L1
        eigvals_perturbed, eigvecs_perturbed = eigh(L_perturbed)

        # Find perturbed Fiedler value (should be near same index)
        fiedler_idx_perturbed = self._find_fiedler_index_perturbed(eigvals_perturbed)
        lambda_actual = eigvals_perturbed[fiedler_idx_perturbed]
        x_actual = eigvecs_perturbed[:, fiedler_idx_perturbed]

        # Align sign (eigenvectors have sign ambiguity)
        if np.dot(x_actual, self.x0) < 0:
            x_actual = -x_actual

        results['numerical']['lambda_actual'] = lambda_actual
        results['numerical']['fiedler_idx'] = fiedler_idx_perturbed

        # Compute errors
        lambda_error = abs(lambda_actual - (self.lambda0 + kappa * lambda1_theory))
        lambda_rel_error = lambda_error / abs(self.lambda0 + 1e-10)

        x_diff = x_actual - (self.x0 + kappa * x1_theory)
        x_error = np.linalg.norm(x_diff)

        results['errors'] = {
            'lambda_absolute': lambda_error,
            'lambda_relative': lambda_rel_error,
            'x_l2_distance': x_error,
            'x_cosine_similarity': np.dot(x_actual, self.x0) / (np.linalg.norm(x_actual) * np.linalg.norm(self.x0))
        }

        # Monte Carlo validation (multiple noise realizations)
        print(f"\n✓ Running Monte Carlo validation ({num_samples} samples)...")
        lambda_samples = []
        x_distances = []

        for i in range(num_samples):
            L1_sample = self.generate_laplacian_noise(epsilon)
            L_sample = self.L0 + kappa * L1_sample
            eigvals_sample, eigvecs_sample = eigh(L_sample)

            fiedler_idx_sample = self._find_fiedler_index_perturbed(eigvals_sample)
            lambda_samples.append(eigvals_sample[fiedler_idx_sample])

            x_sample = eigvecs_sample[:, fiedler_idx_sample]
            if np.dot(x_sample, self.x0) < 0:
                x_sample = -x_sample
            x_distances.append(np.linalg.norm(x_sample - self.x0))

        results['monte_carlo'] = {
            'lambda_mean': np.mean(lambda_samples),
            'lambda_std': np.std(lambda_samples),
            'lambda_samples': lambda_samples,
            'x_distance_mean': np.mean(x_distances),
            'x_distance_std': np.std(x_distances),
            'x_distance_samples': x_distances
        }

        # Print summary
        self._print_results(results)

        return results

    def _find_fiedler_index_perturbed(self, eigvals: np.ndarray) -> int:
        """Find Fiedler index in perturbed eigenvalues."""
        zero_count = np.sum(np.abs(eigvals) < self.tol)
        return zero_count if zero_count < len(eigvals) else len(eigvals) - 1

    def _print_results(self, results: Dict):
        """Print formatted analysis results."""
        print("\n" + "="*70)
        print("PERTURBATION ANALYSIS RESULTS")
        print("="*70)
        print(f"Privacy parameter ε: {results['epsilon']:.4f}")
        print(f"Perturbation scale κ: {results['kappa']:.4f}")
        print(f"\nOriginal Fiedler value λ₀: {results['lambda0']:.6f}")
        print(f"\nTHEORETICAL (First-Order Perturbation Theory):")
        print(f"  λ₁ (correction): {results['theoretical']['lambda1']:.6f}")
        print(f"  λ(κ) predicted: {results['theoretical']['lambda_predicted']:.6f}")
        print(f"  ||x₁|| (correction norm): {results['theoretical']['x1_norm']:.6f}")
        print(f"\nNUMERICAL (Actual Computation):")
        print(f"  λ(κ) actual: {results['numerical']['lambda_actual']:.6f}")
        print(f"\nERRORS:")
        print(f"  |λ_actual - λ_predicted|: {results['errors']['lambda_absolute']:.6e}")
        print(f"  Relative error: {results['errors']['lambda_relative']:.2%}")
        print(f"  ||x_actual - x_predicted||: {results['errors']['x_l2_distance']:.6e}")
        print(f"  Cosine similarity: {results['errors']['x_cosine_similarity']:.6f}")
        print(f"\nMONTE CARLO ({len(results['monte_carlo']['lambda_samples'])} samples):")
        print(f"  λ mean ± std: {results['monte_carlo']['lambda_mean']:.6f} ± {results['monte_carlo']['lambda_std']:.6f}")
        print(f"  x distance mean ± std: {results['monte_carlo']['x_distance_mean']:.6f} ± {results['monte_carlo']['x_distance_std']:.6f}")
        print("="*70)

    def plot_perturbation_analysis(self, epsilon_range: np.ndarray, kappa: float = 1.0):
        """
        Plot Fiedler value and vector perturbations across privacy levels.

        Literature: Section 5 (Computational Verification), page 476-477
        """
        lambda_predicted = []
        lambda_actual = []
        x_distances = []
        condition_numbers = []

        for eps in epsilon_range:
            L1 = self.generate_laplacian_noise(eps)
            lambda1 = self.first_order_eigenvalue_perturbation(L1)

            # Theoretical
            lambda_pred = self.lambda0 + kappa * lambda1
            lambda_predicted.append(lambda_pred)

            # Numerical
            L_perturbed = self.L0 + kappa * L1
            eigvals_p, eigvecs_p = eigh(L_perturbed)
            idx = self._find_fiedler_index_perturbed(eigvals_p)
            lambda_actual.append(eigvals_p[idx])

            x_p = eigvecs_p[:, idx]
            if np.dot(x_p, self.x0) < 0:
                x_p = -x_p
            x_distances.append(np.linalg.norm(x_p - self.x0))

            # Condition number (1 / |y₀ᴴx₀|)
            # Literature: Section 2, Remark 2.2, page 466
            cond = 1.0 / abs(np.dot(self.y0.conj(), self.x0))
            condition_numbers.append(cond)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Eigenvalue perturbation
        axes[0, 0].plot(epsilon_range, lambda_predicted, 'b-', label='Theory', linewidth=2)
        axes[0, 0].plot(epsilon_range, lambda_actual, 'r--', label='Actual', linewidth=2)
        axes[0, 0].axhline(self.lambda0, color='k', linestyle=':', label='λ₀')
        axes[0, 0].set_xlabel('Privacy ε')
        axes[0, 0].set_ylabel('Fiedler Value λ(κ)')
        axes[0, 0].set_title('Eigenvalue Perturbation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Prediction error
        errors = np.abs(np.array(lambda_actual) - np.array(lambda_predicted))
        axes[0, 1].semilogy(epsilon_range, errors, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Privacy ε')
        axes[0, 1].set_ylabel('|λ_actual - λ_predicted|')
        axes[0, 1].set_title('First-Order Approximation Error')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Eigenvector distance
        axes[1, 0].plot(epsilon_range, x_distances, 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Privacy ε')
        axes[1, 0].set_ylabel('||x(κ) - x₀||')
        axes[1, 0].set_title('Fiedler Vector Perturbation')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Privacy-Utility Tradeoff
        # Utility = 1 / (eigenvector distance)
        utility = 1.0 / (np.array(x_distances) + 1e-6)
        axes[1, 1].plot(epsilon_range, utility, 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Privacy ε (higher = less privacy)')
        axes[1, 1].set_ylabel('Utility (higher = better)')
        axes[1, 1].set_title('Privacy-Utility Tradeoff')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def main_demo():
    """Demonstration using a simple graph."""
    # Create a simple connected graph Laplacian
    n = 10
    np.random.seed(42)

    # Random adjacency matrix (make connected)
    A = np.random.rand(n, n)
    A = (A + A.T) / 2  # Symmetric
    A = (A > 0.5).astype(float)  # Threshold
    np.fill_diagonal(A, 0)  # No self-loops

    # Ensure connected (add path if needed)
    for i in range(n-1):
        A[i, i+1] = 1
        A[i+1, i] = 1

    # Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A

    print("="*70)
    print("FIEDLER PERTURBATION ANALYSIS DEMO")
    print("="*70)

    # Initialize analyzer
    analyzer = FiedlerPerturbationAnalyzer(L)

    # Single perturbation analysis
    print("\n--- Single Perturbation Analysis ---")
    results = analyzer.analyze_perturbation(epsilon=1.0, kappa=0.1, num_samples=50)

    # Range analysis
    print("\n--- Privacy Parameter Sweep ---")
    epsilon_range = np.linspace(0.1, 2.0, 20)
    fig = analyzer.plot_perturbation_analysis(epsilon_range, kappa=0.1)
    plt.savefig('fiedler_perturbation_plots.png', dpi=150, bbox_inches='tight')
    print("✓ Plots saved to fiedler_perturbation_plots.png")

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main_demo()
