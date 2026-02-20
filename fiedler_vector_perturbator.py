"""
Direct Fiedler Vector Perturbation for Graph Privacy
=====================================================

This module implements differential privacy by perturbing the Fiedler vector
DIRECTLY rather than the Laplacian matrix. This approach may preserve graph
utility better for spectral clustering tasks.

Key Insight:
-----------
Instead of: Perturb L → Compute Fiedler vector
We do:      Compute Fiedler vector → Perturb vector → Use for clustering

Advantages:
- Lower dimensional (n vs n²)
- More direct control over clustering utility
- Potentially better SNR (vector has n elements vs matrix has n² elements)

Research References:
-------------------
[1] Blocki et al. (2013): "Differentially Private Data Analysis of Social Networks"
[2] Kasiviswanathan et al. (2013): "Analyzing Graphs with Node Differential Privacy"
[3] Day et al. (2016): "Publishing Graph Degree Distribution with Node DP"

Author: Mahdi Mohammad Shibli
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.stats import laplace, norm
from enum import Enum
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


class VectorPerturbationMethod(Enum):
    """Perturbation methods for Fiedler vector."""
    LAPLACE_VECTOR = "laplace_vector"              # Standard Laplace on each element
    GAUSSIAN_VECTOR = "gaussian_vector"            # Gaussian noise per element
    EXPONENTIAL_SPHERE = "exponential_sphere"      # Sample from unit sphere
    TRUNCATED_LAPLACE = "truncated_laplace"        # Bounded Laplace noise
    DISCRETE_BINS = "discrete_bins"                # Discretize then perturb


@dataclass
class VectorPerturbationResult:
    """Results from Fiedler vector perturbation."""
    original_vector: np.ndarray
    perturbed_vector: np.ndarray
    noise_vector: np.ndarray
    noise_scale: float
    epsilon_used: float
    sensitivity: float
    method: str
    original_fiedler_value: float
    metadata: Dict


class FiedlerVectorPerturbation:
    """
    Framework for perturbing Fiedler vectors with differential privacy.

    Theoretical Foundation:
    ----------------------
    The Fiedler vector (eigenvector of λ₂) is used for spectral clustering.
    By perturbing only this vector, we:
    1. Reduce dimensionality: n elements vs n² matrix elements
    2. Preserve graph structure better
    3. Get better SNR (less noise needed)

    Sensitivity Analysis:
    --------------------
    For Fiedler vector v₂ ∈ ℝⁿ with ||v₂|| = 1:

    Adding/removing one edge changes at most 2 elements significantly.
    Global sensitivity Δf ≈ O(1/√n) for normalized vectors.

    However, for conservative DP, we use Δf = 2 (worst case).
    """

    def __init__(self,
                 epsilon_total: float = 1.0,
                 delta: float = 1e-5,
                 rose_threshold: float = 5.0,
                 vector_sensitivity: float = 2.0):
        """
        Initialize Fiedler vector perturbation engine.

        Args:
            epsilon_total: Total privacy budget
            delta: Failure probability for (ε,δ)-DP
            rose_threshold: SNR threshold for utility
            vector_sensitivity: Sensitivity of Fiedler vector (default: 2.0)
        """
        self.epsilon_total = epsilon_total
        self.epsilon_remaining = epsilon_total
        self.delta = delta
        self.rose_threshold = rose_threshold
        self.vector_sensitivity = vector_sensitivity

        print(f"Fiedler Vector Perturbation Engine Initialized:")
        print(f"  ε_total: {epsilon_total}")
        print(f"  Vector sensitivity: {vector_sensitivity}")
        print(f"  Rose threshold: {rose_threshold}")

    def extract_fiedler_vector(self, L: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Extract Fiedler vector from Laplacian.

        Args:
            L: Laplacian matrix

        Returns:
            (fiedler_vector, fiedler_value, fiedler_index)
        """
        eigvals, eigvecs = eigh(L)

        # Find first non-zero eigenvalue (Fiedler value)
        zero_tol = 1e-10
        zero_count = np.sum(np.abs(eigvals) < zero_tol)

        if zero_count >= len(eigvals):
            raise ValueError("Graph is completely disconnected!")

        fiedler_idx = zero_count
        fiedler_value = eigvals[fiedler_idx]
        fiedler_vector = eigvecs[:, fiedler_idx]

        # Normalize (unit norm)
        fiedler_vector = fiedler_vector / np.linalg.norm(fiedler_vector)

        return fiedler_vector, fiedler_value, fiedler_idx

    def perturb(self,
                fiedler_vector: np.ndarray,
                method: VectorPerturbationMethod,
                epsilon_allocated: float,
                fiedler_value: float,
                sensitivity: Optional[float] = None) -> VectorPerturbationResult:
        """
        Apply DP perturbation to Fiedler vector.

        Args:
            fiedler_vector: Original Fiedler vector (unit norm)
            method: Perturbation method to use
            epsilon_allocated: Privacy budget for this operation
            fiedler_value: Original Fiedler value (for metadata)
            sensitivity: Override default sensitivity

        Returns:
            VectorPerturbationResult with perturbed vector
        """
        if sensitivity is None:
            sensitivity = self.vector_sensitivity

        if epsilon_allocated > self.epsilon_remaining:
            raise ValueError(
                f"Insufficient budget. Requested: {epsilon_allocated}, "
                f"Remaining: {self.epsilon_remaining}"
            )

        # Route to method
        method_map = {
            VectorPerturbationMethod.LAPLACE_VECTOR: self._laplace_vector,
            VectorPerturbationMethod.GAUSSIAN_VECTOR: self._gaussian_vector,
            VectorPerturbationMethod.EXPONENTIAL_SPHERE: self._exponential_sphere,
            VectorPerturbationMethod.TRUNCATED_LAPLACE: self._truncated_laplace,
            VectorPerturbationMethod.DISCRETE_BINS: self._discrete_bins,
        }

        perturb_func = method_map[method]
        result = perturb_func(fiedler_vector, epsilon_allocated, sensitivity, fiedler_value)

        # Update budget
        self.epsilon_remaining -= epsilon_allocated

        return result

    def _laplace_vector(self,
                       vector: np.ndarray,
                       epsilon: float,
                       sensitivity: float,
                       fiedler_value: float) -> VectorPerturbationResult:
        """
        Standard Laplace mechanism on vector elements.

        Literature: Dwork & Roth (2014)

        For each element v_i, add noise ~ Lap(Δf/ε)

        Key advantage: Lower dimensional than matrix perturbation
        n noise draws vs n² for full matrix
        """
        n = len(vector)

        # Noise scale for each element
        noise_scale = sensitivity / epsilon

        # Generate Laplace noise for each element
        noise = laplace.rvs(loc=0, scale=noise_scale, size=n)

        # Perturb
        perturbed = vector + noise

        # Optional: Re-normalize to unit norm (preserves interpretation)
        # perturbed = perturbed / np.linalg.norm(perturbed)

        return VectorPerturbationResult(
            original_vector=vector.copy(),
            perturbed_vector=perturbed,
            noise_vector=noise,
            noise_scale=noise_scale,
            epsilon_used=epsilon,
            sensitivity=sensitivity,
            method="Laplace Vector",
            original_fiedler_value=fiedler_value,
            metadata={
                'vector_dimension': n,
                'theoretical_variance_per_element': 2 * noise_scale**2,
                'normalized': False
            }
        )

    def _gaussian_vector(self,
                        vector: np.ndarray,
                        epsilon: float,
                        sensitivity: float,
                        fiedler_value: float) -> VectorPerturbationResult:
        """
        Gaussian mechanism for (ε,δ)-DP on vectors.

        Literature: Dwork & Roth (2014), Section 3.5
        """
        n = len(vector)

        # Gaussian noise scale for (ε,δ)-DP
        noise_scale = (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / self.delta))

        # Generate Gaussian noise
        noise = norm.rvs(loc=0, scale=noise_scale, size=n)

        # Perturb
        perturbed = vector + noise

        return VectorPerturbationResult(
            original_vector=vector.copy(),
            perturbed_vector=perturbed,
            noise_vector=noise,
            noise_scale=noise_scale,
            epsilon_used=epsilon,
            sensitivity=sensitivity,
            method="Gaussian Vector",
            original_fiedler_value=fiedler_value,
            metadata={
                'vector_dimension': n,
                'delta': self.delta,
                'privacy_type': f'({epsilon}, {self.delta})-DP',
                'noise_variance_per_element': noise_scale**2
            }
        )

    def _exponential_sphere(self,
                          vector: np.ndarray,
                          epsilon: float,
                          sensitivity: float,
                          fiedler_value: float) -> VectorPerturbationResult:
        """
        Exponential mechanism sampling from unit sphere.

        Key idea: Sample vectors on unit sphere with probability
        proportional to similarity to original vector.

        This preserves unit norm naturally.
        """
        n = len(vector)

        # Generate candidate vectors on unit sphere
        num_candidates = 100
        candidates = []

        for _ in range(num_candidates):
            # Random vector on unit sphere
            candidate = np.random.randn(n)
            candidate = candidate / np.linalg.norm(candidate)
            candidates.append(candidate)

        # Quality function: dot product with original (cosine similarity)
        qualities = np.array([np.dot(vector, c) for c in candidates])

        # Exponential mechanism probabilities
        scores = (epsilon / (2 * sensitivity)) * qualities
        scores -= scores.max()  # Numerical stability
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()

        # Sample candidate
        selected_idx = np.random.choice(num_candidates, p=probabilities)
        perturbed = candidates[selected_idx]

        # Compute effective noise
        noise = perturbed - vector
        noise_scale = np.std(noise)

        return VectorPerturbationResult(
            original_vector=vector.copy(),
            perturbed_vector=perturbed,
            noise_vector=noise,
            noise_scale=noise_scale,
            epsilon_used=epsilon,
            sensitivity=sensitivity,
            method="Exponential Sphere",
            original_fiedler_value=fiedler_value,
            metadata={
                'num_candidates': num_candidates,
                'selected_quality': qualities[selected_idx],
                'selection_probability': probabilities[selected_idx],
                'preserves_unit_norm': True
            }
        )

    def _truncated_laplace(self,
                          vector: np.ndarray,
                          epsilon: float,
                          sensitivity: float,
                          fiedler_value: float) -> VectorPerturbationResult:
        """
        Truncated Laplace mechanism with bounded noise.

        Truncates noise to prevent extreme perturbations that would
        destroy clustering structure.
        """
        n = len(vector)

        # Base noise scale
        noise_scale = sensitivity / epsilon

        # Generate Laplace noise
        noise = laplace.rvs(loc=0, scale=noise_scale, size=n)

        # Truncate to [-3σ, 3σ] (covers 99.7% of distribution)
        truncation_bound = 3 * noise_scale
        noise = np.clip(noise, -truncation_bound, truncation_bound)

        # Perturb
        perturbed = vector + noise

        return VectorPerturbationResult(
            original_vector=vector.copy(),
            perturbed_vector=perturbed,
            noise_vector=noise,
            noise_scale=noise_scale,
            epsilon_used=epsilon,
            sensitivity=sensitivity,
            method="Truncated Laplace",
            original_fiedler_value=fiedler_value,
            metadata={
                'truncation_bound': truncation_bound,
                'truncation_sigma_multiplier': 3,
                'proportion_truncated': np.mean(np.abs(noise) >= truncation_bound * 0.99)
            }
        )

    def _discrete_bins(self,
                      vector: np.ndarray,
                      epsilon: float,
                      sensitivity: float,
                      fiedler_value: float) -> VectorPerturbationResult:
        """
        Discretize vector into bins, then perturb bin assignments.

        This is a discrete alternative that may preserve clustering better.
        """
        n = len(vector)

        # Discretize into bins
        num_bins = 10
        bins = np.linspace(vector.min(), vector.max(), num_bins + 1)
        bin_indices = np.digitize(vector, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Perturb bin indices with geometric mechanism
        # Geometric distribution for discrete DP
        perturbed_bins = bin_indices.copy()

        for i in range(n):
            # Probability of moving k bins away: exp(-ε|k|/Δf)
            max_shift = 3  # Allow shifts up to 3 bins
            shifts = np.arange(-max_shift, max_shift + 1)

            # Probabilities
            probs = np.exp(-epsilon * np.abs(shifts) / sensitivity)
            probs /= probs.sum()

            # Sample shift
            shift = np.random.choice(shifts, p=probs)
            perturbed_bins[i] = np.clip(perturbed_bins[i] + shift, 0, num_bins - 1)

        # Map back to continuous values (bin centers)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        perturbed = bin_centers[perturbed_bins]

        # Compute noise
        noise = perturbed - vector
        noise_scale = np.std(noise)

        return VectorPerturbationResult(
            original_vector=vector.copy(),
            perturbed_vector=perturbed,
            noise_vector=noise,
            noise_scale=noise_scale,
            epsilon_used=epsilon,
            sensitivity=sensitivity,
            method="Discrete Bins",
            original_fiedler_value=fiedler_value,
            metadata={
                'num_bins': num_bins,
                'max_shift': max_shift,
                'bins_changed': np.sum(bin_indices != perturbed_bins),
                'proportion_changed': np.mean(bin_indices != perturbed_bins)
            }
        )

    def validate_utility(self,
                        result: VectorPerturbationResult) -> Dict:
        """
        Validate utility of perturbed Fiedler vector.

        Metrics:
        -------
        1. SNR (Rose Criterion)
        2. Cosine similarity (angle preservation)
        3. Euclidean distance
        4. Partition agreement (clustering preserved?)
        5. Correlation coefficient
        """
        original = result.original_vector
        perturbed = result.perturbed_vector
        noise_scale = result.noise_scale

        validation = {}

        # 1. SNR (Rose Criterion)
        signal = np.mean(np.abs(original))
        noise = noise_scale
        snr = signal / (noise + 1e-10)

        validation['snr'] = snr
        validation['meets_rose_criterion'] = snr >= self.rose_threshold
        validation['rose_threshold'] = self.rose_threshold

        # 2. Cosine Similarity (angle between vectors)
        cosine_sim = np.dot(original, perturbed) / (
            np.linalg.norm(original) * np.linalg.norm(perturbed) + 1e-10
        )
        validation['cosine_similarity'] = cosine_sim
        validation['angle_degrees'] = np.degrees(np.arccos(np.clip(cosine_sim, -1, 1)))

        # 3. Euclidean Distance
        euclidean_dist = np.linalg.norm(original - perturbed)
        validation['euclidean_distance'] = euclidean_dist
        validation['relative_distance'] = euclidean_dist / (np.linalg.norm(original) + 1e-10)

        # 4. Partition Agreement (for binary clustering)
        # Fiedler vector is used by checking sign for 2-way partition
        partition_original = original >= 0
        partition_perturbed = perturbed >= 0
        agreement = np.mean(partition_original == partition_perturbed)
        validation['partition_agreement'] = agreement
        validation['clustering_preserved'] = agreement >= 0.85

        # 5. Correlation Coefficient
        correlation = np.corrcoef(original, perturbed)[0, 1]
        validation['correlation'] = correlation
        validation['high_correlation'] = correlation >= 0.95

        # 6. Overall Assessment
        validation['overall_pass'] = (
            validation['meets_rose_criterion'] and
            validation['clustering_preserved'] and
            validation['high_correlation']
        )

        return validation

    def reset_budget(self, new_epsilon: Optional[float] = None):
        """Reset privacy budget."""
        if new_epsilon is not None:
            self.epsilon_total = new_epsilon
        self.epsilon_remaining = self.epsilon_total


def run_fiedler_vector_analysis(L: np.ndarray,
                                graph_name: str = "Graph",
                                epsilon_range: np.ndarray = None,
                                amplification_factors: list = None,
                                methods_to_test: list = None,
                                base_sensitivity: float = 2.0):
    """
    Comprehensive analysis of direct Fiedler vector perturbation.

    Args:
        L: Original Laplacian matrix
        graph_name: Name for plots
        epsilon_range: Privacy budgets to test
        amplification_factors: Signal scaling factors
        methods_to_test: Perturbation methods
        base_sensitivity: Base sensitivity value

    Returns:
        DataFrame with results
    """
    # Defaults
    if epsilon_range is None:
        epsilon_range = np.linspace(0.1, 2.0, 10)

    if amplification_factors is None:
        amplification_factors = [1.0, 10.0, 50.0, 100.0]

    if methods_to_test is None:
        methods_to_test = [
            VectorPerturbationMethod.LAPLACE_VECTOR,
            VectorPerturbationMethod.GAUSSIAN_VECTOR,
            VectorPerturbationMethod.EXPONENTIAL_SPHERE,
            VectorPerturbationMethod.TRUNCATED_LAPLACE,
        ]

    print("="*80)
    print(f"FIEDLER VECTOR PERTURBATION ANALYSIS: {graph_name}")
    print("="*80)
    print(f"Approach: Perturb Fiedler VECTOR directly (n elements)")
    print(f"Compare to: Matrix perturbation (n² elements)")
    print(f"Expected advantage: Better SNR (lower dimensionality)")
    print("="*80)

    all_results = []

    for amp_idx, amp_factor in enumerate(amplification_factors):
        print(f"\n[{amp_idx+1}/{len(amplification_factors)}] Amplification: {amp_factor}x")

        # Scale Laplacian
        L_amplified = L * amp_factor
        current_sensitivity = base_sensitivity * amp_factor

        # Extract Fiedler vector from amplified Laplacian
        perturbator = FiedlerVectorPerturbation(
            epsilon_total=max(epsilon_range),
            vector_sensitivity=current_sensitivity,
            rose_threshold=5.0
        )

        fiedler_vec, fiedler_val, fiedler_idx = perturbator.extract_fiedler_vector(L_amplified)

        print(f"  Fiedler value: {fiedler_val:.4f}")
        print(f"  Vector dimension: {len(fiedler_vec)}")
        print(f"  Mean |element|: {np.mean(np.abs(fiedler_vec)):.4f}")

        for eps_idx, eps in enumerate(epsilon_range):
            if eps_idx % 3 == 0:
                print(f"  ε = {eps:.2f}...", end=" ", flush=True)

            for method in methods_to_test:
                # Reset budget
                perturbator.reset_budget(new_epsilon=eps)

                # Perturb VECTOR
                result = perturbator.perturb(
                    fiedler_vec,
                    method=method,
                    epsilon_allocated=eps,
                    fiedler_value=fiedler_val,
                    sensitivity=current_sensitivity
                )

                # Validate utility
                utility = perturbator.validate_utility(result)

                # Store results
                all_results.append({
                    'amplification_factor': amp_factor,
                    'epsilon': eps,
                    'method': method.value,
                    'sensitivity': current_sensitivity,
                    # Utility metrics
                    'snr': utility['snr'],
                    'cosine_similarity': utility['cosine_similarity'],
                    'euclidean_distance': utility['euclidean_distance'],
                    'partition_agreement': utility['partition_agreement'],
                    'correlation': utility['correlation'],
                    'rose_criterion_met': utility['meets_rose_criterion'],
                    # Vector info
                    'fiedler_value': fiedler_val,
                    'vector_dimension': len(fiedler_vec),
                    'noise_scale': result.noise_scale,
                    'mean_abs_element': np.mean(np.abs(fiedler_vec)),
                })

        print()  # New line

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return pd.DataFrame(all_results)


def plot_fiedler_vector_results(df: pd.DataFrame, graph_name: str = "Graph",
                                save_path: Optional[str] = None):
    """
    Visualize Fiedler vector perturbation results.

    Args:
        df: Results DataFrame
        graph_name: Name for plots
        save_path: Optional path to save figures
    """
    amplification_factors = sorted(df['amplification_factor'].unique())
    methods = df['method'].unique()

    for amp_factor in amplification_factors:
        data = df[df['amplification_factor'] == amp_factor]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"{graph_name}: Fiedler VECTOR Perturbation (Amp={amp_factor}x)",
                    fontsize=16, fontweight='bold')

        for method in methods:
            method_data = data[data['method'] == method]
            label = method.replace('_', ' ').title()

            # Plot 1: Euclidean Distance
            axes[0, 0].plot(method_data['epsilon'],
                          method_data['euclidean_distance'],
                          marker='o', label=label, linewidth=2)

            # Plot 2: SNR
            axes[0, 1].plot(method_data['epsilon'],
                          method_data['snr'],
                          marker='s', linewidth=2)

            # Plot 3: Cosine Similarity
            axes[0, 2].plot(method_data['epsilon'],
                          method_data['cosine_similarity'],
                          marker='^', linewidth=2)

            # Plot 4: Partition Agreement
            axes[1, 0].plot(method_data['epsilon'],
                          method_data['partition_agreement'],
                          marker='d', linewidth=2)

            # Plot 5: Correlation
            axes[1, 1].plot(method_data['epsilon'],
                          method_data['correlation'],
                          marker='x', linewidth=2)

            # Plot 6: Comparison metric
            # SNR / Vector dimension (efficiency metric)
            efficiency = method_data['snr'] / method_data['vector_dimension']
            axes[1, 2].plot(method_data['epsilon'],
                          efficiency,
                          marker='v', linewidth=2)

        # Formatting
        axes[0, 0].set_title("A. Vector Euclidean Distance", fontweight='bold')
        axes[0, 0].set_ylabel("||v - v'||₂")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title("B. Signal-to-Noise Ratio", fontweight='bold')
        axes[0, 1].set_ylabel("SNR")
        axes[0, 1].axhline(y=5.0, color='r', linestyle=':', linewidth=2, label='Rose')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].set_title("C. Cosine Similarity", fontweight='bold')
        axes[0, 2].set_ylabel("cos(θ)")
        axes[0, 2].axhline(y=0.9, color='g', linestyle=':', linewidth=2, label='Target')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].set_title("D. Partition Agreement", fontweight='bold')
        axes[1, 0].set_ylabel("Agreement Ratio")
        axes[1, 0].set_xlabel("Privacy Budget ε")
        axes[1, 0].axhline(y=0.85, color='orange', linestyle=':', linewidth=2, label='Target')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title("E. Vector Correlation", fontweight='bold')
        axes[1, 1].set_ylabel("Pearson ρ")
        axes[1, 1].set_xlabel("Privacy Budget ε")
        axes[1, 1].axhline(y=0.95, color='purple', linestyle=':', linewidth=2, label='Target')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].set_title("F. SNR per Vector Element", fontweight='bold')
        axes[1, 2].set_ylabel("SNR / n (efficiency)")
        axes[1, 2].set_xlabel("Privacy Budget ε")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            filename = f"{save_path}_vector_amp{int(amp_factor)}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {filename}")

        plt.show()


def compare_vector_vs_matrix_perturbation(L: np.ndarray, epsilon: float = 1.0):
    """
    Compare vector perturbation vs matrix perturbation approaches.

    Shows SNR advantage of perturbing n-dimensional vector
    vs n²-dimensional matrix.
    """
    n = L.shape[0]

    print("="*80)
    print("VECTOR vs MATRIX PERTURBATION COMPARISON")
    print("="*80)

    # Vector approach
    vec_perturbator = FiedlerVectorPerturbation(epsilon_total=epsilon)
    fiedler_vec, fiedler_val, _ = vec_perturbator.extract_fiedler_vector(L)

    vec_result = vec_perturbator.perturb(
        fiedler_vec,
        VectorPerturbationMethod.LAPLACE_VECTOR,
        epsilon_allocated=epsilon,
        fiedler_value=fiedler_val
    )

    vec_utility = vec_perturbator.validate_utility(vec_result)

    # Matrix approach (from your code)
    from comprehensive_perturbation_analysis import GraphLaplacianPerturbation, PerturbationMethod

    mat_perturbator = GraphLaplacianPerturbation(epsilon_total=epsilon)
    mat_result = mat_perturbator.perturb(
        L,
        method=PerturbationMethod.LAPLACE_STANDARD,
        epsilon_allocated=epsilon
    )

    mat_utility = mat_perturbator.validate_utility(
        L, mat_result.perturbed_laplacian, mat_result.noise_scale
    )

    # Compare
    print(f"\nVector Approach (n={n} elements):")
    print(f"  SNR: {vec_utility['snr']:.4f}")
    print(f"  Partition agreement: {vec_utility['partition_agreement']:.2%}")
    print(f"  Cosine similarity: {vec_utility['cosine_similarity']:.4f}")

    print(f"\nMatrix Approach (n²={n**2} elements):")
    print(f"  SNR: {mat_utility.get('snr', 0):.4f}")
    print(f"  Structural correlation: {mat_utility['structural_correlation']:.4f}")

    print(f"\nAdvantage:")
    if vec_utility['snr'] > 0 and mat_utility.get('snr', 0) > 0:
        snr_ratio = vec_utility['snr'] / mat_utility['snr']
        print(f"  Vector SNR is {snr_ratio:.1f}x better!")

    print(f"  Dimensionality reduction: {n**2} → {n} ({n**2/n:.0f}x smaller)")
    print("="*80)
