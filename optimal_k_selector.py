"""
Optimal k Selection for k-NN Graphs
====================================

Determines the best k value for constructing k-NN graphs based on:
1. Graph connectivity (ensure connected graph)
2. Spectral properties (good Fiedler value/gap)
3. Statistical rules of thumb
4. Data dimensionality

Literature:
- von Luxburg (2007): "A tutorial on spectral clustering"
- Maier et al. (2009): "Optimal construction of k-nearest-neighbor graphs"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh
from typing import Tuple, Dict, List
import warnings


class OptimalKSelector:
    """
    Analyzes k-NN graph properties across different k values to recommend optimal k.
    """

    def __init__(self, X: np.ndarray):
        """
        Initialize with dataset.

        Args:
            X: Data matrix (n_samples × n_features)
        """
        self.X = X
        self.n, self.d = X.shape

        print(f"Dataset: n={self.n} samples, d={self.d} features")

        # Standardize data
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

    def analyze_k_range(self, k_min: int = None, k_max: int = None,
                       k_step: int = 1) -> pd.DataFrame:
        """
        Analyze graph properties across a range of k values.

        Args:
            k_min: Minimum k to test (default: log(n))
            k_max: Maximum k to test (default: sqrt(n))
            k_step: Step size for k values

        Returns:
            DataFrame with analysis results for each k
        """
        # Default ranges based on theory
        if k_min is None:
            k_min = max(2, int(np.log(self.n)))
        if k_max is None:
            k_max = min(int(np.sqrt(self.n)), self.n // 3)

        k_values = range(k_min, k_max + 1, k_step)

        results = []

        print(f"\nAnalyzing k ∈ [{k_min}, {k_max}] (step={k_step})...")
        print("-" * 80)

        for k in k_values:
            print(f"k={k:3d} ", end="", flush=True)

            # Build graph
            A = kneighbors_graph(
                self.X_scaled,
                n_neighbors=k,
                mode="connectivity",
                include_self=False
            ).toarray()

            # Symmetrize
            A = np.maximum(A, A.T)

            # Compute Laplacian
            D = np.diag(A.sum(axis=1))
            L = D - A

            # Eigendecomposition
            eigvals, eigvecs = eigh(L)

            # Find connected components (count zero eigenvalues)
            zero_tol = 1e-10
            n_components = np.sum(eigvals < zero_tol)

            # Find Fiedler value (first non-zero eigenvalue)
            if n_components < len(eigvals):
                fiedler_idx = n_components
                fiedler_value = eigvals[fiedler_idx]

                # Spectral gap (to next eigenvalue)
                if fiedler_idx + 1 < len(eigvals):
                    spectral_gap = eigvals[fiedler_idx + 1] - fiedler_value
                else:
                    spectral_gap = 0

                # Relative spectral gap
                rel_gap = spectral_gap / (fiedler_value + 1e-10)
            else:
                fiedler_value = 0
                spectral_gap = 0
                rel_gap = 0

            # Graph statistics
            n_edges = int(A.sum() / 2)
            avg_degree = A.sum(axis=1).mean()
            density = n_edges / (self.n * (self.n - 1) / 2)

            # Condition number estimate (for Fiedler eigenvalue)
            # For symmetric matrices: condition = 1 (theoretically)
            # But we compute empirical sensitivity
            if fiedler_value > zero_tol:
                # Approximate condition using ratio to largest eigenvalue
                condition = eigvals[-1] / fiedler_value
            else:
                condition = np.inf

            results.append({
                'k': k,
                'n_components': n_components,
                'is_connected': n_components == 1,
                'fiedler_value': fiedler_value,
                'spectral_gap': spectral_gap,
                'relative_gap': rel_gap,
                'n_edges': n_edges,
                'avg_degree': avg_degree,
                'density': density,
                'condition_number': condition,
                'eigenvalues': eigvals.copy()
            })

            # Progress indicator
            if n_components == 1:
                print("✓ connected", end=" ")
            else:
                print(f"✗ {n_components} components", end=" ")

            if k % 5 == 0:
                print()

        print("\n" + "-" * 80)

        return pd.DataFrame(results)

    def recommend_k(self, results: pd.DataFrame) -> Dict:
        """
        Recommend optimal k based on multiple criteria.

        Args:
            results: DataFrame from analyze_k_range()

        Returns:
            Dictionary with recommendations and reasoning
        """
        # Filter to connected graphs only
        connected = results[results['is_connected']]

        if len(connected) == 0:
            warnings.warn("No k values produce connected graphs!")
            return {
                'recommended_k': None,
                'reason': 'No connected graphs found',
                'min_k_for_connectivity': results['k'].max() + 1
            }

        recommendations = {}

        # 1. Minimum k for connectivity
        min_k_connected = connected['k'].min()
        recommendations['min_connected'] = {
            'k': min_k_connected,
            'reason': 'Smallest k that ensures connectivity'
        }

        # 2. Maximum spectral gap (best separation)
        idx_max_gap = connected['spectral_gap'].idxmax()
        recommendations['max_spectral_gap'] = {
            'k': connected.loc[idx_max_gap, 'k'],
            'gap': connected.loc[idx_max_gap, 'spectral_gap'],
            'reason': 'Maximizes gap between λ₂ and λ₃ (clearest clustering)'
        }

        # 3. Optimal Fiedler value (moderate connectivity)
        # Want λ₂ not too small (weak) or too large (over-connected)
        # Target: λ₂ ∈ [0.1, 0.5] range
        target_fiedler = 0.3
        connected['fiedler_distance'] = np.abs(connected['fiedler_value'] - target_fiedler)
        idx_target = connected['fiedler_distance'].idxmin()
        recommendations['target_fiedler'] = {
            'k': connected.loc[idx_target, 'k'],
            'fiedler': connected.loc[idx_target, 'fiedler_value'],
            'reason': f'Fiedler value closest to target {target_fiedler}'
        }

        # 4. Elbow method (diminishing returns)
        # Find where adding more edges doesn't improve connectivity much
        if len(connected) >= 3:
            fiedler_vals = connected['fiedler_value'].values
            k_vals = connected['k'].values

            # Compute second derivative (curvature)
            if len(fiedler_vals) >= 3:
                curvature = np.abs(np.diff(fiedler_vals, 2))
                if len(curvature) > 0:
                    elbow_idx = np.argmax(curvature)
                    recommendations['elbow'] = {
                        'k': k_vals[elbow_idx + 1],  # +1 for diff offset
                        'reason': 'Elbow point (diminishing returns)'
                    }

        # 5. Statistical rules of thumb
        # Rule 1: k ≈ log(n)
        k_log = int(np.log(self.n))
        if k_log in connected['k'].values:
            recommendations['log_rule'] = {
                'k': k_log,
                'reason': 'k ≈ log(n) (theoretical rule)'
            }

        # Rule 2: k ≈ sqrt(n)
        k_sqrt = int(np.sqrt(self.n))
        if k_sqrt in connected['k'].values:
            recommendations['sqrt_rule'] = {
                'k': k_sqrt,
                'reason': 'k ≈ √n (practical rule)'
            }

        # Rule 3: Average degree ≈ 2d (data dimensionality)
        target_degree = 2 * self.d
        connected['degree_distance'] = np.abs(connected['avg_degree'] - target_degree)
        idx_degree = connected['degree_distance'].idxmin()
        recommendations['dimension_rule'] = {
            'k': connected.loc[idx_degree, 'k'],
            'avg_degree': connected.loc[idx_degree, 'avg_degree'],
            'reason': f'Average degree ≈ 2d = {target_degree} (dimensionality rule)'
        }

        # 6. OVERALL RECOMMENDATION (weighted scoring)
        # Score each k based on multiple criteria
        scores = pd.Series(0.0, index=connected.index)

        # Connectivity (must have)
        scores += connected['is_connected'].astype(float) * 10

        # Good spectral gap (normalized)
        max_gap = connected['spectral_gap'].max()
        if max_gap > 0:
            scores += (connected['spectral_gap'] / max_gap) * 5

        # Moderate Fiedler value (prefer 0.1-0.5 range)
        fiedler_score = 1 - connected['fiedler_distance'] / connected['fiedler_distance'].max()
        scores += fiedler_score * 3

        # Not too dense (penalize very high k)
        density_penalty = connected['density']
        scores -= density_penalty * 2

        # Best overall
        idx_best = scores.idxmax()
        recommendations['overall_best'] = {
            'k': connected.loc[idx_best, 'k'],
            'score': scores.loc[idx_best],
            'fiedler': connected.loc[idx_best, 'fiedler_value'],
            'gap': connected.loc[idx_best, 'spectral_gap'],
            'density': connected.loc[idx_best, 'density'],
            'reason': 'Best overall score (weighted criteria)'
        }

        return recommendations

    def plot_analysis(self, results: pd.DataFrame, recommendations: Dict):
        """
        Visualize k selection analysis.
        """
        connected = results[results['is_connected']]
        disconnected = results[~results['is_connected']]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Connectivity
        ax = axes[0, 0]
        ax.plot(results['k'], results['n_components'], 'o-', linewidth=2)
        ax.axhline(1, color='green', linestyle='--', label='Connected (1 component)')
        if len(connected) > 0:
            ax.axvline(connected['k'].min(), color='red', linestyle=':',
                      label=f'Min k for connectivity: {connected["k"].min()}')
        ax.set_xlabel('k (neighbors)', fontsize=11)
        ax.set_ylabel('Number of Components', fontsize=11)
        ax.set_title('Graph Connectivity', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Fiedler value
        ax = axes[0, 1]
        if len(connected) > 0:
            ax.plot(connected['k'], connected['fiedler_value'], 'bo-', linewidth=2, label='Fiedler value λ₂')
            ax.axhline(0.3, color='orange', linestyle='--', alpha=0.5, label='Target: 0.3')

            # Highlight recommendation
            if 'overall_best' in recommendations:
                best_k = recommendations['overall_best']['k']
                best_fiedler = recommendations['overall_best']['fiedler']
                ax.plot(best_k, best_fiedler, 'r*', markersize=20, label=f'Recommended: k={best_k}')

        ax.set_xlabel('k (neighbors)', fontsize=11)
        ax.set_ylabel('Fiedler Value λ₂', fontsize=11)
        ax.set_title('Algebraic Connectivity', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Spectral gap
        ax = axes[0, 2]
        if len(connected) > 0:
            ax.plot(connected['k'], connected['spectral_gap'], 'go-', linewidth=2)

            # Highlight max gap
            if 'max_spectral_gap' in recommendations:
                max_k = recommendations['max_spectral_gap']['k']
                max_gap = recommendations['max_spectral_gap']['gap']
                ax.plot(max_k, max_gap, 'r*', markersize=20, label=f'Max gap: k={max_k}')

        ax.set_xlabel('k (neighbors)', fontsize=11)
        ax.set_ylabel('Spectral Gap (λ₃ - λ₂)', fontsize=11)
        ax.set_title('Eigenvalue Separation', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Graph density
        ax = axes[1, 0]
        ax.plot(results['k'], results['density'], 'mo-', linewidth=2)
        ax.set_xlabel('k (neighbors)', fontsize=11)
        ax.set_ylabel('Graph Density', fontsize=11)
        ax.set_title('Edge Density', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 5: Average degree
        ax = axes[1, 1]
        ax.plot(results['k'], results['avg_degree'], 'co-', linewidth=2, label='Avg degree')
        ax.axhline(2 * self.d, color='orange', linestyle='--', label=f'2d = {2*self.d}')
        ax.set_xlabel('k (neighbors)', fontsize=11)
        ax.set_ylabel('Average Degree', fontsize=11)
        ax.set_title('Node Connectivity', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Eigenvalue spectrum for recommended k
        ax = axes[1, 2]
        if 'overall_best' in recommendations and len(connected) > 0:
            best_k = recommendations['overall_best']['k']
            best_row = connected[connected['k'] == best_k].iloc[0]
            eigvals = best_row['eigenvalues']

            ax.plot(eigvals[:20], 'o-', linewidth=2)
            ax.axvline(1, color='red', linestyle='--', label='Fiedler (λ₂)')
            ax.set_xlabel('Eigenvalue Index', fontsize=11)
            ax.set_ylabel('Eigenvalue', fontsize=11)
            ax.set_title(f'Spectrum (k={best_k})', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def analyze_iris_full_dataset():
    """
    Analyze optimal k for full Iris dataset (150 samples).
    """
    # Load Iris data (you'll need to adjust path)
    try:
        df = pd.read_csv("Iris.csv")
    except:
        print("Note: Iris.csv not found, using synthetic data for demonstration")
        # Generate synthetic data similar to Iris
        np.random.seed(42)
        n_samples = 150
        n_features = 4
        X = np.random.randn(n_samples, n_features)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    # Extract features
    feature_cols = [col for col in df.columns if col not in ['Id', 'Species']]
    X = df[feature_cols].values

    print("="*80)
    print("OPTIMAL k SELECTION FOR FULL IRIS DATASET")
    print("="*80)

    # Initialize selector
    selector = OptimalKSelector(X)

    # Analyze k range
    # For n=150: log(150)≈5, sqrt(150)≈12, so test wider range
    results = selector.analyze_k_range(k_min=3, k_max=30, k_step=1)

    # Get recommendations
    recommendations = selector.recommend_k(results)

    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    for name, rec in recommendations.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(f"  k = {rec['k']}")
        print(f"  Reason: {rec['reason']}")
        if 'fiedler' in rec:
            print(f"  Fiedler value: {rec['fiedler']:.4f}")
        if 'gap' in rec:
            print(f"  Spectral gap: {rec['gap']:.4f}")

    # Overall recommendation
    print("\n" + "="*80)
    print("🎯 FINAL RECOMMENDATION")
    print("="*80)
    if 'overall_best' in recommendations:
        best = recommendations['overall_best']
        print(f"Recommended k: {best['k']}")
        print(f"Reasoning: {best['reason']}")
        print(f"\nProperties at k={best['k']}:")
        print(f"  - Fiedler value: {best['fiedler']:.4f} (connectivity strength)")
        print(f"  - Spectral gap: {best['gap']:.4f} (cluster separation)")
        print(f"  - Density: {best['density']:.4f} (edge ratio)")

        # Interpretation
        print(f"\nInterpretation:")
        if best['fiedler'] < 0.2:
            print("  - WEAKLY connected → Easy to partition → Good for clustering")
        elif best['fiedler'] < 0.5:
            print("  - MODERATELY connected → Balanced structure")
        else:
            print("  - STRONGLY connected → Hard to partition → May reduce ML accuracy")

    # Visualize
    fig = selector.plot_analysis(results, recommendations)
    plt.savefig('optimal_k_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to optimal_k_analysis.png")

    # Save detailed results
    results_file = 'k_analysis_results.csv'
    results.drop('eigenvalues', axis=1).to_csv(results_file, index=False)
    print(f"✓ Detailed results saved to k_analysis_results.csv")

    return selector, results, recommendations


if __name__ == "__main__":
    selector, results, recommendations = analyze_iris_full_dataset()
