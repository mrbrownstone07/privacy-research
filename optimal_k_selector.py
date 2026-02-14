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
