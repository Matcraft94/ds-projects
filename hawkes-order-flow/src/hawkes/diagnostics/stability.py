"""Stability diagnostics for Hawkes processes."""

import numpy as np
from typing import Optional
from scipy.linalg import eigvals, svdvals
import matplotlib.pyplot as plt


class StabilityDiagnostics:
    """Diagnostics for assessing Hawkes process stability and clustering.
    
    Key metrics:
    - Spectral radius: ρ(B) must be < 1 for stationarity
    - Branching ratio: Mean number of offspring per event
    - Criticality: Distance to critical regime (ρ = 1)
    
    Reference:
        Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015). 
        Hawkes processes in finance. Market Microstructure and Liquidity.
    """
    
    def __init__(self, branching_matrix: np.ndarray):
        """Initialize with branching ratio matrix.
        
        Args:
            branching_matrix: B_ij matrix (expected offspring from j to i)
        """
        self.B = np.array(branching_matrix)
        self.n_dims = self.B.shape[0]
        
        # Compute eigendecomposition
        self.eigenvalues = eigvals(self.B)
        self.spectral_radius = np.max(np.abs(self.eigenvalues))
        
    def is_stable(self, tol: float = 1e-6) -> bool:
        """Check if process is stable (stationary).
        
        A Hawkes process is stable iff ρ(B) < 1.
        
        Args:
            tol: Numerical tolerance
            
        Returns:
            True if stable
        """
        return self.spectral_radius < (1 - tol)
    
    def get_criticality(self) -> float:
        """Measure distance from critical regime.
        
        Returns:
            ρ(B), where values close to 1 indicate near-critical behavior
        """
        return self.spectral_radius
    
    def get_total_branching_ratio(self) -> float:
        """Compute total branching ratio across all dimensions.
        
        This is the expected total number of offspring per event
        (averaged across event types weighted by their frequency).
        """
        return np.sum(self.B)
    
    def get_dimension_branching_ratios(self) -> np.ndarray:
        """Compute branching ratio for each dimension.
        
        Returns:
            Array of shape (n_dims,) with B_i = Σ_j B_ij
        """
        return np.sum(self.B, axis=0)
    
    def get_causal_influence(self) -> np.ndarray:
        """Compute causal influence matrix.
        
        Returns the normalized influence where entry (i,j) represents
        the fraction of events in dimension i caused by dimension j.
        
        Returns:
            Normalized influence matrix
        """
        row_sums = np.sum(self.B, axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        return self.B / row_sums
    
    def get_endogenous_ratio(self) -> float:
        """Compute endogenous vs exogenous event ratio.
        
        The fraction of events that are caused by previous events
        (endogenous) versus background (exogenous).
        
        For stable processes:
            n_endog / n_total = ρ(B) / (1 + ρ(B)) approximately
        """
        n_branching = self.get_total_branching_ratio()
        return n_branching / (1 + n_branching)
    
    def get_memory_horizon(self, threshold: float = 0.95) -> float:
        """Estimate effective memory horizon.
        
        For exponential kernels with mean decay rate β_mean,
        the memory horizon is the time for kernel to decay below threshold.
        
        Args:
            threshold: Cumulative contribution threshold (default 0.95)
            
        Returns:
            Effective memory horizon in time units
        """
        # Estimate from spectral properties
        # This is a heuristic based on mean reversion time
        if self.spectral_radius > 0:
            # Higher spectral radius -> longer memory
            return -np.log(1 - threshold) / (1 - self.spectral_radius)
        return np.inf
    
    def analyze_clustering(self) -> dict:
        """Analyze clustering properties of the process.
        
        Returns:
            Dictionary with clustering metrics
        """
        # Variance-to-mean ratio for cluster sizes
        # For Hawkes: Var[N(t)] / E[N(t)] > 1 indicates overdispersion (clustering)
        
        # Estimate from branching ratio
        n_branch = self.get_total_branching_ratio()
        
        if n_branch < 1:
            # Theoretical variance inflation factor
            variance_factor = 1 / ((1 - n_branch) ** 2)
        else:
            variance_factor = np.inf
        
        return {
            'variance_inflation': variance_factor,
            'is_overdispersed': variance_factor > 1.5,
            'cluster_tendency': 'high' if n_branch > 0.5 else 'low'
        }
    
    def plot_branching_matrix(self, labels: Optional[list] = None) -> plt.Figure:
        """Visualize branching ratio matrix as heatmap.
        
        Args:
            labels: Dimension labels (default: dim_0, dim_1, ...)
            
        Returns:
            Matplotlib figure
        """
        if labels is None:
            labels = [f"dim_{i}" for i in range(self.n_dims)]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Branching matrix heatmap
        im1 = axes[0].imshow(self.B, cmap='YlOrRd', aspect='auto')
        axes[0].set_xticks(range(self.n_dims))
        axes[0].set_yticks(range(self.n_dims))
        axes[0].set_xticklabels(labels, rotation=45)
        axes[0].set_yticklabels(labels)
        axes[0].set_title('Branching Ratio Matrix B_ij')
        axes[0].set_xlabel('Source dimension (j)')
        axes[0].set_ylabel('Target dimension (i)')
        plt.colorbar(im1, ax=axes[0])
        
        # Add text annotations
        for i in range(self.n_dims):
            for j in range(self.n_dims):
                axes[0].text(j, i, f'{self.B[i, j]:.2f}',
                           ha='center', va='center', fontsize=9)
        
        # Causal influence matrix
        influence = self.get_causal_influence()
        im2 = axes[1].imshow(influence, cmap='Blues', aspect='auto')
        axes[1].set_xticks(range(self.n_dims))
        axes[1].set_yticks(range(self.n_dims))
        axes[1].set_xticklabels(labels, rotation=45)
        axes[1].set_yticklabels(labels)
        axes[1].set_title('Causal Influence Matrix')
        axes[1].set_xlabel('Source dimension (j)')
        axes[1].set_ylabel('Target dimension (i)')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        return fig
    
    def plot_stability_analysis(self) -> plt.Figure:
        """Create comprehensive stability analysis plot.
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Eigenvalues in complex plane
        ax = axes[0, 0]
        ax.scatter(self.eigenvalues.real, self.eigenvalues.imag, s=100, alpha=0.6)
        circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', label='Unit circle')
        ax.add_patch(circle)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Real part')
        ax.set_ylabel('Imaginary part')
        ax.set_title(f'Eigenvalues (ρ = {self.spectral_radius:.3f})')
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # 2. Branching ratios by dimension
        ax = axes[0, 1]
        dim_ratios = self.get_dimension_branching_ratios()
        colors = ['green' if r < 1 else 'orange' if r < 1.5 else 'red' for r in dim_ratios]
        bars = ax.bar(range(self.n_dims), dim_ratios, color=colors, alpha=0.7)
        ax.axhline(y=1, color='red', linestyle='--', label='Stability threshold')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Branching ratio')
        ax.set_title('Branching Ratio by Dimension')
        ax.legend()
        ax.set_xticks(range(self.n_dims))
        
        # 3. Singular values
        ax = axes[1, 0]
        singular_values = svdvals(self.B)
        ax.plot(range(1, len(singular_values) + 1), singular_values, 'o-', linewidth=2)
        ax.axhline(y=1, color='red', linestyle='--', label='Threshold')
        ax.set_xlabel('Index')
        ax.set_ylabel('Singular value')
        ax.set_title('Singular Value Decomposition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        Stability Diagnostics Summary
        =============================
        
        Spectral Radius (ρ):     {self.spectral_radius:.4f}
        Stable:                  {'Yes ✓' if self.is_stable() else 'No ✗'}
        
        Total Branching Ratio:   {self.get_total_branching_ratio():.4f}
        Endogenous Ratio:        {self.get_endogenous_ratio():.4f}
        
        Criticality Distance:    {abs(1 - self.spectral_radius):.4f}
        Clustering Tendency:     {self.analyze_clustering()['cluster_tendency']}
        """
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> dict:
        """Generate comprehensive stability report.
        
        Returns:
            Dictionary with all diagnostics
        """
        clustering = self.analyze_clustering()
        
        return {
            'spectral_radius': float(self.spectral_radius),
            'is_stable': self.is_stable(),
            'criticality': self.get_criticality(),
            'total_branching_ratio': float(self.get_total_branching_ratio()),
            'dimension_branching_ratios': self.get_dimension_branching_ratios().tolist(),
            'endogenous_ratio': float(self.get_endogenous_ratio()),
            'clustering': clustering,
            'branching_matrix': self.B.tolist(),
            'eigenvalues': self.eigenvalues.tolist()
        }
