"""Residual analysis for Hawkes processes.

This module implements diagnostic tools for assessing the goodness-of-fit
of Hawkes process models, including:
- Compensator-based residuals (Ogata's method)
- Q-Q plots for exponentiality testing
- Kolmogorov-Smirnov tests
- Autocorrelation analysis of residuals

Reference:
    Ogata, Y. (1988). Statistical models for earthquake occurrences 
    and residual analysis for point processes. Journal of the American 
    Statistical Association.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import kstest, probplot


class ResidualDiagnostics:
    """Residual diagnostics for Hawkes process goodness-of-fit.
    
    For a well-specified Hawkes model, the compensator-transformed 
    residuals should be i.i.d. exponential(1).
    
    The compensator for dimension i is:
        Λ_i(t) = ∫₀ᵗ λ_i(s) ds
    
    The residuals are:
        r_i^k = Λ_i(t_i^k) - Λ_i(t_i^{k-1})
    
    which should be exponential(1) under the null hypothesis.
    """
    
    def __init__(self, events: list[np.ndarray], end_time: float):
        """Initialize with event data.
        
        Args:
            events: List of event time arrays for each dimension
            end_time: End of observation period
        """
        self.events = events
        self.end_time = end_time
        self.n_dims = len(events)
        self._compensators: Optional[list[np.ndarray]] = None
        self._residuals: Optional[list[np.ndarray]] = None
    
    def compute_compensators(
        self,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> list[np.ndarray]:
        """Compute compensator Λ(t) for each dimension.
        
        For exponential kernels:
            Λ_i(t) = μ_i * t + Σ_j (α_ij/β_ij) * Σ_{t_j^k < t} 
                     [1 - exp(-β_ij * (t - t_j^k))]
        
        Args:
            mu: Baseline intensities
            alpha: Excitation matrix
            beta: Decay matrix
            
        Returns:
            List of compensator values at each event time
        """
        compensators = []
        
        for i in range(self.n_dims):
            events_i = self.events[i]
            n_events = len(events_i)
            
            if n_events == 0:
                compensators.append(np.array([]))
                continue
            
            comp_values = np.zeros(n_events)
            
            for k, t_k in enumerate(events_i):
                # Baseline contribution
                comp = mu[i] * t_k
                
                # Kernel contributions from all past events
                for j in range(self.n_dims):
                    events_j = self.events[j]
                    past_events = events_j[events_j < t_k]
                    
                    if len(past_events) > 0 and beta[i, j] > 0:
                        # ∫_0^{t_k - t_j} α*exp(-β*t) dt 
                        # = (α/β) * (1 - exp(-β*(t_k - t_j)))
                        dt = t_k - past_events
                        # Safe division: only compute if beta > 0
                        ratio = alpha[i, j] / beta[i, j]
                        comp += np.sum(ratio * (1 - np.exp(-beta[i, j] * dt)))
                
                comp_values[k] = comp
            
            compensators.append(comp_values)
        
        self._compensators = compensators
        return compensators
    
    def compute_residuals(
        self,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> list[np.ndarray]:
        """Compute inter-arrival residuals via time rescaling.
        
        Uses Ogata's time rescaling theorem:
            r_i^k = Λ_i(t_i^k) - Λ_i(t_i^{k-1})
        
        Under H0 (correctly specified model), r_i^k ~ Exp(1).
        
        Args:
            mu: Baseline intensities
            alpha: Excitation matrix
            beta: Decay matrix
            
        Returns:
            List of residual arrays for each dimension
        """
        compensators = self.compute_compensators(mu, alpha, beta)
        residuals = []
        
        for comp in compensators:
            if len(comp) <= 1:
                residuals.append(np.array([]))
            else:
                # Differences: Λ(t_k) - Λ(t_{k-1})
                # Should be positive since compensator is monotonically increasing
                res = np.diff(comp)
                
                # Filter out non-positive values (numerical errors)
                res = res[res > 0]
                
                if len(res) == 0:
                    # If no positive residuals, use normalized inter-arrival times
                    # as fallback (approximate)
                    res = np.diff(comp)
                    res = np.abs(res)  # Take absolute value as last resort
                    res = res[res > 0]
                
                residuals.append(res)
        
        self._residuals = residuals
        return residuals
    
    def ks_test(self, residuals: Optional[list[np.ndarray]] = None) -> dict:
        """Kolmogorov-Smirnov test for exponential(1) distribution.
        
        Tests H0: residuals ~ Exp(1)
        
        Args:
            residuals: Pre-computed residuals. If None, uses stored values.
            
        Returns:
            Dictionary with test results for each dimension
        """
        if residuals is None:
            residuals = self._residuals
        
        if residuals is None:
            raise ValueError("Must compute residuals first")
        
        results = []
        for i, res in enumerate(residuals):
            if len(res) > 0:
                # Test against exponential(1)
                statistic, pvalue = kstest(res, 'expon', args=(0, 1))
                results.append({
                    'dimension': i,
                    'n_observations': len(res),
                    'ks_statistic': statistic,
                    'p_value': pvalue,
                    'reject_h0': pvalue < 0.05,
                    'mean_residual': np.mean(res),
                    'std_residual': np.std(res)
                })
        
        return {
            'by_dimension': results,
            'overall_pvalue': np.mean([r['p_value'] for r in results]) if results else None
        }
    
    def ljung_box_test(self, residuals: Optional[list[np.ndarray]] = None, 
                       lags: int = 10) -> dict:
        """Ljung-Box test for autocorrelation in residuals.
        
        Tests H0: residuals are independently distributed (no autocorrelation)
        
        Args:
            residuals: Pre-computed residuals
            lags: Number of lags to test
            
        Returns:
            Dictionary with test results
        """
        if residuals is None:
            residuals = self._residuals
        
        if residuals is None:
            raise ValueError("Must compute residuals first")
        
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        results = []
        for i, res in enumerate(residuals):
            if len(res) > lags + 1:
                lb_test = acorr_ljungbox(res, lags=lags, return_df=True)
                results.append({
                    'dimension': i,
                    'lb_statistic': lb_test['lb_stat'].iloc[-1],
                    'p_value': lb_test['lb_pvalue'].iloc[-1],
                    'reject_h0': lb_test['lb_pvalue'].iloc[-1] < 0.05
                })
        
        return {'by_dimension': results}
    
    def plot_diagnostics(self, residuals: Optional[list[np.ndarray]] = None,
                        figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Create comprehensive residual diagnostic plots.
        
        Plots:
        1. Q-Q plot vs exponential(1)
        2. Histogram with exponential fit
        3. Autocorrelation function
        4. Cumulative residuals
        
        Args:
            residuals: Pre-computed residuals
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if residuals is None:
            residuals = self._residuals
        
        if residuals is None:
            raise ValueError("Must compute residuals first")
        
        n_dims = len(residuals)
        fig, axes = plt.subplots(n_dims, 3, figsize=figsize)
        
        if n_dims == 1:
            axes = axes.reshape(1, -1)
        
        for i, res in enumerate(residuals):
            if len(res) < 2:
                continue
            
            # Q-Q plot
            probplot(res, dist=stats.expon, sparams=(0, 1), 
                    plot=axes[i, 0])
            axes[i, 0].set_title(f'Dim {i}: Q-Q vs Exp(1)')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Histogram with exponential fit
            axes[i, 1].hist(res, bins=30, density=True, alpha=0.7, 
                           label='Residuals')
            x_range = np.linspace(0, max(res), 100)
            axes[i, 1].plot(x_range, stats.expon.pdf(x_range, 0, 1),
                           'r-', linewidth=2, label='Exp(1)')
            axes[i, 1].set_title(f'Dim {i}: Residual Distribution')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
            
            # Autocorrelation
            from statsmodels.tsa.stattools import acf
            acf_vals = acf(res, nlags=min(20, len(res)//2), fft=True)
            axes[i, 2].bar(range(len(acf_vals)), acf_vals)
            axes[i, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[i, 2].axhline(y=1.96/np.sqrt(len(res)), color='r', 
                              linestyle='--', alpha=0.5, label='95% CI')
            axes[i, 2].axhline(y=-1.96/np.sqrt(len(res)), color='r', 
                              linestyle='--', alpha=0.5)
            axes[i, 2].set_title(f'Dim {i}: Autocorrelation')
            axes[i, 2].set_xlabel('Lag')
            axes[i, 2].legend()
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, mu: np.ndarray, alpha: np.ndarray, 
                       beta: np.ndarray) -> dict:
        """Generate comprehensive diagnostic report.
        
        Args:
            mu: Baseline intensities
            alpha: Excitation matrix
            beta: Decay matrix
            
        Returns:
            Dictionary with all diagnostic results
        """
        # Compute residuals
        residuals = self.compute_residuals(mu, alpha, beta)
        
        # Run tests
        ks_results = self.ks_test(residuals)
        lb_results = self.ljung_box_test(residuals)
        
        # Summary statistics
        summary = {
            'n_events_by_dim': [len(e) for e in self.events],
            'n_residuals_by_dim': [len(r) for r in residuals],
            'mean_residuals': [np.mean(r) if len(r) > 0 else None 
                              for r in residuals],
            'variance_residuals': [np.var(r) if len(r) > 0 else None 
                                  for r in residuals],
        }
        
        # Goodness of fit assessment
        # Under H0: mean = 1, var = 1 for Exp(1)
        gof_assessment = []
        for i, r in enumerate(residuals):
            if len(r) > 0:
                mean_deviation = abs(np.mean(r) - 1.0)
                var_deviation = abs(np.var(r) - 1.0)
                gof_assessment.append({
                    'dimension': i,
                    'mean_ok': mean_deviation < 0.1,
                    'var_ok': var_deviation < 0.2,
                    'ks_pass': ks_results['by_dimension'][i]['reject_h0'] == False
                               if i < len(ks_results['by_dimension']) else None,
                    'overall': mean_deviation < 0.1 and var_deviation < 0.2
                })
        
        return {
            'summary': summary,
            'ks_test': ks_results,
            'ljung_box': lb_results,
            'goodness_of_fit': gof_assessment,
            'model_valid': all(g['overall'] for g in gof_assessment) if gof_assessment else False
        }
