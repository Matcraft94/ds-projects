"""Baseline models for comparison with Hawkes processes.

Implements simpler models that serve as baselines:
- Homogeneous Poisson process
- Inhomogeneous Poisson process (piecewise constant)
- Renewal processes

These allow us to quantify the value added by Hawkes self-excitation.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Optional, Tuple
import pandas as pd


class HomogeneousPoisson:
    """Homogeneous Poisson process baseline.
    
    Simplest baseline with constant intensity λ.
    Log-likelihood: ll = Σᵢ log(λ) - λT = N log(λ) - λT
    MLE: λ̂ = N / T
    """
    
    def __init__(self, n_dims: int):
        """Initialize.
        
        Args:
            n_dims: Number of dimensions
        """
        self.n_dims = n_dims
        self.lambda_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
    
    def fit(self, events: list[np.ndarray], end_time: float):
        """Fit homogeneous Poisson.
        
        Args:
            events: Event data
            end_time: End of observation
        """
        self.lambda_ = np.array([len(e) / end_time for e in events])
        
        # Log-likelihood
        ll = 0.0
        for i in range(self.n_dims):
            n_i = len(events[i])
            if n_i > 0:
                ll += n_i * np.log(self.lambda_[i]) - self.lambda_[i] * end_time
        
        self.log_likelihood_ = ll
        return self
    
    def compute_aic(self) -> float:
        """Compute AIC."""
        n_params = self.n_dims  # One lambda per dimension
        return -2 * self.log_likelihood_ + 2 * n_params
    
    def compute_bic(self, n_events: int) -> float:
        """Compute BIC."""
        n_params = self.n_dims
        return -2 * self.log_likelihood_ + n_params * np.log(n_events)
    
    def predict_intensity(self, times: np.ndarray) -> np.ndarray:
        """Predict intensity (constant)."""
        return np.tile(self.lambda_, (len(times), 1)).T


class PiecewisePoisson:
    """Piecewise constant Poisson (inhomogeneous).
    
    Divides time into bins with constant intensity in each bin.
    """
    
    def __init__(self, n_dims: int, n_bins: int = 10):
        """Initialize.
        
        Args:
            n_dims: Number of dimensions
            n_bins: Number of time bins
        """
        self.n_dims = n_dims
        self.n_bins = n_bins
        self.bins: Optional[np.ndarray] = None
        self.intensities: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
    
    def fit(self, events: list[np.ndarray], end_time: float):
        """Fit piecewise Poisson.
        
        Args:
            events: Event data
            end_time: End of observation
        """
        self.bins = np.linspace(0, end_time, self.n_bins + 1)
        self.intensities = np.zeros((self.n_dims, self.n_bins))
        
        ll = 0.0
        
        for i in range(self.n_dims):
            for b in range(self.n_bins):
                bin_start = self.bins[b]
                bin_end = self.bins[b + 1]
                bin_width = bin_end - bin_start
                
                # Count events in bin
                n_bin = np.sum((events[i] >= bin_start) & (events[i] < bin_end))
                
                # MLE intensity
                self.intensities[i, b] = n_bin / bin_width if bin_width > 0 else 0
                
                # Log-likelihood
                if n_bin > 0:
                    ll += n_bin * np.log(self.intensities[i, b] + 1e-10)
                ll -= self.intensities[i, b] * bin_width
        
        self.log_likelihood_ = ll
        return self
    
    def compute_aic(self) -> float:
        """Compute AIC."""
        n_params = self.n_dims * self.n_bins
        return -2 * self.log_likelihood_ + 2 * n_params
    
    def compute_bic(self, n_events: int) -> float:
        """Compute BIC."""
        n_params = self.n_dims * self.n_bins
        return -2 * self.log_likelihood_ + n_params * np.log(n_events)


class ModelComparison:
    """Compare Hawkes models against baselines.
    
    Provides likelihood ratio tests and information criteria comparisons
    to quantify the value of self-excitation.
    """
    
    def __init__(self, events: list[np.ndarray], end_time: float):
        """Initialize.
        
        Args:
            events: Event data
            end_time: End of observation
        """
        self.events = events
        self.end_time = end_time
        self.n_dims = len(events)
        self.total_events = sum(len(e) for e in events)
        self.results: Optional[pd.DataFrame] = None
    
    def fit_all(self, hawkes_estimator) -> pd.DataFrame:
        """Fit all baseline models and compare with Hawkes.
        
        Args:
            hawkes_estimator: Fitted Hawkes estimator
            
        Returns:
            Comparison DataFrame
        """
        results = []
        
        # 1. Homogeneous Poisson
        poisson_hom = HomogeneousPoisson(self.n_dims)
        poisson_hom.fit(self.events, self.end_time)
        
        results.append({
            'model': 'Homogeneous Poisson',
            'n_params': self.n_dims,
            'log_likelihood': poisson_hom.log_likelihood_,
            'aic': poisson_hom.compute_aic(),
            'bic': poisson_hom.compute_bic(self.total_events),
            'description': 'Constant intensity, no self-excitation'
        })
        
        # 2. Piecewise Poisson (10 bins)
        poisson_piece = PiecewisePoisson(self.n_dims, n_bins=10)
        poisson_piece.fit(self.events, self.end_time)
        
        results.append({
            'model': 'Piecewise Poisson (10 bins)',
            'n_params': self.n_dims * 10,
            'log_likelihood': poisson_piece.log_likelihood_,
            'aic': poisson_piece.compute_aic(),
            'bic': poisson_piece.compute_bic(self.total_events),
            'description': 'Time-varying intensity, no self-excitation'
        })
        
        # 3. Hawkes model
        hawkes_ll = hawkes_estimator.log_likelihood_
        hawkes_n_params = self.n_dims + self.n_dims**2 * 2  # mu, alpha, beta
        
        results.append({
            'model': 'Hawkes (Exponential)',
            'n_params': hawkes_n_params,
            'log_likelihood': hawkes_ll,
            'aic': -2 * hawkes_ll + 2 * hawkes_n_params,
            'bic': -2 * hawkes_ll + hawkes_n_params * np.log(self.total_events),
            'description': 'Self-exciting process with exponential kernel'
        })
        
        self.results = pd.DataFrame(results)
        
        # Add comparison metrics
        self.results['likelihood_ratio_vs_poisson'] = (
            2 * (self.results['log_likelihood'] - poisson_hom.log_likelihood_)
        )
        self.results['delta_aic'] = self.results['aic'] - self.results['aic'].min()
        self.results['delta_bic'] = self.results['bic'] - self.results['bic'].min()
        
        # Akaike weights
        self.results['akaike_weight'] = np.exp(
            -0.5 * self.results['delta_aic']
        ) / np.sum(np.exp(-0.5 * self.results['delta_aic']))
        
        return self.results
    
    def likelihood_ratio_test(
        self,
        null_model: str = 'Homogeneous Poisson',
        alternative_model: str = 'Hawkes (Exponential)'
    ) -> dict:
        """Likelihood ratio test between nested models.
        
        Tests H0: Poisson (no self-excitation) vs H1: Hawkes
        
        Args:
            null_model: Name of null model
            alternative_model: Name of alternative model
            
        Returns:
            Test results dictionary
        """
        if self.results is None:
            raise ValueError("Must run fit_all first")
        
        null_row = self.results[self.results['model'] == null_model].iloc[0]
        alt_row = self.results[self.results['model'] == alternative_model].iloc[0]
        
        # Likelihood ratio statistic
        lr_stat = 2 * (alt_row['log_likelihood'] - null_row['log_likelihood'])
        
        # Degrees of freedom
        df = alt_row['n_params'] - null_row['n_params']
        
        # P-value (chi-square)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr_stat, df)
        
        return {
            'lr_statistic': lr_stat,
            'df': df,
            'p_value': p_value,
            'reject_null': p_value < 0.05,
            'null_model': null_model,
            'alternative_model': alternative_model,
            'conclusion': (
                'Reject H0: Self-excitation is significant' if p_value < 0.05 
                else 'Fail to reject H0: No significant self-excitation'
            )
        }
    
    def print_summary(self):
        """Print formatted comparison summary."""
        if self.results is None:
            print("Must run fit_all first")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        for _, row in self.results.iterrows():
            print(f"\n{row['model']}:")
            print(f"  {row['description']}")
            print(f"  Parameters: {row['n_params']}")
            print(f"  Log-likelihood: {row['log_likelihood']:.2f}")
            print(f"  AIC: {row['aic']:.2f} (ΔAIC: {row['delta_aic']:.2f})")
            print(f"  BIC: {row['bic']:.2f} (ΔBIC: {row['delta_bic']:.2f})")
            print(f"  Akaike weight: {row['akaike_weight']:.3f}")
        
        # Best model by each criterion
        best_aic = self.results.loc[self.results['aic'].idxmin(), 'model']
        best_bic = self.results.loc[self.results['bic'].idxmin(), 'model']
        best_ll = self.results.loc[self.results['log_likelihood'].idxmax(), 'model']
        
        print("\n" + "-"*80)
        print("BEST MODELS:")
        print(f"  By AIC: {best_aic}")
        print(f"  By BIC: {best_bic}")
        print(f"  By Log-likelihood: {best_ll}")
        
        # Likelihood ratio test
        lrt = self.likelihood_ratio_test()
        print("\n" + "-"*80)
        print("LIKELIHOOD RATIO TEST (Poisson vs Hawkes):")
        print(f"  LR statistic: {lrt['lr_statistic']:.2f}")
        print(f"  Degrees of freedom: {lrt['df']}")
        print(f"  P-value: {lrt['p_value']:.2e}")
        print(f"  Conclusion: {lrt['conclusion']}")
        print("="*80)
